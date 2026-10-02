"""监控层 HTTP 路由（接口层只做解析/装配，逻辑都在 app.monitor 与仓储里）。

- GET  .../stability        人群稳定性（逐特征 + 总分 PSI/占比对比 + 三档结论）
- POST .../score-baseline   老版本用原始建卡样本补总分基准（严格校验后才生效）
- POST .../backfills        上传表现回填，后台作业
- GET  .../backfills/{id}   回填进度与拒收清单
- GET  .../performance      已回填放款的坏率/平均PD/KS/AUC + 总分分段对比

时间参数：start/end 为 ISO 8601（日期或日期时间，日期按 UTC 00:00 处理），
区间半开 [start, end)，打分时间（scored_at, UTC）落在其中才计入。
不同版本的留痕在查询中各算各的（version 必填）。
"""
from __future__ import annotations

from datetime import datetime, time, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..core.config import settings
from ..core.sample import parse_csv
from ..deps import backfills, repo
from ..monitor.baseline import build_score_baseline, verify_feature_counts
from ..monitor.performance import performance_report
from ..monitor.stability import stability_report
from ..api.schemas import BackfillRequest

router = APIRouter()


def _parse_bound(value: str | None, which: str) -> datetime | None:
    if value is None:
        return None
    try:
        if len(value) == 10:  # YYYY-MM-DD
            d = datetime.strptime(value, "%Y-%m-%d")
            if which == "end":
                return datetime.combine(d.date(), time.max, timezone.utc)
            return datetime.combine(d.date(), time.min, timezone.utc)
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(400, f"{which} 时间格式非法（需 ISO 8601）：{exc}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_version(card_name: str, version: int) -> dict:
    artifacts = repo.load_version(card_name, version)
    return artifacts


def _check_version_exists(card_name: str, version: int) -> None:
    if repo.get_version_summary(card_name, version) is None:
        raise HTTPException(404, f"卡 {card_name!r} 版本 {version} 不存在")


# ------------------------------------------------------------- 人群稳定性
@router.get("/cards/{card_name}/versions/{version}/stability")
def get_stability(
    card_name: str,
    version: int,
    start: str | None = None,
    end: str | None = None,
    warning: float = settings.psi_warning,
    significant: float = settings.psi_significant,
):
    _check_version_exists(card_name, version)
    if not (0 < warning < significant):
        raise HTTPException(400, "阈值需满足 0 < warning < significant")
    start_dt = _parse_bound(start, "start")
    end_dt = _parse_bound(end, "end")
    if start_dt and end_dt and end_dt <= start_dt:
        raise HTTPException(400, "end 必须晚于 start")
    records = repo.list_score_records(card_name, version, start_dt, end_dt)
    artifacts = _resolve_version(card_name, version)
    report = stability_report(artifacts, records,
                              warn=warning, alarm=significant)
    report["card_name"] = card_name
    report["version"] = version
    report["start"] = start_dt.isoformat() if start_dt else None
    report["end"] = end_dt.isoformat() if end_dt else None
    return report


# ----------------------------------------------------- 老版本补总分基准
@router.post("/cards/{card_name}/versions/{version}/score-baseline",
             status_code=200)
async def supplement_score_baseline(
    card_name: str,
    version: int,
    file: UploadFile = File(..., description="该版本的原始建卡样本 CSV"),
    label_col: str = Form("label"),
    n_bins: int = Form(settings.score_baseline_bins),
):
    """升级前的老版本产物里没有总分基准。若原始建卡样本仍在，可用本接口补：

    1) 用在线引擎逐行落箱核对**每个入模特征每箱（含缺失箱）计数**与产物
       完全一致（防止拿错样本），样本量也要一致；任何不一致整体 400 拒绝；
    2) 校验通过才写入总分基准（只新增 artifacts.score_baseline，不动其它产物）。
    """
    _check_version_exists(card_name, version)
    if n_bins < 2:
        raise HTTPException(400, "n_bins 至少为 2")
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, f"CSV 必须是 UTF-8 编码：{exc}")
    artifacts = _resolve_version(card_name, version)
    if artifacts.get("score_baseline"):
        raise HTTPException(
            409, "该版本已存在总分基准，基准不可覆盖（如需重算请联系管理员清除）")
    try:
        sample = parse_csv(text, label_col=label_col)
    except Exception as exc:
        raise HTTPException(400, f"建卡样本解析失败：{exc}")

    n_train = artifacts["data_summary"]["n"]
    if sample.n != n_train:
        raise HTTPException(
            400, f"样本量 {sample.n} 与建卡样本量 {n_train} 不一致，拒绝补录")
    rows = _rows_from_sample(sample)
    ok, diffs = verify_feature_counts(artifacts, rows)
    if not ok:
        raise HTTPException(
            400, "样本逐特征分箱计数与建卡产物不一致，拒绝补录："
                 + "；".join(diffs[:10])
                 + (f"（等共 {len(diffs)} 处）" if len(diffs) > 10 else ""))
    baseline = build_score_baseline(artifacts, rows, n_bins=n_bins)
    repo.set_score_baseline(card_name, version, baseline)
    return {"status": "ok", "card_name": card_name, "version": version,
            "n": baseline["n"], "bins": len(baseline["counts"]),
            "edges": baseline["edges"]}


def _rows_from_sample(sample) -> list[dict]:
    import numpy as np
    rows: list[dict] = [{} for _ in range(sample.n)]
    for name in sample.feature_names:
        col = sample.features[name]
        if sample.types[name] == "numeric":
            for i, v in enumerate(col):
                rows[i][name] = None if np.isnan(v) else float(v)
        else:
            for i, v in enumerate(col):
                rows[i][name] = v
    return rows


# ------------------------------------------------------------- 表现回填
@router.post("/cards/{card_name}/backfills", status_code=202)
def create_backfill(card_name: str, req: BackfillRequest):
    if not any(c["card_name"] == card_name for c in repo.list_cards()):
        raise HTTPException(404, f"卡 {card_name!r} 不存在")
    # 标签非法在这里不做整体拒绝：逐条进拒收清单（需求约定）
    items = [{"request_id": it.request_id, "label": it.label}
             for it in req.items]
    job_id = repo.create_backfill_job(card_name, items)
    backfills.submit(job_id, card_name, items)
    return {"job_id": job_id, "card_name": card_name,
            "total": len(items), "status": "pending"}


@router.get("/cards/{card_name}/backfills/{job_id}")
def get_backfill(card_name: str, job_id: int):
    job = repo.get_backfill_job(job_id)
    if job is None or job["card_name"] != card_name:
        raise HTTPException(404, f"卡 {card_name!r} 的回填作业 {job_id} 不存在")
    return job


@router.get("/cards/{card_name}/backfills")
def list_backfills(card_name: str):
    return repo.list_backfill_jobs(card_name)


# ------------------------------------------------------------- 表现分析
@router.get("/cards/{card_name}/versions/{version}/performance")
def get_performance(
    card_name: str,
    version: int,
    start: str | None = None,
    end: str | None = None,
):
    """只统计区间内**已回填标签**的留痕；KS/AUC 直接复用建卡同一套 metrics。"""
    _check_version_exists(card_name, version)
    start_dt = _parse_bound(start, "start")
    end_dt = _parse_bound(end, "end")
    if start_dt and end_dt and end_dt <= start_dt:
        raise HTTPException(400, "end 必须晚于 start")
    records = repo.list_score_records(
        card_name, version, start_dt, end_dt, labeled=True)
    rows = [{"total_score": r["total_score"], "pd": r["pd"],
             "label": r["label"]} for r in records]
    artifacts = _resolve_version(card_name, version)
    report = performance_report(rows, artifacts.get("score_baseline"))
    report["card_name"] = card_name
    report["version"] = version
    report["start"] = start_dt.isoformat() if start_dt else None
    report["end"] = end_dt.isoformat() if end_dt else None
    return report
