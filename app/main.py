"""FastAPI 应用：建卡作业、版本查询、在线打分（仅 HTTP，无前端）。

建卡为后台作业：POST /cards/{name}/build 上传 CSV，作业开始前的校验
（样本量/标签/PDO/特征存在性等）同步完成，失败直接 400 且不产生作业；
校验通过才入队，通过 GET /jobs/{id} 轮询成败与失败原因。
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .audit.service import IdempotentConflict
from .core.exceptions import ValidationError
from .core.pipeline import BuildParams
from .core.sample import parse_csv
from .deps import audit, backfills, monitoring, performance, repo, runtimes, scheduler
from .api.schemas import (
    BackfillEnqueueResponse,
    BackfillRequest,
    BatchScoreRequest,
    BatchScoreResponse,
    BuildResponse,
    CardOut,
    JobOut,
    ScoreResponse,
)
from .performance.scheduler import validate_items


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    repo.init_schema()
    yield
    scheduler._pool.shutdown(wait=True)
    backfills._pool.shutdown(wait=True)


app = FastAPI(title="内部评分卡服务", version="1.0.0", lifespan=lifespan)


@app.exception_handler(ValidationError)
def _validation_handler(request, exc: ValidationError):  # noqa: ANN001
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------- 建卡

@app.post("/cards/{card_name}/build", response_model=BuildResponse, status_code=202)
async def build_card(
    card_name: str,
    file: UploadFile = File(..., description="开发样本 CSV"),
    label_col: str = Form("label"),
    features: str | None = Form(
        None, description="逗号分隔的入模特征清单；留空按 IV 阈值自动筛选"),
    iv_threshold: float = Form(0.02),
    max_bins: int = Form(20),
    min_bin_pct: float = Form(0.05),
    base_score: float = Form(600.0),
    base_odds: float = Form(20.0),
    pdo: float = Form(50.0),
):
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, f"CSV 必须是 UTF-8 编码：{exc}")

    feature_list = None
    if features is not None and features.strip():
        feature_list = [f.strip() for f in features.split(",") if f.strip()]

    params = BuildParams(
        label_col=label_col,
        features=feature_list,
        iv_threshold=iv_threshold,
        max_bins=max_bins,
        min_bin_pct=min_bin_pct,
        base_score=base_score,
        base_odds=base_odds,
        pdo=pdo,
    )

    # 作业开始前同步校验：解析 + 全部前置规则，失败不产生作业
    try:
        sample = parse_csv(text, label_col=label_col)
        params.validate(sample)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    repo.ensure_card(card_name)
    payload = {
        "filename": file.filename,
        "params": {
            "label_col": label_col,
            "features": feature_list,
            "iv_threshold": iv_threshold,
            "max_bins": max_bins,
            "min_bin_pct": min_bin_pct,
            "base_score": base_score,
            "base_odds": base_odds,
            "pdo": pdo,
        },
    }
    job_id = repo.create_job(card_name, payload)
    scheduler.submit(job_id, card_name, text, params)
    return BuildResponse(job_id=job_id, card_name=card_name)


@app.get("/cards", response_model=list[CardOut])
def get_cards():
    return repo.list_cards()


@app.get("/cards/{card_name}/versions")
def get_versions(card_name: str):
    vers = repo.list_versions(card_name)
    if not vers and not any(c["card_name"] == card_name for c in repo.list_cards()):
        raise HTTPException(404, f"卡 {card_name!r} 不存在")
    return vers


@app.get("/cards/{card_name}/versions/{version}")
def get_version_detail(card_name: str, version: int):
    summary = repo.get_version_summary(card_name, version)
    if summary is None:
        raise HTTPException(404, f"卡 {card_name!r} 版本 {version} 不存在")
    return summary


@app.get("/cards/{card_name}/versions/{version}/artifact")
def get_version_artifact(card_name: str, version: int):
    """完整建卡产物：分箱边界/好坏计数/WOE/IV/合并日志/迭代历史/分值表全量。"""
    try:
        return repo.load_version(card_name, version)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int):
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"作业 {job_id} 不存在")
    return JobOut(
        id=job["id"], card_name=job["card_name"], status=job["status"],
        version=job.get("version"), error=job.get("error"),
        newton_history=job.get("newton_history"),
    )


@app.get("/jobs")
def list_jobs(card_name: str | None = None):
    return [
        {k: v for k, v in j.items() if k not in ("request",)}
        for j in repo.list_jobs(card_name)
    ]


# ---------------------------------------------------------------- 打分

def _load_runtime(card_name: str, version: int | None):
    try:
        return runtimes.get(card_name, version)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/cards/{card_name}/score", response_model=ScoreResponse)
def score_applicant(card_name: str, body: dict):
    version = body.get("version")
    features = body.get("features")
    if not isinstance(features, dict):
        raise HTTPException(400, "features 必须是对象：{特征名: 原始值}")
    request_id = body.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise HTTPException(400, "request_id 必须是字符串")
    ver, rt = _load_runtime(card_name, version)
    try:
        result, _replayed = audit.score_single(
            card_name, ver, rt, features, request_id)
    except IdempotentConflict:
        raise HTTPException(409, f"请求标识 {request_id!r} 已用于不同内容，"
                                 "拒绝覆盖原记录")
    except ValueError as exc:
        raise HTTPException(400, f"该申请人打分失败：{exc}")
    # 原有响应字段与数值不变：card_name/version 在外层，其余来自打分结果
    return {"card_name": card_name, **result}


@app.post("/cards/{card_name}/score/batch", response_model=BatchScoreResponse)
def score_batch(card_name: str, req: BatchScoreRequest):
    ver, rt = _load_runtime(card_name, req.version)
    applicants = [item.model_dump() for item in req.applicants]
    results = audit.score_batch(card_name, ver, rt, applicants)
    ok_cnt = sum(1 for r in results if r["ok"])
    fail_cnt = sum(1 for r in results if not r["ok"])
    replay_cnt = sum(1 for r in results if r["replayed"])
    conflict_cnt = sum(1 for r in results if r["conflict"])
    return BatchScoreResponse(
        card_name=card_name, version=ver, results=results,
        succeeded=ok_cnt, failed=fail_cnt,
        replayed=replay_cnt, conflicts=conflict_cnt,
    )


# ---------------------------------------------------------------- 监控查询

def _parse_ts(value: str | None, field: str) -> datetime | None:
    """ISO8601 时间；不带时区一律按 UTC 解释（与存储层一致）。"""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(400, f"{field} 不是合法 ISO8601 时间：{value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _check_version(card_name: str, version: int) -> None:
    if repo.get_version_summary(card_name, version) is None:
        raise HTTPException(404, f"卡 {card_name!r} 版本 {version} 不存在")


@app.get("/cards/{card_name}/versions/{version}/stability")
def get_stability(card_name: str, version: int,
                  start: str | None = None, end: str | None = None):
    _check_version(card_name, version)
    t0, t1 = _parse_ts(start, "start"), _parse_ts(end, "end")
    if t0 is not None and t1 is not None and t1 <= t0:
        raise HTTPException(400, "end 必须晚于 start")
    return monitoring.stability(card_name, version, t0, t1)


@app.get("/cards/{card_name}/versions/{version}/performance")
def get_performance(card_name: str, version: int,
                    start: str | None = None, end: str | None = None,
                    n_bands: int = 10):
    _check_version(card_name, version)
    if n_bands < 2 or n_bands > 50:
        raise HTTPException(400, "n_bands 必须在 2..50 之间")
    t0, t1 = _parse_ts(start, "start"), _parse_ts(end, "end")
    if t0 is not None and t1 is not None and t1 <= t0:
        raise HTTPException(400, "end 必须晚于 start")
    return performance.report(card_name, version, t0, t1, n_bands=n_bands)


@app.get("/cards/{card_name}/versions/{version}/score-logs")
def get_score_logs(card_name: str, version: int,
                   start: str | None = None, end: str | None = None,
                   limit: int = 1000):
    """留痕查询（对账用）：按时间区间列出该版本的打分记录。"""
    _check_version(card_name, version)
    limit = min(max(limit, 1), 10_000)
    t0, t1 = _parse_ts(start, "start"), _parse_ts(end, "end")
    return {
        "card_name": card_name, "version": version,
        "limit": limit,
        "logs": repo.query_score_logs(card_name, version, t0, t1, limit),
    }


# ---------------------------------------------------------------- 表现回填

@app.post("/cards/{card_name}/performance/backfill",
          response_model=BackfillEnqueueResponse, status_code=202)
def enqueue_backfill(card_name: str, req: BackfillRequest):
    if not repo.list_versions(card_name):
        raise HTTPException(404, f"卡 {card_name!r} 不存在或尚无成功版本")
    raw = [item.model_dump() for item in req.items]
    valid, invalid = validate_items(raw)
    if not raw:
        raise HTTPException(400, "items 不能为空")
    payload = {"items": valid}
    job_id = repo.create_backfill_job(card_name, payload)
    backfills.submit(job_id, card_name, valid)
    return BackfillEnqueueResponse(
        job_id=job_id, card_name=card_name, total=len(valid),
        invalid_rejected=len(invalid), invalid_rejected_items=invalid,
    )


@app.get("/performance/backfills/{job_id}")
def get_backfill_job(job_id: int):
    job = repo.get_backfill_job(job_id)
    if job is None:
        raise HTTPException(404, f"回填作业 {job_id} 不存在")
    return job


@app.get("/performance/backfills")
def list_backfill_jobs(card_name: str | None = None):
    return repo.list_backfill_jobs(card_name)


@app.get("/health")
def health():
    return {"status": "ok"}
