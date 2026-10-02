"""FastAPI 应用：建卡作业、版本查询、在线打分（仅 HTTP，无前端）。

建卡为后台作业：POST /cards/{name}/build 上传 CSV，作业开始前的校验
（样本量/标签/PDO/特征存在性等）同步完成，失败直接 400 且不产生作业；
校验通过才入队，通过 GET /jobs/{id} 轮询成败与失败原因。
"""
from __future__ import annotations

import contextlib

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .core.exceptions import ValidationError
from .core.pipeline import BuildParams
from .core.sample import parse_csv
from .deps import auditor, backfills, repo, runtimes, scheduler
from .api.monitor import router as monitor_router
from .api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    BuildResponse,
    CardOut,
    JobOut,
    ScoreResponse,
)
from .monitor.audit import InvalidRequestId, RequestIdConflict


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    repo.init_schema()
    yield
    scheduler._pool.shutdown(wait=True)
    backfills.shutdown()


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


@app.post("/cards/{card_name}/score", response_model=ScoreResponse)
def score_applicant(card_name: str, body: dict):
    if not isinstance(body.get("features"), dict):
        raise HTTPException(400, "features 必须是对象：{特征名: 原始值}")
    try:
        outcome = auditor.score_single(card_name, body)
    except InvalidRequestId as exc:
        raise HTTPException(400, str(exc))
    except RequestIdConflict as exc:
        # 同标识不同内容：明确拒绝（409），绝不覆盖原留痕
        raise HTTPException(409, str(exc))
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, f"该申请人打分失败：{exc}")
    # 响应字段与数值保持原样（request_id 为入参，不回显新字段）
    return {"card_name": card_name, "version": outcome.version,
            **outcome.result}


@app.post("/cards/{card_name}/score/batch", response_model=BatchScoreResponse)
def score_batch(card_name: str, req: BatchScoreRequest):
    try:
        ver, _rt = runtimes.get(card_name, req.version)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    # 逐项留痕；单条出错（含幂等冲突）只影响那一条，原批量隔离语义不变
    results = auditor.score_batch(card_name, req.version, req.applicants)
    ok_cnt = sum(1 for r in results if r["ok"])
    fail_cnt = len(results) - ok_cnt
    return BatchScoreResponse(
        card_name=card_name, version=ver, results=results,
        succeeded=ok_cnt, failed=fail_cnt,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# 监控路由在所有 @app 装饰器路由定义之后挂载（集中管理）
app.include_router(monitor_router)
