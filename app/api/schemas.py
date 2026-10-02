"""HTTP 接口的请求/响应模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BuildResponse(BaseModel):
    job_id: int
    card_name: str
    status: str = "pending"


class JobOut(BaseModel):
    id: int
    card_name: str
    status: str
    version: int | None = None
    error: str | None = None
    newton_history: list[dict[str, Any]] | None = None


class CardOut(BaseModel):
    card_name: str
    latest_version: int | None = None
    version_count: int


class FeatureResult(BaseModel):
    name: str
    bin_label: str | None
    woe: float | None
    score: float
    status: str
    raw_value: Any = None
    detail: str = ""


class ScoreResponse(BaseModel):
    card_name: str
    version: int
    total_score: float
    pd: float
    features: list[FeatureResult]
    has_unseen: bool
    has_missing: bool


class BatchItem(BaseModel):
    applicant_id: str = Field(..., description="调用方自定义的申请人标识")
    features: dict[str, Any]
    request_id: str | None = Field(
        None, description="调用方请求标识：同标识同内容重放幂等、不同内容拒绝")


class BatchScoreRequest(BaseModel):
    card_name: str | None = None   # 卡名以路径参数为准，这里可选仅为兼容
    version: int | None = None
    applicants: list[BatchItem]


class ScoreRequest(BaseModel):
    version: int | None = None
    features: dict[str, Any]
    request_id: str | None = Field(
        None, description="调用方请求标识：同标识同内容重放幂等、不同内容拒绝")


class BatchResult(BaseModel):
    applicant_id: str
    ok: bool
    total_score: float | None = None
    pd: float | None = None
    features: list[FeatureResult] | None = None
    has_unseen: bool = False
    has_missing: bool = False
    error: str | None = None


class BatchScoreResponse(BaseModel):
    card_name: str
    version: int
    results: list[BatchResult]
    succeeded: int
    failed: int


# ------------------------------------------------------------ 表现回填
class BackfillItem(BaseModel):
    request_id: str
    label: Any = Field(..., description="实际是否违约，必须为 0/1")


class BackfillRequest(BaseModel):
    items: list[BackfillItem] = Field(..., min_length=1)


class BackfillAccepted(BaseModel):
    job_id: int
    card_name: str
    total: int
    status: str = "pending"
