"""打分留痕与请求标识幂等（服务层，接口层只做薄封装）。

规则
----
- 每一次成功打分（单条，及批量中成功的每一条）追加一条不可变留痕：卡名、
  版本、调用方请求标识、入模原始特征、逐特征落箱（箱标签 / 缺失 / 未见 /
  越界）、总分、PD、打分时间（UTC）。
- 调用方可带 request_id。同一 (卡名, request_id) 携带**同样内容**重放：
  返回与第一次完全相同的结果（直接取原留痕，不再打分），计数不加。
  携带**不同内容**（包括指向另一版本）：抛 RequestIdConflict，绝不覆盖。
- 不带 request_id：照常打分、照常留痕（唯一性约束为部分索引，NULL 不参与）。
- 内容指纹：{"version": 解析后的版本号, "features": 入参特征} 的规范化 JSON
  SHA-256（键排序、紧凑分隔）。注意整数 35 与浮点 35.0 规范化后不同，
  视为不同内容——调用方应保持类型一致。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from ..storage.repository import IdempotencyConflict

MAX_REQUEST_ID_LEN = 128


class RequestIdConflict(Exception):
    """同一请求标识已存在但内容（或版本）不一致。"""


class InvalidRequestId(Exception):
    """请求标识非法（过长等）。"""


def normalize_request_id(raw) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise InvalidRequestId("request_id 必须是字符串")
    rid = raw.strip()
    if not rid:
        return None
    if len(rid) > MAX_REQUEST_ID_LEN:
        raise InvalidRequestId(
            f"request_id 最长 {MAX_REQUEST_ID_LEN} 个字符，实际 {len(rid)}")
    return rid


def content_hash(version: int, features: dict) -> str:
    body = {"version": version,
            "features": json.dumps(features, sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=False,
                                   default=str)}
    # 先把 features 自身规范化（default=str 兜底非常规类型），再整体哈希
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class AuditOutcome:
    status: str                  # "inserted" | "duplicate"
    version: int
    result: dict                 # 与 rt.score 同构的完整结果
    record: dict


class ScoreAudit:
    def __init__(self, repo, runtimes) -> None:
        self.repo = repo
        self.runtimes = runtimes

    def _record_or_score(
        self, card_name: str, version: int | None, features: dict,
        request_id: str | None, now: datetime | None = None,
    ) -> AuditOutcome:
        if request_id is None:
            resolved, rt = self.runtimes.get(card_name, version)
            result = rt.score(features)
            return self._insert(card_name, resolved, request_id, features,
                                result, now)

        # 显式指定版本：先确认版本存在（缺版本是 404，不能被幂等键盖成 409）
        if version is not None:
            self.runtimes.resolve_version(card_name, version)

        existing = self.repo.find_score_record(card_name, request_id)
        if existing is not None:
            # 用原留痕的解析版本算指纹：默认版本在卡升级后重放也必须一致
            h = content_hash(existing["version"], features)
            if h != existing["request_hash"] or (
                    version is not None and version != existing["version"]):
                raise RequestIdConflict(
                    f"请求标识 {request_id!r} 已存在但内容（或版本）与本次不一致，"
                    "拒绝覆盖原留痕")
            return AuditOutcome("duplicate", existing["version"],
                                copy.deepcopy(existing["result"]), existing)

        resolved, rt = self.runtimes.get(card_name, version)
        h = content_hash(resolved, features)
        result = rt.score(features)
        try:
            outcome = self._insert(
                card_name, resolved, request_id, features, result, now, h)
        except IdempotencyConflict:
            # 并发下被另一个线程抢先插入：退化为重放/冲突判定
            existing = self.repo.find_score_record(card_name, request_id)
            assert existing is not None
            if existing["request_hash"] != content_hash(
                    existing["version"], features) or (
                    version is not None and version != existing["version"]):
                raise
            return AuditOutcome("duplicate", existing["version"],
                                copy.deepcopy(existing["result"]), existing)
        return outcome

    def _insert(self, card_name, version, request_id, features, result,
                now, h=None):
        if h is None:
            h = content_hash(version, features)
        scored_at = now or datetime.now(timezone.utc)
        status, record = self.repo.insert_score_record(
            card_name=card_name, version=version, request_id=request_id,
            request_hash=h, features=features, result=result,
            scored_at=scored_at)
        return AuditOutcome(status, version, result, record)

    # 对外 -------------------------------------------------------------
    def score_single(self, card_name: str, body: dict,
                     now: datetime | None = None) -> AuditOutcome:
        features = body["features"]
        rid = normalize_request_id(body.get("request_id"))
        return self._record_or_score(card_name, body.get("version"),
                                     features, rid, now)

    def score_batch(self, card_name: str, version: int | None,
                    items: list, now: datetime | None = None) -> list[dict]:
        """逐项留痕，单项失败/冲突只影响该项（与既有批量隔离语义一致）。

        item 需有 applicant_id / features / 可选 request_id。
        返回与接口 BatchResult 同构的 dict 列表。
        """
        out = []
        for item in items:
            try:
                rid = normalize_request_id(getattr(item, "request_id", None))
            except InvalidRequestId as exc:
                out.append(self._fail_item(getattr(item, "applicant_id", None),
                                           f"InvalidRequestId: {exc}"))
                continue
            try:
                outcome = self._record_or_score(
                    card_name, version, item.features, rid, now)
                r = outcome.result
                out.append({
                    "applicant_id": item.applicant_id,
                    "ok": True,
                    "total_score": r["total_score"],
                    "pd": r["pd"],
                    "features": r["features"],
                    "has_unseen": r["has_unseen"],
                    "has_missing": r["has_missing"],
                    "error": None,
                })
            except RequestIdConflict as exc:
                out.append(self._fail_item(item.applicant_id,
                                           f"RequestIdConflict: {exc}"))
            except Exception as exc:  # 单条出错只影响那一条
                out.append(self._fail_item(
                    item.applicant_id, f"{type(exc).__name__}: {exc}"))
        return out

    @staticmethod
    def _fail_item(applicant_id, error) -> dict:
        return {
            "applicant_id": applicant_id, "ok": False,
            "total_score": None, "pd": None, "features": None,
            "has_unseen": False, "has_missing": False, "error": error,
        }
