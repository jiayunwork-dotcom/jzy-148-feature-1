"""打分留痕与请求标识幂等（服务层，不依赖 HTTP）。

幂等语义
--------
- 调用方可在单条打分的 body 或批量条目中带 request_id（每卡每申请人一个
  命名空间，单条与批量共用）。
- 同标识 + 同内容重放：直接返回第一次的规范化打分结果（逐字段一致），
  且只落一条打分记录，重试不撑大统计。
- 同标识 + 不同内容：明确拒绝（HTTP 层映射为 409 / 批量条目 conflict），
  原记录永不被覆盖。
- 首次打分失败（如数值特征收到非数值）会回滚 pending 登记，调用方修正后
  可用同标识重试。
- 并发同标识同内容：后到者等待先到者提交后重放，全集群（数据库后端下跨
  进程）只有一次打分落痕。
- 不带标识：照常打分、照常落痕，不登记幂等。

打分记录是监控统计的唯一事实源：只在成功打分后写入一次。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone

from ..storage.repository import Repository


class IdempotentConflict(Exception):
    """同标识不同内容。"""


def _json_safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None  # NaN/Inf 不是合法 JSON，落库时归一为 null
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    return v


def content_hash(version: int, features: dict, applicant_id: str | None = None) -> str:
    """规范化内容哈希：键排序的紧凑 JSON；NaN/Inf 归一 null。

    比较的是语义内容而非序列化形式：{"a":1,"b":2} 与 {"b":2,"a":1} 同哈希；
    数字按 JSON 数值比较（1 与 1.0 视为相同）。
    """
    payload = {"version": version, "features": _json_safe(features)}
    if applicant_id is not None:
        payload["applicant_id"] = applicant_id
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _feature_bins(result: dict) -> list[dict]:
    return [{
        "name": f["name"],
        "bin_label": f["bin_label"],
        "status": f["status"],
        "raw_value": _json_safe(f.get("raw_value")),
    } for f in result["features"]]


def _canonical_result(version: int, result: dict) -> dict:
    """幂等表中保存的规范化结果：与单条打分响应去掉 card_name 后逐字段一致。"""
    return {
        "version": version,
        "total_score": result["total_score"],
        "pd": result["pd"],
        "features": result["features"],
        "has_unseen": result["has_unseen"],
        "has_missing": result["has_missing"],
    }


class AuditService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    # ---------------------------------------------------------- 单条
    def score_single(self, card_name: str, version: int, runtime,
                     features: dict, request_id: str | None) -> tuple[dict, bool]:
        """成功时返回 (规范化打分结果（含 version，不含 card_name）, 是否重放)。

        冲突时抛 IdempotentConflict；打分本身失败抛底层 ValueError。
        """
        if request_id is None:
            result = runtime.score(features)
            self.repo.insert_score_logs(
                [self._log_row(card_name, version, result)])
            return _canonical_result(version, result), False

        h = content_hash(version, features)
        state = self.repo.idempotent_begin(card_name, request_id, h, version)
        if state == "conflict":
            raise IdempotentConflict(request_id)
        if state in ("replayed", "pending"):
            if state == "pending":
                state = self.repo.idempotent_wait_committed(
                    card_name, request_id, timeout=10.0)
                if state == "pending":
                    # 首个请求长时间未落定（异常退出且未 abort）：让调用方稍后重试，
                    # 绝不替它重算，避免重复落痕
                    raise RuntimeError(
                        f"请求标识 {request_id} 的首个请求长时间未完成，请稍后重试")
                if state == "aborted":
                    state = self.repo.idempotent_begin(
                        card_name, request_id, h, version)
                    if state == "conflict":
                        raise IdempotentConflict(request_id)
                    assert state == "created"
                    return self._score_and_commit(
                        card_name, version, runtime, features, request_id)
            stored = self.repo.get_idempotent_response(card_name, request_id)
            return copy.deepcopy(stored), True
        return self._score_and_commit(
            card_name, version, runtime, features, request_id)

    def _score_and_commit(self, card_name, version, runtime, features,
                          request_id) -> tuple[dict, bool]:
        try:
            result = runtime.score(features)
        except Exception:
            self.repo.idempotent_abort(card_name, request_id)
            raise
        canonical = _canonical_result(version, result)
        row = self._log_row(card_name, version, result)
        self.repo.idempotent_commit(card_name, request_id, canonical, [row])
        return copy.deepcopy(canonical), False

    # ---------------------------------------------------------- 批量
    def score_batch(self, card_name: str, version: int, runtime,
                    applicants: list[dict]) -> list[dict]:
        """返回与入参同序的逐条结果。成功条目结构同既有批量响应，另带
        replayed（本次是否幂等重放，重放不算新打分）；失败/冲突条目 ok=False。
        """
        out: list[dict] = []
        direct_rows: list[dict] = []
        for item in applicants:
            rid = item.get("request_id")
            feats = item["features"]
            applicant_id = item["applicant_id"]
            if rid is None:
                try:
                    r = runtime.score(feats)
                except Exception as exc:
                    out.append(self._fail_item(applicant_id, exc))
                    continue
                direct_rows.append(self._log_row(card_name, version, r))
                out.append(self._ok_item(applicant_id, r, replayed=False))
                continue
            try:
                r, replayed = self.score_single(
                    card_name, version, runtime, feats, rid)
                out.append(self._ok_item(applicant_id, r, replayed=replayed))
            except IdempotentConflict:
                out.append({
                    "applicant_id": applicant_id, "ok": False,
                    "total_score": None, "pd": None, "features": None,
                    "has_unseen": False, "has_missing": False,
                    "error": f"IdempotentConflict: 请求标识 {rid!r} 已用于不同内容",
                    "replayed": False, "conflict": True,
                })
            except ValueError as exc:
                out.append(self._fail_item(applicant_id, exc))
            except RuntimeError as exc:
                out.append(self._fail_item(applicant_id, exc))
        if direct_rows:
            self.repo.insert_score_logs(direct_rows)
        return out

    @staticmethod
    def _ok_item(applicant_id: str, result: dict, replayed: bool) -> dict:
        return {
            "applicant_id": applicant_id, "ok": True,
            "total_score": result["total_score"], "pd": result["pd"],
            "features": result["features"],
            "has_unseen": result["has_unseen"],
            "has_missing": result["has_missing"],
            "error": None, "replayed": replayed, "conflict": False,
        }

    @staticmethod
    def _fail_item(applicant_id: str, exc: Exception) -> dict:
        return {
            "applicant_id": applicant_id, "ok": False,
            "total_score": None, "pd": None, "features": None,
            "has_unseen": False, "has_missing": False,
            "error": f"{type(exc).__name__}: {exc}",
            "replayed": False, "conflict": False,
        }

    # ---------------------------------------------------------- 记录构造
    def _log_row(self, card_name: str, version: int, result: dict,
                 request_id: str | None = None) -> dict:
        return {
            "card_name": card_name,
            "version": version,
            "request_id": request_id,
            "total_score": float(result["total_score"]),
            "pd": float(result["pd"]),
            "feature_bins": _feature_bins(result),
            "scored_at": datetime.now(timezone.utc),
        }
