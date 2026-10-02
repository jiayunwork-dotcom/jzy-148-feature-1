"""表现回填后台作业：批量接收请求标识与实际 0/1 标签。

- 标签合法性在入队前校验（只接受 0/1；true/false 等布尔不算 0/1，非法条目
  进拒收清单，不影响其它条目）。
- 作业在独立线程池执行，进度/结果通过 GET 查询。
- 找不到标识 -> rejected(not_found)；同标识同标签再传 -> duplicated，幂等
  只算一遍；同标识不同标签 -> rejected(label_conflict)，首次标签永不覆盖，
  由监管/数据人工核对，服务不替业务决定以哪次为准。
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from ..storage.repository import Repository


def validate_items(raw_items: list[dict]) -> tuple[list[dict], list[dict]]:
    """拆分为合法 (request_id,label) 与拒收清单。合法标签只接受 int 0/1。"""
    valid: list[dict] = []
    rejected: list[dict] = []
    for it in raw_items:
        rid = it.get("request_id")
        label = it.get("label")
        if not isinstance(rid, str) or not rid.strip():
            rejected.append({"request_id": rid, "label": label,
                             "reason": "invalid_request_id"})
            continue
        # bool 是 int 的子类，必须显式排除；只接受真正的 0/1
        if isinstance(label, bool) or label not in (0, 1) or isinstance(label, str):
            rejected.append({"request_id": rid, "label": label,
                             "reason": "invalid_label"})
            continue
        valid.append({"request_id": rid, "label": int(label)})
    return valid, rejected


class BackfillScheduler:
    def __init__(self, repo: Repository, max_workers: int = 2) -> None:
        self.repo = repo
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="backfill")
        self._lock = threading.Lock()
        self._submitted: set[int] = set()

    def submit(self, job_id: int, card_name: str, items: list[dict]) -> None:
        with self._lock:
            self._submitted.add(job_id)
        self._pool.submit(self._run, job_id, card_name, items)

    def _run(self, job_id: int, card_name: str, items: list[dict]) -> None:
        try:
            self.repo.update_backfill_job(job_id, "running")
            res = self.repo.apply_perf_labels(job_id, card_name, items)
            self.repo.update_backfill_job(
                job_id, "succeeded", applied=res["applied"],
                duplicated=res["duplicated"], rejected=res["rejected"],
            )
        except Exception as exc:  # 后台作业任何异常都落 failed，不静默吞掉
            self.repo.update_backfill_job(
                job_id, "failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._submitted.discard(job_id)
