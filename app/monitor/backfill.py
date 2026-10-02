"""表现回填：上游成批上传 (request_id, label)，后台逐条核对并回填。

- 标签非法（不是 0/1）、请求标识缺失/为空/超长、在该卡全部留痕中找不到：
  进拒收清单（带序号、原因），不影响其它条目。
- 同一标识重复回填且标签不同：**先到为准，后到拒收**（进拒收清单，原因
  label_conflict），绝不改标签、绝不重复计入坏/好；标签相同的重复上传视为
  幂等重放（计 duplicate，不做任何修改）。
- request_id 不在路径卡名下（哪怕存在于别的卡）：按找不到处理，不同卡的
  标识空间互不干扰。
- 作业状态 pending -> running -> finished（部分拒收仍为 finished；
  全部拒收也是 finished，结论看计数）。进度/结果可查。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..core.config import settings
from .audit import MAX_REQUEST_ID_LEN


def _coerce_label(raw) -> int | None:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, str) and raw.strip() in ("0", "1"):
        return int(raw.strip())
    if isinstance(raw, (int, float)) and float(raw) in (0.0, 1.0):
        return int(raw)
    return None


class BackfillScheduler:
    def __init__(self, repo, max_workers: int | None = None) -> None:
        self.repo = repo
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or settings.backfill_workers,
            thread_name_prefix="backfill")

    def submit(self, job_id: int, card_name: str, items: list[dict]) -> None:
        self.repo.update_backfill_job(job_id, "running", total=len(items))
        self._pool.submit(self._run, job_id, card_name, items)

    def _run(self, job_id: int, card_name: str, items: list[dict]) -> None:
        applied = duplicates = 0
        rejected: list[dict] = []
        processed = 0
        for idx, item in enumerate(items):
            rid = item.get("request_id")
            label = _coerce_label(item.get("label"))
            if not isinstance(rid, str) or not rid.strip():
                rejected.append({"index": idx, "request_id": rid,
                                 "reason": "request_id 缺失或为空"})
            elif len(rid) > MAX_REQUEST_ID_LEN:
                rejected.append({"index": idx, "request_id": rid,
                                 "reason": f"request_id 超长（>{MAX_REQUEST_ID_LEN}）"})
            elif label is None:
                rejected.append({"index": idx, "request_id": rid,
                                 "reason": f"标签非法：{item.get('label')!r}，必须是 0/1"})
            else:
                outcome = self.repo.apply_label(card_name, rid.strip(), label)
                if outcome == "applied":
                    applied += 1
                elif outcome == "duplicate":
                    duplicates += 1
                elif outcome == "conflict":
                    rejected.append({
                        "index": idx, "request_id": rid.strip(),
                        "reason": f"label_conflict：该标识已回填不同标签（"
                                  f"现有 {self._existing_label(card_name, rid)}，"
                                  f"本次 {label}），先到为准，拒绝覆盖"})
                else:  # missing
                    rejected.append({"index": idx, "request_id": rid.strip(),
                                     "reason": "在该卡的打分留痕中找不到该请求标识"})
            processed += 1
            if processed % settings.backfill_progress_every == 0:
                self.repo.update_backfill_job(
                    job_id, "running", total=len(items), processed=processed,
                    applied=applied, duplicates=duplicates, rejected=rejected)

        self.repo.update_backfill_job(
            job_id, "finished", total=len(items), processed=processed,
            applied=applied, duplicates=duplicates, rejected=rejected)

    def _existing_label(self, card_name: str, request_id: str):
        rec = self.repo.find_score_record(card_name, request_id)
        return None if rec is None else rec.get("label")

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
