"""后台建卡作业调度。

- ThreadPoolExecutor 执行建卡，多个作业并发跑时各自持有独立的 Sample、
  FeatureBinning、设计矩阵与 beta，不存在共享可变状态，中间结果不会互串。
- 作业失败（含牛顿迭代不收敛）状态落 failed，错误原因与可拿到的牛顿迭代
  历史一并落库。
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.exceptions import BuildError
from ..core.pipeline import BuildParams, run_pipeline
from ..core.sample import parse_csv
from ..storage.repository import Repository


class JobScheduler:
    def __init__(self, repo: Repository, max_workers: int = 4) -> None:
        self.repo = repo
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="build")
        self._submitted: set[int] = set()
        self._lock = threading.Lock()

    def submit(self, job_id: int, card_name: str, csv_text: str,
               params: BuildParams) -> None:
        with self._lock:
            self._submitted.add(job_id)
        self._pool.submit(self._run, job_id, card_name, csv_text, params)

    def _run(self, job_id: int, card_name: str, csv_text: str,
             params: BuildParams) -> None:
        try:
            self.repo.update_job(job_id, "running")
            # 每个作业在自己的栈上重新解析样本：并发作业之间零共享
            sample = parse_csv(csv_text, label_col=params.label_col)
            artifacts = run_pipeline(sample, params)
            version = self.repo.save_version(card_name, artifacts, job_id)
            self.repo.update_job(
                job_id, "succeeded", version=version,
                newton_history=artifacts["regression"]["history"],
            )
        except BuildError as exc:
            history = getattr(exc, "history", None)
            self.repo.update_job(
                job_id, "failed", error=str(exc), newton_history=history,
            )
        except Exception as exc:  # 任何意外都记录为作业失败，不让 worker 静默死掉
            self.repo.update_job(
                job_id, "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self._lock:
                self._submitted.discard(job_id)
