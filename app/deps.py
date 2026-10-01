"""应用级单例：仓储、调度器、打分运行时缓存。

STORAGE_BACKEND=memory 时使用进程内仓储（无需 PostgreSQL，供测试使用）；
默认 postgres（docker-compose 部署）。
"""
from __future__ import annotations

import os
import threading

from .jobs.scheduler import JobScheduler
from .scoring.engine import ScorecardRuntime
from .storage.postgres import PostgresRepository
from .storage.repository import InMemoryRepository, Repository


class RuntimeCache:
    """已加载评分卡运行时的缓存。版本产物不可变，缓存无需失效。"""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self._cache: dict[tuple[str, int], ScorecardRuntime] = {}
        self._lock = threading.Lock()

    def resolve_version(self, card_name: str,
                        version: int | None) -> tuple[int, dict]:
        summaries = self.repo.list_versions(card_name)
        if not summaries:
            raise KeyError(f"卡 {card_name!r} 不存在或尚无成功版本")
        versions = {s["version"] for s in summaries}
        if version is None:
            version = max(versions)
        elif version not in versions:
            raise KeyError(f"卡 {card_name!r} 没有版本 {version}")
        return version, self.repo.load_version(card_name, version)

    def get(self, card_name: str, version: int | None) -> tuple[int, ScorecardRuntime]:
        version, artifacts = self.resolve_version(card_name, version)
        key = (card_name, version)
        with self._lock:
            rt = self._cache.get(key)
            if rt is None:
                rt = ScorecardRuntime(artifacts)
                self._cache[key] = rt
        return version, rt


def build_repository() -> Repository:
    backend = os.getenv("STORAGE_BACKEND", "postgres").lower()
    if backend == "memory":
        return InMemoryRepository()
    from .config import settings
    return PostgresRepository(settings.database_url)


repo: Repository = build_repository()
scheduler = JobScheduler(repo)
runtimes = RuntimeCache(repo)
