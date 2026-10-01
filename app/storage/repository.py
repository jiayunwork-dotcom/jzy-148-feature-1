"""版本存储抽象层。

两套实现共用同一接口：
- InMemoryRepository：进程内实现，测试与本地无数据库运行时使用；
- PostgresRepository：PostgreSQL 16，版本号在事务内用 FOR UPDATE 行锁分配，
  同名卡并发建卡不会串版本。
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod


class Repository(ABC):
    @abstractmethod
    def init_schema(self) -> None: ...

    # 卡 ---------------------------------------------------------------
    @abstractmethod
    def ensure_card(self, name: str) -> None: ...

    @abstractmethod
    def list_cards(self) -> list[dict]: ...

    @abstractmethod
    def list_versions(self, card_name: str) -> list[dict]: ...

    @abstractmethod
    def save_version(self, card_name: str, artifacts: dict, job_id: int) -> int: ...

    @abstractmethod
    def load_version(self, card_name: str, version: int | None) -> dict:
        """version=None 时返回最新成功版本。"""

    @abstractmethod
    def get_version_summary(self, card_name: str, version: int) -> dict | None: ...

    # 作业 -------------------------------------------------------------
    @abstractmethod
    def create_job(self, card_name: str, payload: dict) -> int: ...

    @abstractmethod
    def update_job(self, job_id: int, status: str, **fields) -> None: ...

    @abstractmethod
    def get_job(self, job_id: int) -> dict | None: ...

    @abstractmethod
    def list_jobs(self, card_name: str | None = None) -> list[dict]: ...


class InMemoryRepository(Repository):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cards: dict[str, dict] = {}
        self._versions: dict[str, dict[int, dict]] = {}
        self._jobs: dict[int, dict] = {}
        self._job_seq = 0

    def init_schema(self) -> None:
        return None

    def ensure_card(self, name: str) -> None:
        with self._lock:
            self._cards.setdefault(name, {"name": name})

    def list_cards(self) -> list[dict]:
        with self._lock:
            out = []
            for name in sorted(self._cards):
                vers = self._versions.get(name, {})
                successful = sorted(vers)
                out.append({
                    "card_name": name,
                    "latest_version": successful[-1] if successful else None,
                    "version_count": len(successful),
                })
            return out

    def list_versions(self, card_name: str) -> list[dict]:
        with self._lock:
            vers = self._versions.get(card_name, {})
            return [vers[v]["summary"] for v in sorted(vers)]

    def save_version(self, card_name: str, artifacts: dict, job_id: int) -> int:
        with self._lock:
            vers = self._versions.setdefault(card_name, {})
            version = (max(vers) + 1) if vers else 1
            summary = {
                "card_name": card_name,
                "version": version,
                "job_id": job_id,
                "status": "succeeded",
                "params": artifacts["params"],
                "data_summary": artifacts["data_summary"],
                "selected_features": artifacts["selected_features"],
                "metrics": artifacts["metrics"],
                "validation": artifacts["validation"],
            }
            vers[version] = {"summary": summary, "artifacts": artifacts}
            return version

    def load_version(self, card_name: str, version: int | None) -> dict:
        with self._lock:
            vers = self._versions.get(card_name)
            if not vers:
                raise KeyError(f"卡 {card_name!r} 不存在或尚无成功版本")
            if version is None:
                version = max(vers)
            if version not in vers:
                raise KeyError(f"卡 {card_name!r} 没有版本 {version}")
            return vers[version]["artifacts"]

    def get_version_summary(self, card_name: str, version: int) -> dict | None:
        with self._lock:
            vers = self._versions.get(card_name, {})
            if version not in vers:
                return None
            return vers[version]["summary"]

    def create_job(self, card_name: str, payload: dict) -> int:
        with self._lock:
            self._job_seq += 1
            job = {
                "id": self._job_seq,
                "card_name": card_name,
                "status": "pending",
                "request": payload,
                "error": None,
                "newton_history": None,
                "version": None,
            }
            self._jobs[job["id"]] = job
            return job["id"]

    def update_job(self, job_id: int, status: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = status
            for k, v in fields.items():
                job[k] = v

    def get_job(self, job_id: int) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None

    def list_jobs(self, card_name: str | None = None) -> list[dict]:
        with self._lock:
            jobs = [dict(j) for j in self._jobs.values()]
        if card_name:
            jobs = [j for j in jobs if j["card_name"] == card_name]
        jobs.sort(key=lambda j: j["id"])
        return jobs
