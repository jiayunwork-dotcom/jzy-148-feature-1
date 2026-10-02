"""版本存储抽象层。

两套实现共用同一接口：
- InMemoryRepository：进程内实现，测试与本地无数据库运行时使用；
- PostgresRepository：PostgreSQL 16，版本号在事务内用 FOR UPDATE 行锁分配，
  同名卡并发建卡不会串版本。

监控层（投产后）在本抽象上新增四类操作，两套实现行为一致：
- insert/find/list score_records：不可变打分留痕 + 请求标识幂等
- apply_label：表现回填（先到为准，冲突不覆盖）
- backfill jobs：回填后台作业进度/结果
- get/set/clear score_baseline：老版本总分基准的补录管理（新版本建卡时自带）
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class IdempotencyConflict(Exception):
    """幂等键已存在但内容指纹不同（两套存储实现都会抛）。"""


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

    # 打分留痕（监控层） -----------------------------------------------
    @abstractmethod
    def find_score_record(self, card_name: str,
                          request_id: str) -> dict | None: ...

    @abstractmethod
    def insert_score_record(
        self, card_name: str, version: int, request_id: str | None,
        request_hash: str, features: dict, result: dict,
        scored_at: datetime,
    ) -> tuple[str, dict]:
        """插入一条打分留痕。返回 (status, record)：
        status="inserted" 正常新记录；"duplicate" 同标识同指纹的并发重放
        （此时 record 为已存在的原记录）。
        同标识不同指纹必须抛 RequestIdConflict，绝不覆盖。
        request_id=None 总是追加（部分唯一索引/NULL 不参与）。
        """

    @abstractmethod
    def list_score_records(
        self, card_name: str, version: int,
        start: datetime | None = None, end: datetime | None = None,
        labeled: bool | None = None,
    ) -> list[dict]:
        """按版本 + [start, end) 打分时间列出留痕。labeled=True 只返回已
        回填标签的；False 只返回未回填的。"""

    @abstractmethod
    def count_score_records(
        self, card_name: str, version: int | None = None,
        start: datetime | None = None, end: datetime | None = None,
    ) -> int: ...

    # 表现回填 ----------------------------------------------------------
    @abstractmethod
    def apply_label(self, card_name: str, request_id: str,
                    label: int) -> str:
        """返回 "applied" | "duplicate"(标签相同) | "conflict"(标签不同)
        | "missing"(留痕不存在)。先到为准，conflict 绝不覆盖。"""

    @abstractmethod
    def create_backfill_job(self, card_name: str,
                            items: list[dict]) -> int: ...

    @abstractmethod
    def update_backfill_job(self, job_id: int, status: str, **fields) -> None: ...

    @abstractmethod
    def get_backfill_job(self, job_id: int) -> dict | None: ...

    @abstractmethod
    def list_backfill_jobs(self, card_name: str | None = None) -> list[dict]: ...

    # 总分基准管理（老版本补录） ---------------------------------------
    @abstractmethod
    def set_score_baseline(self, card_name: str, version: int,
                           baseline: dict) -> None: ...

    @abstractmethod
    def clear_score_baseline(self, card_name: str, version: int) -> None: ...


class InMemoryRepository(Repository):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cards: dict[str, dict] = {}
        self._versions: dict[str, dict[int, dict]] = {}
        self._jobs: dict[int, dict] = {}
        self._job_seq = 0
        # ---- 监控层状态 ----
        self._score_records: list[dict] = []
        self._score_index: dict[tuple[str, str], dict] = {}
        self._backfill_jobs: dict[int, dict] = {}
        self._backfill_seq = 0

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

    # ------------------------------------------------------------ 打分留痕
    def find_score_record(self, card_name: str,
                          request_id: str) -> dict | None:
        with self._lock:
            rec = self._score_index.get((card_name, request_id))
            return None if rec is None else self._copy_record(rec)

    @staticmethod
    def _copy_record(rec: dict) -> dict:
        import copy
        out = dict(rec)
        out["features"] = copy.deepcopy(rec["features"])
        out["result"] = copy.deepcopy(rec["result"])
        return out

    def insert_score_record(
        self, card_name: str, version: int, request_id: str | None,
        request_hash: str, features: dict, result: dict,
        scored_at: datetime,
    ) -> tuple[str, dict]:
        import copy
        with self._lock:
            if request_id is not None:
                existing = self._score_index.get((card_name, request_id))
                if existing is not None:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyConflict(
                            f"请求标识 {request_id!r} 已存在但内容不同，拒绝覆盖")
                    return "duplicate", self._copy_record(existing)
            rec = {
                "id": len(self._score_records) + 1,
                "card_name": card_name,
                "version": version,
                "request_id": request_id,
                "request_hash": request_hash,
                "features": copy.deepcopy(features),
                "result": copy.deepcopy(result),
                "total_score": result["total_score"],
                "pd": result["pd"],
                "features_detail": copy.deepcopy(result["features"]),
                "label": None,
                "scored_at": _as_utc(scored_at),
            }
            self._score_records.append(rec)
            if request_id is not None:
                self._score_index[(card_name, request_id)] = rec
            return "inserted", self._copy_record(rec)

    def list_score_records(
        self, card_name: str, version: int,
        start: datetime | None = None, end: datetime | None = None,
        labeled: bool | None = None,
    ) -> list[dict]:
        su = _as_utc(start) if start else None
        eu = _as_utc(end) if end else None
        with self._lock:
            out = []
            for rec in self._score_records:
                if rec["card_name"] != card_name or rec["version"] != version:
                    continue
                ts = rec["scored_at"]
                if su and ts < su:
                    continue
                if eu and ts >= eu:
                    continue
                if labeled is True and rec["label"] is None:
                    continue
                if labeled is False and rec["label"] is not None:
                    continue
                out.append(self._record_view(rec))
            return out

    @staticmethod
    def _record_view(rec: dict) -> dict:
        """对外留痕视图：含稳定性/表现计算所需字段。"""
        return {
            "id": rec["id"],
            "card_name": rec["card_name"],
            "version": rec["version"],
            "request_id": rec["request_id"],
            "total_score": rec["total_score"],
            "pd": rec["pd"],
            "features": rec["features_detail"],
            "label": rec["label"],
            "scored_at": rec["scored_at"],
        }

    def count_score_records(
        self, card_name: str, version: int | None = None,
        start: datetime | None = None, end: datetime | None = None,
    ) -> int:
        su = _as_utc(start) if start else None
        eu = _as_utc(end) if end else None
        with self._lock:
            return sum(
                1 for rec in self._score_records
                if rec["card_name"] == card_name
                and (version is None or rec["version"] == version)
                and not (su and rec["scored_at"] < su)
                and not (eu and rec["scored_at"] >= eu)
            )

    # ------------------------------------------------------------ 表现回填
    def apply_label(self, card_name: str, request_id: str,
                    label: int) -> str:
        with self._lock:
            rec = self._score_index.get((card_name, request_id))
            if rec is None:
                return "missing"
            if rec["label"] is None:
                rec["label"] = int(label)
                return "applied"
            if int(rec["label"]) == int(label):
                return "duplicate"
            return "conflict"

    def create_backfill_job(self, card_name: str,
                            items: list[dict]) -> int:
        import copy
        with self._lock:
            self._backfill_seq += 1
            job = {
                "id": self._backfill_seq,
                "card_name": card_name,
                "status": "pending",
                "total": len(items),
                "processed": 0,
                "applied": 0,
                "duplicates": 0,
                "rejected": [],
                "items": copy.deepcopy(items),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            self._backfill_jobs[job["id"]] = job
            return job["id"]

    def update_backfill_job(self, job_id: int, status: str, **fields) -> None:
        with self._lock:
            job = self._backfill_jobs[job_id]
            job["status"] = status
            job["updated_at"] = datetime.now(timezone.utc)
            for k, v in fields.items():
                if k in ("total", "processed", "applied", "duplicates",
                         "rejected"):
                    job[k] = v

    def get_backfill_job(self, job_id: int) -> dict | None:
        with self._lock:
            j = self._backfill_jobs.get(job_id)
            if j is None:
                return None
            import copy
            out = {k: v for k, v in j.items() if k != "items"}
            out["rejected"] = copy.deepcopy(j["rejected"])
            out["created_at"] = j["created_at"].isoformat()
            out["updated_at"] = j["updated_at"].isoformat()
            return out

    def list_backfill_jobs(self, card_name: str | None = None) -> list[dict]:
        with self._lock:
            ids = sorted(self._backfill_jobs)
            jobs = [self.get_backfill_job(i) for i in ids]
        if card_name:
            jobs = [j for j in jobs if j["card_name"] == card_name]
        return jobs

    # ------------------------------------------------------------ 总分基准
    def set_score_baseline(self, card_name: str, version: int,
                           baseline: dict) -> None:
        with self._lock:
            vers = self._versions[card_name]
            vers[version]["artifacts"]["score_baseline"] = baseline
            vers[version]["summary"].setdefault("score_baseline", True)

    def clear_score_baseline(self, card_name: str, version: int) -> None:
        with self._lock:
            art = self._versions[card_name][version]["artifacts"]
            art["score_baseline"] = None
            vers = self._versions[card_name][version]["summary"]
            vers["score_baseline"] = False
