"""版本存储抽象层。

两套实现共用同一接口：
- InMemoryRepository：进程内实现，测试与本地无数据库运行时使用；
- PostgresRepository：PostgreSQL 16，版本号在事务内用 FOR UPDATE 行锁分配，
  同名卡并发建卡不会串版本。

投产后监控（留痕 / 回填）的表与接口追加在本文件后半部分：
score_logs（打分记录，唯一事实源）、idempotency（请求标识幂等）、
backfill_jobs（表现回填后台作业）、perf_labels（实际标签）。
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone


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

    # 打分留痕 ---------------------------------------------------------
    @abstractmethod
    def insert_score_logs(self, rows: list[dict]) -> None:
        """无请求标识的成功打分，直接落痕。"""

    @abstractmethod
    def idempotent_begin(self, card_name: str, request_id: str,
                         content_hash: str, version: int) -> str:
        """登记带标识请求（单条打分与批量条目共用每申请人一个命名空间）。
        返回：
        "created"  首次出现，调用方打分后必须 idempotent_commit，失败须 abort；
        "replayed" 同标识同内容已提交，调用方取 get_idempotent_response 原样返回；
        "conflict" 同标识不同内容（明确拒绝，不覆盖原记录）；
        "pending"  同标识同内容但首个请求仍在打分，调用方 wait 后再查。
        """

    @abstractmethod
    def idempotent_commit(self, card_name: str, request_id: str,
                          result: dict, log_rows: list[dict]) -> None:
        """首次请求打分成功：原子写入打分记录并发布规范化打分结果。"""

    @abstractmethod
    def idempotent_abort(self, card_name: str, request_id: str) -> None:
        """首次请求打分失败：删除 pending 登记，使同标识重试可重新进行。"""

    @abstractmethod
    def idempotent_wait_committed(self, card_name: str, request_id: str,
                                  timeout: float = 10.0) -> str:
        """等待另一线程的同标识请求落定，返回最终状态
        committed / conflict-pending 不可能出现；abort 后返回 "aborted"，超时 "pending"。"""

    @abstractmethod
    def get_idempotent_response(self, card_name: str, request_id: str) -> dict | None:
        """已提交标识的规范化打分结果（含 version、总分、PD、逐特征落箱）。"""

    @abstractmethod
    def query_score_logs(self, card_name: str, version: int,
                         start: datetime | None, end: datetime | None,
                         limit: int) -> list[dict]:
        """按 [start, end) 取某卡某版本的打分记录（scored_at 升序）。"""

    # 表现回填 ---------------------------------------------------------
    @abstractmethod
    def create_backfill_job(self, card_name: str, payload: dict) -> int: ...

    @abstractmethod
    def update_backfill_job(self, job_id: int, status: str, **fields) -> None: ...

    @abstractmethod
    def get_backfill_job(self, job_id: int) -> dict | None: ...

    @abstractmethod
    def list_backfill_jobs(self, card_name: str | None = None) -> list[dict]: ...

    @abstractmethod
    def apply_perf_labels(self, job_id: int, card_name: str,
                          items: list[dict]) -> dict:
        """逐条落标签，返回计数与拒收清单。items: [{request_id,label}]。
        - request_id 在打分记录中找不到 -> rejected(not_found)
        - 该标识此前已回填且标签不同 -> rejected(label_conflict)
        - 该标识此前已回填且标签相同 -> duplicated（不算第二遍）
        - 同一作业内重复提交同一标识：第一次落库，之后按 duplicated/rejected。
        """

    @abstractmethod
    def query_performance(self, card_name: str, version: int,
                          start: datetime | None, end: datetime | None,
                          limit: int) -> list[dict]:
        """取某区间内已有表现标签的成功打分记录（含 label），scored_at 升序。"""


class InMemoryRepository(Repository):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conds: dict[tuple[str, str], threading.Condition] = {}
        self._cards: dict[str, dict] = {}
        self._versions: dict[str, dict[int, dict]] = {}
        self._jobs: dict[int, dict] = {}
        self._job_seq = 0
        # 投产后监控
        self._score_logs: list[dict] = []
        self._idem: dict[tuple[str, str], dict] = {}
        self._backfill_jobs: dict[int, dict] = {}
        self._backfill_seq = 0
        self._perf: dict[tuple[str, str], dict] = {}

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
    def insert_score_logs(self, rows: list[dict]) -> None:
        with self._lock:
            for r in rows:
                r.setdefault("request_id", None)
                r.setdefault("backfill_job_id", None)
                r.setdefault("label", None)
                self._score_logs.append(r)

    def _cond(self, card_name: str, request_id: str) -> threading.Condition:
        # 调用方须持 self._lock；条件对象按键复用，避免广播唤醒无关等待者
        return self._conds.setdefault(
            (card_name, request_id), threading.Condition(self._lock))

    def idempotent_begin(self, card_name: str, request_id: str,
                         content_hash: str, version: int) -> str:
        with self._lock:
            key = (card_name, request_id)
            rec = self._idem.get(key)
            if rec is None:
                self._idem[key] = {
                    "content_hash": content_hash, "version": version,
                    "status": "pending", "result": None,
                }
                return "created"
            if rec["status"] == "committed":
                return "replayed" if rec["content_hash"] == content_hash else "conflict"
            # 已有 pending
            if rec["content_hash"] != content_hash:
                return "conflict"
            return "pending"

    def idempotent_commit(self, card_name: str, request_id: str,
                          result: dict, log_rows: list[dict]) -> None:
        with self._lock:
            rec = self._idem[(card_name, request_id)]
            rec["status"] = "committed"
            rec["result"] = result
            for r in log_rows:
                r["request_id"] = request_id
                r.setdefault("backfill_job_id", None)
                r.setdefault("label", None)
                self._score_logs.append(r)
            self._cond(card_name, request_id).notify_all()

    def idempotent_abort(self, card_name: str, request_id: str) -> None:
        with self._lock:
            rec = self._idem.get((card_name, request_id))
            if rec is not None and rec["status"] == "pending":
                del self._idem[(card_name, request_id)]
                self._cond(card_name, request_id).notify_all()

    def idempotent_wait_committed(self, card_name: str, request_id: str,
                                  timeout: float = 10.0) -> str:
        import time
        with self._lock:
            cond = self._cond(card_name, request_id)
            deadline = time.monotonic() + timeout
            while True:
                rec = self._idem.get((card_name, request_id))
                if rec is None:
                    return "aborted"
                if rec["status"] == "committed":
                    return "committed"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "pending"
                cond.wait(timeout=remaining)

    def get_idempotent_response(self, card_name: str,
                                request_id: str) -> dict | None:
        with self._lock:
            rec = self._idem.get((card_name, request_id))
            if rec is None or rec["status"] != "committed":
                return None
            return rec["result"]

    def query_score_logs(self, card_name: str, version: int,
                         start: datetime | None, end: datetime | None,
                         limit: int) -> list[dict]:
        with self._lock:
            out = []
            for r in self._score_logs:
                if r["card_name"] != card_name or r["version"] != version:
                    continue
                ts = r["scored_at"]
                if start is not None and ts < start:
                    continue
                if end is not None and ts >= end:
                    continue
                out.append(dict(r))
                if len(out) >= limit:
                    break
            return out

    # ------------------------------------------------------------ 表现回填
    def create_backfill_job(self, card_name: str, payload: dict) -> int:
        with self._lock:
            self._backfill_seq += 1
            job = {
                "id": self._backfill_seq, "card_name": card_name,
                "status": "pending", "payload": payload,
                "total": len(payload.get("items", [])),
                "applied": 0, "duplicated": 0, "rejected": [],
                "error": None,
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
                job[k] = v

    def get_backfill_job(self, job_id: int) -> dict | None:
        with self._lock:
            j = self._backfill_jobs.get(job_id)
            return dict(j) if j else None

    def list_backfill_jobs(self, card_name: str | None = None) -> list[dict]:
        with self._lock:
            jobs = [dict(j) for j in self._backfill_jobs.values()]
        if card_name:
            jobs = [j for j in jobs if j["card_name"] == card_name]
        jobs.sort(key=lambda j: j["id"])
        return jobs

    def apply_perf_labels(self, job_id: int, card_name: str,
                          items: list[dict]) -> dict:
        with self._lock:
            applied = duplicated = 0
            rejected: list[dict] = []
            # 作业内重复：同一 request_id 在本作业第一次出现正常落库，其后的
            # 同标识条目按既有标签判 duplicated / label_conflict（见下）
            logs = {r["request_id"]: r
                    for r in self._score_logs
                    if r["card_name"] == card_name and r["request_id"] is not None}
            for it in items:
                rid, label = it["request_id"], it["label"]
                existing = self._perf.get((card_name, rid))
                if existing is not None:
                    if existing["label"] == label:
                        duplicated += 1  # 同标识同标签重放：幂等，不算第二遍
                    else:
                        rejected.append({"request_id": rid, "label": label,
                                         "reason": "label_conflict",
                                         "existing_label": existing["label"]})
                    continue
                if rid not in logs:
                    rejected.append({"request_id": rid, "label": label,
                                     "reason": "not_found"})
                    continue
                self._perf[(card_name, rid)] = {
                    "card_name": card_name, "request_id": rid,
                    "label": int(label), "backfill_job_id": job_id,
                }
                logs[rid]["label"] = int(label)
                logs[rid]["backfill_job_id"] = job_id
                applied += 1
            return {"applied": applied, "duplicated": duplicated,
                    "rejected": rejected}

    def query_performance(self, card_name: str, version: int,
                          start: datetime | None, end: datetime | None,
                          limit: int) -> list[dict]:
        with self._lock:
            out = []
            for r in self._score_logs:
                if r["card_name"] != card_name or r["version"] != version:
                    continue
                if r.get("label") is None:
                    continue
                ts = r["scored_at"]
                if start is not None and ts < start:
                    continue
                if end is not None and ts >= end:
                    continue
                out.append(dict(r))
                if len(out) >= limit:
                    break
            return out
