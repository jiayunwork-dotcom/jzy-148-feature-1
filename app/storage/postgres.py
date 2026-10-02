"""PostgreSQL 16 存储实现（psycopg3）。

schema
------
cards(name PK)
jobs(id BIGSERIAL PK, card_name, status, request JSONB, error,
     newton_history JSONB, version, 时间戳)
versions(card_name, version, job_id, status, summary JSONB, artifacts JSONB,
         PK(card_name, version))

同名卡并发建卡：save_version 在事务内先对 cards 行 FOR UPDATE 取锁再分配
max(version)+1，因此并发作业串行分配、绝不重号。
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import psycopg
from psycopg.types.json import Jsonb

from .repository import Repository

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    name        TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS jobs (
    id              BIGSERIAL PRIMARY KEY,
    card_name       TEXT NOT NULL REFERENCES cards(name),
    status          TEXT NOT NULL,
    request         JSONB NOT NULL,
    error           TEXT,
    newton_history  JSONB,
    version         INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_card_idx ON jobs(card_name, id);
CREATE TABLE IF NOT EXISTS versions (
    card_name   TEXT NOT NULL REFERENCES cards(name),
    version     INTEGER NOT NULL,
    job_id      BIGINT NOT NULL REFERENCES jobs(id),
    status      TEXT NOT NULL,
    summary     JSONB NOT NULL,
    artifacts   JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (card_name, version)
);

-- ---------------------------------------------------------------- 投产后监控
-- 打分留痕：每一次成功打分一条（幂等重放不新增）。唯一事实源，
-- 稳定性与表现统计一律查询时从本表现算，不做增量计数。
CREATE TABLE IF NOT EXISTS score_logs (
    id            BIGSERIAL PRIMARY KEY,
    card_name     TEXT NOT NULL,
    version       INTEGER NOT NULL,
    request_id    TEXT,
    total_score   DOUBLE PRECISION NOT NULL,
    pd            DOUBLE PRECISION NOT NULL,
    feature_bins  JSONB NOT NULL,   -- [{name, bin_label, status, raw_value}]
    scored_at     TIMESTAMPTZ NOT NULL,
    label         SMALLINT,         -- 回填后写入 0/1
    backfill_job_id BIGINT
);
CREATE UNIQUE INDEX IF NOT EXISTS score_logs_request_uniq
    ON score_logs(card_name, request_id) WHERE request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS score_logs_query_idx
    ON score_logs(card_name, version, scored_at);

-- 请求标识幂等登记：pending -> committed；内容哈希不同永远 conflict
CREATE TABLE IF NOT EXISTS idempotency (
    card_name     TEXT NOT NULL,
    request_id    TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    version       INTEGER NOT NULL,
    status        TEXT NOT NULL,     -- pending | committed
    result        JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    committed_at  TIMESTAMPTZ,
    PRIMARY KEY (card_name, request_id)
);

-- 表现回填后台作业
CREATE TABLE IF NOT EXISTS backfill_jobs (
    id          BIGSERIAL PRIMARY KEY,
    card_name   TEXT NOT NULL,
    status      TEXT NOT NULL,      -- pending | running | succeeded | failed
    total       INTEGER NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    applied     INTEGER NOT NULL DEFAULT 0,
    duplicated  INTEGER NOT NULL DEFAULT 0,
    rejected    JSONB NOT NULL DEFAULT '[]'::jsonb,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS backfill_jobs_card_idx ON backfill_jobs(card_name, id);

-- 实际表现标签（按请求标识幂等：标签不同永不覆盖）
CREATE TABLE IF NOT EXISTS perf_labels (
    card_name       TEXT NOT NULL,
    request_id      TEXT NOT NULL,
    label           SMALLINT NOT NULL CHECK (label IN (0, 1)),
    backfill_job_id BIGINT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (card_name, request_id)
);
"""


class PostgresRepository(Repository):
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool = None

    def _p(self):
        if self._pool is None:
            from psycopg_pool import ConnectionPool
            self._pool = ConnectionPool(self.dsn, min_size=1, max_size=10,
                                        open=False, kwargs={"autocommit": False})
            self._pool.open(wait=True)
        return self._pool

    def init_schema(self, retries: int = 30, delay: float = 1.0) -> None:
        import time
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                with self._p().connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(_SCHEMA)
                    conn.commit()
                return
            except Exception as exc:  # 数据库尚未就绪时重试
                last_exc = exc
                time.sleep(delay)
        raise RuntimeError(
            f"数据库在 {retries} 次重试后仍不可用：{last_exc}")

    def ensure_card(self, name: str) -> None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO cards(name) VALUES (%s) ON CONFLICT DO NOTHING",
                    (name,),
                )
            conn.commit()

    def list_cards(self) -> list[dict]:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT c.name,
                           MAX(v.version) FILTER (WHERE v.status='succeeded') AS latest,
                           COUNT(v.*) AS cnt
                    FROM cards c
                    LEFT JOIN versions v ON v.card_name = c.name
                    GROUP BY c.name ORDER BY c.name
                """)
                rows = cur.fetchall()
            return [{"card_name": r[0], "latest_version": r[1],
                     "version_count": r[2]} for r in rows]

    def list_versions(self, card_name: str) -> list[dict]:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT summary FROM versions
                    WHERE card_name=%s ORDER BY version
                """, (card_name,))
                rows = cur.fetchall()
            return [r[0] for r in rows]

    def save_version(self, card_name: str, artifacts: dict, job_id: int) -> int:
        summary = {
            "card_name": card_name,
            "job_id": job_id,
            "status": "succeeded",
            "params": artifacts["params"],
            "data_summary": artifacts["data_summary"],
            "selected_features": artifacts["selected_features"],
            "metrics": artifacts["metrics"],
            "validation": artifacts["validation"],
        }
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    # 锁住卡行，串行化同卡的版本号分配
                    cur.execute("SELECT 1 FROM cards WHERE name=%s FOR UPDATE",
                                (card_name,))
                    if cur.fetchone() is None:
                        cur.execute(
                            "INSERT INTO cards(name) VALUES (%s)", (card_name,))
                    cur.execute(
                        "SELECT COALESCE(MAX(version), 0) + 1 FROM versions "
                        "WHERE card_name=%s",
                        (card_name,),
                    )
                    version = cur.fetchone()[0]
                    summary["version"] = version
                    cur.execute("""
                        INSERT INTO versions(card_name, version, job_id, status,
                                             summary, artifacts)
                        VALUES (%s, %s, %s, 'succeeded', %s, %s)
                    """, (card_name, version, job_id,
                          Jsonb(summary), Jsonb(artifacts)))
                    cur.execute(
                        "UPDATE jobs SET version=%s, updated_at=now() WHERE id=%s",
                        (version, job_id),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return version

    def load_version(self, card_name: str, version: int | None) -> dict:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                if version is None:
                    cur.execute("""
                        SELECT artifacts FROM versions
                        WHERE card_name=%s AND status='succeeded'
                        ORDER BY version DESC LIMIT 1
                    """, (card_name,))
                else:
                    cur.execute("""
                        SELECT artifacts FROM versions
                        WHERE card_name=%s AND version=%s AND status='succeeded'
                    """, (card_name, version))
                row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"卡 {card_name!r} 版本 {version or '最新'} 不存在"
            )
        return row[0]

    def get_version_summary(self, card_name: str, version: int) -> dict | None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT summary FROM versions WHERE card_name=%s AND version=%s",
                    (card_name, version),
                )
                row = cur.fetchone()
        return row[0] if row else None

    def create_job(self, card_name: str, payload: dict) -> int:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO cards(name) VALUES (%s) ON CONFLICT DO NOTHING",
                    (card_name,),
                )
                cur.execute("""
                    INSERT INTO jobs(card_name, status, request)
                    VALUES (%s, 'pending', %s) RETURNING id
                """, (card_name, Jsonb(payload)))
                job_id = cur.fetchone()[0]
            conn.commit()
        return job_id

    def update_job(self, job_id: int, status: str, **fields) -> None:
        allowed = {"error", "newton_history", "version"}
        sets = ["status=%s", "updated_at=now()"]
        args: list = [status]
        for k, v in fields.items():
            if k not in allowed:
                continue
            sets.append(f"{k}=%s")
            args.append(Jsonb(v) if k == "newton_history" else v)
        args.append(job_id)
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE jobs SET {', '.join(sets)} WHERE id=%s", args)
            conn.commit()

    def get_job(self, job_id: int) -> dict | None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, card_name, status, request, error,
                           newton_history, version,
                           created_at, updated_at
                    FROM jobs WHERE id=%s
                """, (job_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "card_name": row[1], "status": row[2],
            "request": row[3], "error": row[4], "newton_history": row[5],
            "version": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
        }

    def list_jobs(self, card_name: str | None = None) -> list[dict]:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                if card_name:
                    cur.execute("""
                        SELECT id, card_name, status, request, error,
                               newton_history, version, created_at, updated_at
                        FROM jobs WHERE card_name=%s ORDER BY id
                    """, (card_name,))
                else:
                    cur.execute("""
                        SELECT id, card_name, status, request, error,
                               newton_history, version, created_at, updated_at
                        FROM jobs ORDER BY id
                    """)
                rows = cur.fetchall()
        return [{
            "id": r[0], "card_name": r[1], "status": r[2], "request": r[3],
            "error": r[4], "newton_history": r[5], "version": r[6],
            "created_at": r[7].isoformat() if r[7] else None,
            "updated_at": r[8].isoformat() if r[8] else None,
        } for r in rows]

    # ------------------------------------------------------------ 打分留痕
    @staticmethod
    def _row_to_log(r) -> dict:
        return {
            "id": r[0], "card_name": r[1], "version": r[2],
            "request_id": r[3], "total_score": r[4], "pd": r[5],
            "feature_bins": r[6],
            "scored_at": r[7], "label": r[8], "backfill_job_id": r[9],
        }

    _LOG_COLS = ("card_name, version, request_id, total_score, pd, "
                 "feature_bins, scored_at")

    def insert_score_logs(self, rows: list[dict]) -> None:
        if not rows:
            return
        payload = [
            (r["card_name"], r["version"], r.get("request_id"),
             r["total_score"], r["pd"], Jsonb(r["feature_bins"]),
             _utc(r["scored_at"]))
            for r in rows
        ]
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO score_logs({self._LOG_COLS}) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        payload,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def idempotent_begin(self, card_name: str, request_id: str,
                         content_hash: str, version: int) -> str:
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO idempotency(card_name, request_id,
                                                content_hash, version, status)
                        VALUES (%s, %s, %s, %s, 'pending')
                        ON CONFLICT (card_name, request_id) DO NOTHING
                    """, (card_name, request_id, content_hash, version))
                    if cur.rowcount == 1:
                        conn.commit()
                        return "created"
                    # 已存在登记：锁住该行再判定，使「同标识不同内容 + 首个请求
                    # 仍在打分」也能立即拿到 conflict，而不是等待后错误重放。
                    # 打分只持锁极短时间（本事务在读出后立即提交）。
                    cur.execute("""
                        SELECT status, content_hash FROM idempotency
                        WHERE card_name=%s AND request_id=%s FOR UPDATE
                    """, (card_name, request_id))
                    status, existing_hash = cur.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        if status != "committed":
            return "pending" if existing_hash == content_hash else "conflict"
        return "replayed" if existing_hash == content_hash else "conflict"

    def idempotent_commit(self, card_name: str, request_id: str,
                          result: dict, log_rows: list[dict]) -> None:
        assert log_rows, "提交幂等结果必须带打分记录"
        payload = [
            (r["card_name"], r["version"], request_id,
             r["total_score"], r["pd"], Jsonb(r["feature_bins"]),
             _utc(r["scored_at"]))
            for r in log_rows
        ]
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO score_logs({self._LOG_COLS}) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        payload,
                    )
                    # result 与 log 同事务发布：重放方要么看到完整结果，
                    # 要么继续等 pending，绝不会读到半成品
                    cur.execute("""
                        UPDATE idempotency
                        SET status='committed', result=%s, committed_at=now()
                        WHERE card_name=%s AND request_id=%s AND status='pending'
                    """, (Jsonb(result), card_name, request_id))
                    if cur.rowcount != 1:
                        raise RuntimeError("幂等提交时登记不存在或已提交")
                    conn.commit()
            except Exception:
                conn.rollback()
                raise

    def idempotent_abort(self, card_name: str, request_id: str) -> None:
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM idempotency
                        WHERE card_name=%s AND request_id=%s AND status='pending'
                    """, (card_name, request_id))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def idempotent_wait_committed(self, card_name: str, request_id: str,
                                  timeout: float = 10.0) -> str:
        # 打分是毫秒级操作：等待方以 10ms 步长短轮询，避免跨连接 LISTEN 的复杂度
        deadline = time.monotonic() + timeout
        while True:
            with self._p().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT status FROM idempotency
                        WHERE card_name=%s AND request_id=%s
                    """, (card_name, request_id))
                    row = cur.fetchone()
            if row is None:
                return "aborted"
            if row[0] == "committed":
                return "committed"
            if time.monotonic() >= deadline:
                return "pending"
            time.sleep(0.01)

    def get_idempotent_response(self, card_name: str,
                                request_id: str) -> dict | None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT result FROM idempotency
                    WHERE card_name=%s AND request_id=%s AND status='committed'
                """, (card_name, request_id))
                row = cur.fetchone()
        return row[0] if row else None

    def query_score_logs(self, card_name: str, version: int,
                         start: datetime | None, end: datetime | None,
                         limit: int) -> list[dict]:
        sql, args = self._logs_where(card_name, version, start, end, False)
        sql += " ORDER BY scored_at, id LIMIT %s"
        args.append(limit)
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
        return [self._row_to_log(r) for r in rows]

    def query_performance(self, card_name: str, version: int,
                          start: datetime | None, end: datetime | None,
                          limit: int) -> list[dict]:
        sql, args = self._logs_where(card_name, version, start, end, True)
        sql += " ORDER BY scored_at, id LIMIT %s"
        args.append(limit)
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
        return [self._row_to_log(r) for r in rows]

    @staticmethod
    def _logs_where(card_name, version, start, end, with_label):
        sql = """
            SELECT id, card_name, version, request_id, total_score, pd,
                   feature_bins, scored_at, label, backfill_job_id
            FROM score_logs
            WHERE card_name=%s AND version=%s
        """
        args: list = [card_name, version]
        if with_label:
            sql += " AND label IS NOT NULL"
        if start is not None:
            sql += " AND scored_at >= %s"
            args.append(_utc(start))
        if end is not None:
            sql += " AND scored_at < %s"
            args.append(_utc(end))
        return sql, args

    # ------------------------------------------------------------ 表现回填
    def create_backfill_job(self, card_name: str, payload: dict) -> int:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO backfill_jobs(card_name, status, total, payload)
                    VALUES (%s, 'pending', %s, %s) RETURNING id
                """, (card_name, len(payload.get("items", [])),
                      Jsonb(payload)))
                job_id = cur.fetchone()[0]
            conn.commit()
        return job_id

    def update_backfill_job(self, job_id: int, status: str, **fields) -> None:
        allowed = {"applied", "duplicated", "rejected", "error"}
        sets = ["status=%s", "updated_at=now()"]
        args: list = [status]
        for k, v in fields.items():
            if k not in allowed:
                continue
            sets.append(f"{k}=%s")
            args.append(Jsonb(v) if k == "rejected" else v)
        args.append(job_id)
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE backfill_jobs SET {', '.join(sets)} WHERE id=%s",
                    args)
            conn.commit()

    @staticmethod
    def _job_row(r) -> dict:
        return {
            "id": r[0], "card_name": r[1], "status": r[2], "total": r[3],
            "applied": r[4], "duplicated": r[5], "rejected": r[6],
            "error": r[7],
            "created_at": r[8].isoformat() if r[8] else None,
            "updated_at": r[9].isoformat() if r[9] else None,
        }

    def get_backfill_job(self, job_id: int) -> dict | None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, card_name, status, total, applied, duplicated,
                           rejected, error, created_at, updated_at
                    FROM backfill_jobs WHERE id=%s
                """, (job_id,))
                row = cur.fetchone()
        return self._job_row(row) if row else None

    def list_backfill_jobs(self, card_name: str | None = None) -> list[dict]:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                if card_name:
                    cur.execute("""
                        SELECT id, card_name, status, total, applied, duplicated,
                               rejected, error, created_at, updated_at
                        FROM backfill_jobs WHERE card_name=%s ORDER BY id
                    """, (card_name,))
                else:
                    cur.execute("""
                        SELECT id, card_name, status, total, applied, duplicated,
                               rejected, error, created_at, updated_at
                        FROM backfill_jobs ORDER BY id
                    """)
                rows = cur.fetchall()
        return [self._job_row(r) for r in rows]

    def apply_perf_labels(self, job_id: int, card_name: str,
                          items: list[dict]) -> dict:
        applied = duplicated = 0
        rejected: list[dict] = []
        CHUNK = 1000
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    for lo in range(0, len(items), CHUNK):
                        for it in items[lo:lo + CHUNK]:
                            rid, label = it["request_id"], int(it["label"])
                            cur.execute("""
                                SELECT label FROM perf_labels
                                WHERE card_name=%s AND request_id=%s
                            """, (card_name, rid))
                            row = cur.fetchone()
                            if row is not None:
                                if int(row[0]) == label:
                                    duplicated += 1
                                else:
                                    rejected.append({
                                        "request_id": rid, "label": label,
                                        "reason": "label_conflict",
                                        "existing_label": int(row[0])})
                                continue
                            # NOT FOUND 随 UPDATE 判定：request_id 唯一约束
                            # 保证一个标识最多一条成功打分记录
                            cur.execute("""
                                UPDATE score_logs
                                SET label=%s, backfill_job_id=%s
                                WHERE card_name=%s AND request_id=%s
                                  AND label IS NULL
                                RETURNING id
                            """, (label, job_id, card_name, rid))
                            updated = cur.fetchone()
                            if updated is None:
                                # 与别的回填作业并发：标签可能刚被对方提交
                                cur.execute("""
                                    SELECT label FROM perf_labels
                                    WHERE card_name=%s AND request_id=%s
                                """, (card_name, rid))
                                row2 = cur.fetchone()
                                if row2 is not None:
                                    if int(row2[0]) == label:
                                        duplicated += 1
                                    else:
                                        rejected.append({
                                            "request_id": rid, "label": label,
                                            "reason": "label_conflict",
                                            "existing_label": int(row2[0])})
                                else:
                                    rejected.append({
                                        "request_id": rid, "label": label,
                                        "reason": "not_found"})
                                continue
                            cur.execute("""
                                INSERT INTO perf_labels(card_name, request_id,
                                                        label, backfill_job_id)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (card_name, request_id) DO NOTHING
                            """, (card_name, rid, label, job_id))
                            applied += 1
                        conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"applied": applied, "duplicated": duplicated,
                "rejected": rejected}


def _utc(dt: datetime) -> datetime:
    """无时区时间戳一律按 UTC 解释，保证与 in-memory 后端行为一致。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
