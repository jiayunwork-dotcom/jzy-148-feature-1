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
