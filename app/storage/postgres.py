"""PostgreSQL 16 存储实现（psycopg3）。

schema
------
cards(name PK)
jobs(id BIGSERIAL PK, card_name, status, request JSONB, error,
     newton_history JSONB, version, 时间戳)
versions(card_name, version, job_id, status, summary JSONB, artifacts JSONB,
         PK(card_name, version))
score_records(id BIGSERIAL PK, card_name, version, request_id, request_hash,
              features JSONB 原始入参, result JSONB 完整打分结果, total_score,
              pd, label SMALLINT NULL 表现回填, scored_at TIMESTAMPTZ)
  - 幂等：(card_name, request_id) 上的**部分**唯一索引（request_id IS NOT NULL），
    不带标识的重放不去重；插入冲突后比对 request_hash 决定幂等返回或拒绝。
backfill_jobs(回填后台作业：进度计数 + rejected JSONB 拒收清单)

同名卡并发建卡：save_version 在事务内先对 cards 行 FOR UPDATE 取锁再分配
max(version)+1，因此并发作业串行分配、绝不重号。

平滑升级：所有对象 CREATE TABLE/INDEX IF NOT EXISTS，只新增不改动既有表，
在已有数据的老库上直接 init_schema 即可，无需清库。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import psycopg
from psycopg.types.json import Jsonb

from .repository import IdempotencyConflict, Repository

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
CREATE TABLE IF NOT EXISTS score_records (
    id            BIGSERIAL PRIMARY KEY,
    card_name     TEXT NOT NULL REFERENCES cards(name),
    version       INTEGER NOT NULL,
    request_id    TEXT,
    request_hash  TEXT NOT NULL,
    features      JSONB NOT NULL,
    result        JSONB NOT NULL,
    total_score   DOUBLE PRECISION NOT NULL,
    pd            DOUBLE PRECISION NOT NULL,
    label         SMALLINT,
    scored_at     TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS score_records_idem_idx
    ON score_records(card_name, request_id) WHERE request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS score_records_query_idx
    ON score_records(card_name, version, scored_at);
CREATE INDEX IF NOT EXISTS score_records_lookup_idx
    ON score_records(card_name, request_id);
CREATE TABLE IF NOT EXISTS backfill_jobs (
    id          BIGSERIAL PRIMARY KEY,
    card_name   TEXT NOT NULL REFERENCES cards(name),
    status      TEXT NOT NULL,
    total       INTEGER NOT NULL DEFAULT 0,
    processed   INTEGER NOT NULL DEFAULT 0,
    applied     INTEGER NOT NULL DEFAULT 0,
    duplicates  INTEGER NOT NULL DEFAULT 0,
    rejected    JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS backfill_jobs_card_idx
    ON backfill_jobs(card_name, id);
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
    def find_score_record(self, card_name: str,
                          request_id: str) -> dict | None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, card_name, version, request_id, request_hash,
                           features, result, total_score, pd, label, scored_at
                    FROM score_records
                    WHERE card_name=%s AND request_id=%s
                """, (card_name, request_id))
                row = cur.fetchone()
        return self._record_from_row(row) if row else None

    @staticmethod
    def _record_from_row(row) -> dict:
        result = row[6]
        return {
            "id": row[0], "card_name": row[1], "version": row[2],
            "request_id": row[3], "request_hash": row[4],
            "features": row[5], "result": result,
            "total_score": row[7], "pd": row[8],
            "features_detail": result["features"],
            "label": row[9],
            "scored_at": row[10],
        }

    @staticmethod
    def _record_view(row: dict) -> dict:
        return {
            "id": row["id"], "card_name": row["card_name"],
            "version": row["version"], "request_id": row["request_id"],
            "total_score": row["total_score"], "pd": row["pd"],
            "features": row["features_detail"], "label": row["label"],
            "scored_at": row["scored_at"],
        }

    def insert_score_record(
        self, card_name: str, version: int, request_id: str | None,
        request_hash: str, features: dict, result: dict,
        scored_at: datetime,
    ) -> tuple[str, dict]:
        if scored_at.tzinfo is None:
            scored_at = scored_at.replace(tzinfo=timezone.utc)
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    if request_id is None:
                        cur.execute("""
                            INSERT INTO score_records
                                (card_name, version, request_id, request_hash,
                                 features, result, total_score, pd, scored_at)
                            VALUES (%s, %s, NULL, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, (card_name, version, request_hash,
                              Jsonb(features), Jsonb(result),
                              result["total_score"], result["pd"], scored_at))
                        rec_id = cur.fetchone()[0]
                        status = "inserted"
                    else:
                        cur.execute("""
                            INSERT INTO score_records
                                (card_name, version, request_id, request_hash,
                                 features, result, total_score, pd, scored_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (card_name, request_id)
                            WHERE request_id IS NOT NULL
                            DO NOTHING
                            RETURNING id
                        """, (card_name, version, request_id, request_hash,
                              Jsonb(features), Jsonb(result),
                              result["total_score"], result["pd"], scored_at))
                        row = cur.fetchone()
                        status = "inserted" if row else "duplicate"
                        rec_id = row[0] if row else None
                    if status == "duplicate":
                        cur.execute("""
                            SELECT id, card_name, version, request_id,
                                   request_hash, features, result, total_score,
                                   pd, label, scored_at
                            FROM score_records
                            WHERE card_name=%s AND request_id=%s
                        """, (card_name, request_id))
                        existing_row = cur.fetchone()
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
        if status == "inserted":
            return "inserted", {
                "id": rec_id, "card_name": card_name, "version": version,
                "request_id": request_id, "request_hash": request_hash,
                "features": features, "result": result,
                "total_score": result["total_score"], "pd": result["pd"],
                "features_detail": result["features"], "label": None,
                "scored_at": scored_at,
            }
        existing = self._record_from_row(existing_row)
        if existing["request_hash"] != request_hash:
            raise IdempotencyConflict(
                f"请求标识 {request_id!r} 已存在但内容不同，拒绝覆盖")
        return "duplicate", existing

    def list_score_records(
        self, card_name: str, version: int,
        start: datetime | None = None, end: datetime | None = None,
        labeled: bool | None = None,
    ) -> list[dict]:
        sql = [
            "SELECT id, card_name, version, request_id, request_hash,",
            "       features, result, total_score, pd, label, scored_at",
            "FROM score_records WHERE card_name=%s AND version=%s",
        ]
        args: list = [card_name, version]
        if start is not None:
            sql.append("AND scored_at >= %s")
            args.append(start)
        if end is not None:
            sql.append("AND scored_at < %s")
            args.append(end)
        if labeled is True:
            sql.append("AND label IS NOT NULL")
        elif labeled is False:
            sql.append("AND label IS NULL")
        sql.append("ORDER BY id")
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(" ".join(sql), args)
                rows = cur.fetchall()
        return [self._record_view(self._record_from_row(r)) for r in rows]

    def count_score_records(
        self, card_name: str, version: int | None = None,
        start: datetime | None = None, end: datetime | None = None,
    ) -> int:
        sql = ["SELECT COUNT(*) FROM score_records WHERE card_name=%s"]
        args: list = [card_name]
        if version is not None:
            sql.append("AND version=%s")
            args.append(version)
        if start is not None:
            sql.append("AND scored_at >= %s")
            args.append(start)
        if end is not None:
            sql.append("AND scored_at < %s")
            args.append(end)
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(" ".join(sql), args)
                return int(cur.fetchone()[0])

    # ------------------------------------------------------------ 表现回填
    def apply_label(self, card_name: str, request_id: str,
                    label: int) -> str:
        with self._p().connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE score_records SET label=%s
                        WHERE card_name=%s AND request_id=%s
                          AND label IS NULL
                    """, (int(label), card_name, request_id))
                    if cur.rowcount == 1:
                        outcome = "applied"
                    else:
                        cur.execute("""
                            SELECT label FROM score_records
                            WHERE card_name=%s AND request_id=%s
                        """, (card_name, request_id))
                        row = cur.fetchone()
                        if row is None:
                            outcome = "missing"
                        elif int(row[0]) == int(label):
                            outcome = "duplicate"
                        else:
                            outcome = "conflict"
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return outcome

    def create_backfill_job(self, card_name: str,
                            items: list[dict]) -> int:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO backfill_jobs(card_name, status, total)
                    VALUES (%s, 'pending', %s) RETURNING id
                """, (card_name, len(items)))
                job_id = cur.fetchone()[0]
            conn.commit()
        return job_id

    def update_backfill_job(self, job_id: int, status: str, **fields) -> None:
        allowed = {"total", "processed", "applied", "duplicates", "rejected"}
        sets = ["status=%s", "updated_at=now()"]
        args: list = [status]
        for k, v in fields.items():
            if k in allowed:
                if k == "rejected":
                    sets.append(f"{k}=%s")
                    args.append(Jsonb(v))
                else:
                    sets.append(f"{k}=%s")
                    args.append(v)
        args.append(job_id)
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE backfill_jobs SET {', '.join(sets)} WHERE id=%s",
                    args)
            conn.commit()

    def _backfill_from_row(self, r) -> dict:
        return {
            "id": r[0], "card_name": r[1], "status": r[2], "total": r[3],
            "processed": r[4], "applied": r[5], "duplicates": r[6],
            "rejected": r[7] or [],
            "created_at": r[8].isoformat() if r[8] else None,
            "updated_at": r[9].isoformat() if r[9] else None,
        }

    _BACKFILL_COLS = (
        "id, card_name, status, total, processed, applied, duplicates, "
        "rejected, created_at, updated_at"
    )

    def get_backfill_job(self, job_id: int) -> dict | None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {self._BACKFILL_COLS} FROM backfill_jobs "
                    "WHERE id=%s",
                    (job_id,))
                row = cur.fetchone()
        return self._backfill_from_row(row) if row else None

    def list_backfill_jobs(self, card_name: str | None = None) -> list[dict]:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                if card_name:
                    cur.execute(
                        f"SELECT {self._BACKFILL_COLS} FROM backfill_jobs "
                        "WHERE card_name=%s ORDER BY id",
                        (card_name,))
                else:
                    cur.execute(
                        f"SELECT {self._BACKFILL_COLS} FROM backfill_jobs "
                        "ORDER BY id")
                rows = cur.fetchall()
        return [self._backfill_from_row(r) for r in rows]

    # ------------------------------------------------------------ 总分基准
    def set_score_baseline(self, card_name: str, version: int,
                           baseline: dict) -> None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE versions
                    SET artifacts = jsonb_set(artifacts, '{score_baseline}', %s),
                        summary = jsonb_set(summary, '{score_baseline}', 'true')
                    WHERE card_name=%s AND version=%s
                """, (Jsonb(baseline), card_name, version))
                if cur.rowcount == 0:
                    raise KeyError(f"卡 {card_name!r} 版本 {version} 不存在")
            conn.commit()

    def clear_score_baseline(self, card_name: str, version: int) -> None:
        with self._p().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE versions
                    SET artifacts = jsonb_set(artifacts, '{score_baseline}', 'null'),
                        summary = jsonb_set(summary, '{score_baseline}', 'false')
                    WHERE card_name=%s AND version=%s
                """, (card_name, version))
            conn.commit()
