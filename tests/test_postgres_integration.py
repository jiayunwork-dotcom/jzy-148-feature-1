"""跑在真实 PostgreSQL 16（docker compose 起的 db）上的集成用例。

运行方式：
    docker compose up -d db
    pytest tests/test_postgres_integration.py
默认连 postgresql://scorecard:scorecard@localhost:5432/scorecard，可用
TEST_DATABASE_URL 覆盖。数据库不可达时整模块 skip（普通内存测试不受影响）。

覆盖与内存用例同一批验收性质，并额外验证：
- 8 线程在服务层直接并发打 2000 分：正好 2000，且能和留痕重算对上
  （TestClient 的 portal 会串行化请求，真正的库级并发必须绕过 HTTP）；
- 同标识 100 个并发重放：只插一条；
- 在**旧 schema**（只有 cards/jobs/versions、且已有数据）上跑新 init_schema：
  老数据完整保留、新表可用（平滑升级，不清库）；
- 用一个全新的 PostgresRepository 实例重连同一库（模拟服务重启），查询不变。
"""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psycopg
import pytest

from app.core.metrics import ks_stat, roc_auc
from app.core.pipeline import BuildParams, run_pipeline, _sample_rows_as_applicants
from app.core.sample import parse_csv
from app.monitor.audit import RequestIdConflict, ScoreAudit
from app.monitor.backfill import BackfillScheduler
from app.scoring.engine import ScorecardRuntime
from app.storage.postgres import PostgresRepository

DSN = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://scorecard:scorecard@localhost:5432/scorecard",
)
ADMIN_DSN = os.getenv("TEST_ADMIN_DSN", DSN.rsplit("/", 1)[0] + "/postgres")
TEST_DB = os.getenv("TEST_DATABASE_NAME", "scorecard_monitor_test")


def _db_alive(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_alive(ADMIN_DSN),
    reason="真实 PostgreSQL 不可达（设置 TEST_DATABASE_URL 或 docker compose up -d db）",
)


@pytest.fixture()
def dsn():
    """每个用例跑在一个全新的测试库上（约百毫秒，保证版本号/计数隔离）。"""
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname=%s AND pid<>pg_backend_pid()", (TEST_DB,))
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        conn.execute(f"CREATE DATABASE {TEST_DB}")
    d = DSN.rsplit("/", 1)[0] + "/" + TEST_DB
    yield d
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname=%s AND pid<>pg_backend_pid()", (TEST_DB,))
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")


@pytest.fixture()
def repo(dsn):
    r = PostgresRepository(dsn)
    r.init_schema()
    yield r
    if r._pool is not None:
        r._pool.close()


@pytest.fixture()
def services(repo):
    from app.deps import RuntimeCache
    runtimes = RuntimeCache(repo)
    auditor = ScoreAudit(repo, runtimes)
    backfills = BackfillScheduler(repo)
    yield repo, runtimes, auditor, backfills
    backfills.shutdown()


@pytest.fixture(scope="module")
def sample():
    rows = list(csv.DictReader(open("data/dev_sample.csv", newline="")))
    parsed = parse_csv(open("data/dev_sample.csv", "rb").read())
    return rows, parsed


CARD = "pg_card"
FEAT = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}


def _features_from_row(row: dict) -> dict:
    missing = {"", "na", "null", "none", "nan"}
    out = {}
    for k, v in row.items():
        if k == "label":
            continue
        token = v.strip().lower()
        if token in missing:
            out[k] = None
        elif k in ("age", "income", "debt_ratio"):
            out[k] = float(v)
        else:
            out[k] = v.strip()
    return out


def _build_version(repo, parsed):
    artifacts = run_pipeline(parsed, BuildParams())
    repo.ensure_card(CARD)
    job_id = repo.create_job(CARD, {"params": {}})
    return repo.save_version(CARD, artifacts, job_id), artifacts


# ============================================================ 基础留痕
def test_score_record_persisted_and_response_shape(repo, services, sample):
    _, runtimes, auditor, _ = services
    _, parsed = sample
    _build_version(repo, parsed)
    outcome = auditor.score_single(CARD, {"features": FEAT, "request_id": "k1"})
    assert outcome.status == "inserted"
    rec = repo.find_score_record(CARD, "k1")
    assert rec["version"] == 1
    assert rec["total_score"] == outcome.result["total_score"]
    assert abs(rec["pd"] - outcome.result["pd"]) < 1e-15
    # 留痕里逐特征带 bin_index / status
    assert {"bin_index", "bin_label", "status"} <= set(rec["result"]["features"][0])


def test_replay_identical_and_counts_once(repo, services, sample):
    _, runtimes, auditor, _ = services
    _, parsed = sample
    _build_version(repo, parsed)
    first = auditor.score_single(CARD, {"features": FEAT, "request_id": "r1"})
    for _ in range(99):
        o = auditor.score_single(CARD, {"features": FEAT, "request_id": "r1"})
        assert o.status == "duplicate"
        assert o.result["total_score"] == first.result["total_score"]
        assert abs(o.result["pd"] - first.result["pd"]) == 0.0
    assert repo.count_score_records(CARD, 1) == 1


def test_same_id_different_content_conflict(repo, services, sample):
    _, _, auditor, _ = services
    _, parsed = sample
    _build_version(repo, parsed)
    auditor.score_single(CARD, {"features": FEAT, "request_id": "r2"})
    with pytest.raises(RequestIdConflict):
        auditor.score_single(CARD, {"features": {**FEAT, "age": 41},
                                    "request_id": "r2"})
    assert repo.count_score_records(CARD, 1) == 1


# ============================================================ 真并发
def test_8_threads_2000_scores_exact_count(repo, services, sample):
    _, _, auditor, _ = services
    _, parsed = sample
    _build_version(repo, parsed)

    def hit(i):
        return auditor.score_single(
            CARD, {"features": FEAT, "request_id": f"c-{i}"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(hit, range(2000)))
    assert all(o.status == "inserted" for o in outs)
    assert repo.count_score_records(CARD, 1) == 2000
    # 与留痕逐条重算一致：主键 id 无重复、标识 2000 个
    recs = repo.list_score_records(CARD, 1)
    assert len(recs) == 2000
    assert len({r["id"] for r in recs}) == 2000
    assert len({r["request_id"] for r in recs}) == 2000


def test_8_threads_same_id_100_replays_insert_once(repo, services, sample):
    _, _, auditor, _ = services
    _, parsed = sample
    _build_version(repo, parsed)
    first = auditor.score_single(CARD, {"features": FEAT, "request_id": "hot"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(
            lambda _: auditor.score_single(
                CARD, {"features": FEAT, "request_id": "hot"}),
            range(100)))
    assert all(o.result["total_score"] == first.result["total_score"]
               for o in outs)
    assert repo.count_score_records(CARD, 1) == 1


# ============================================================ 稳定性
def test_training_replay_psi_exactly_zero(repo, services, sample):
    _, _, auditor, _ = services
    rows, parsed = sample
    _, artifacts = _build_version(repo, parsed)
    app_rows = _sample_rows_as_applicants(parsed)
    # 并发分批回放训练样本
    def one(i):
        return auditor.score_single(
            CARD, {"features": app_rows[i], "request_id": f"tr-{i}"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(len(app_rows))))
    from app.monitor.stability import stability_report
    recs = repo.list_score_records(CARD, 1)
    rep = stability_report(artifacts, recs)
    assert rep["n_scored"] == parsed.n
    for f in rep["features"]:
        assert f["psi"] == pytest.approx(0.0, abs=1e-12)
        for b in f["bins"]:
            assert b["actual_count"] == b["baseline_count"]
    assert rep["total_score"]["psi"] == pytest.approx(0.0, abs=1e-12)


def test_version_isolation(repo, services, sample):
    _, _, auditor, _ = services
    _, parsed = sample
    _, a1 = _build_version(repo, parsed)
    # 第二个版本：换基准分
    p2 = BuildParams(base_score=700.0)
    a2 = run_pipeline(parsed, p2)
    v2 = repo.save_version(CARD, a2, repo.create_job(CARD, {}))
    assert v2 == 2
    for i in range(40):
        auditor.score_single(CARD, {"version": 1, "features": FEAT,
                                    "request_id": f"a-{i}"})
        auditor.score_single(CARD, {"version": 2, "features": FEAT,
                                    "request_id": f"b-{i}"})
    assert repo.count_score_records(CARD, 1) == 40
    assert repo.count_score_records(CARD, 2) == 40
    from app.monitor.stability import stability_report
    r1 = stability_report(a1, repo.list_score_records(CARD, 1))
    r2 = stability_report(a2, repo.list_score_records(CARD, 2))
    assert r1["n_scored"] == 40 and r2["n_scored"] == 40


# ============================================================ 回填/表现
def test_backfill_and_performance_metrics(repo, services, sample):
    _, _, auditor, backfills = services
    rows, parsed = sample
    _, artifacts = _build_version(repo, parsed)
    app_rows = _sample_rows_as_applicants(parsed)
    labels = [int(v) for v in parsed.label]
    for i, fr in enumerate(app_rows):
        auditor.score_single(CARD, {"features": fr, "request_id": f"p-{i}"})
    items = [{"request_id": f"p-{i}", "label": labels[i]}
             for i in range(len(labels))]
    job_id = repo.create_backfill_job(CARD, items)
    backfills.submit(job_id, CARD, items)
    for _ in range(500):
        job = repo.get_backfill_job(job_id)
        if job["status"] == "finished":
            break
        time.sleep(0.02)
    assert job["status"] == "finished"
    assert job["applied"] == len(labels) and job["rejected"] == []

    # 拒收：找不到 / 非法标签 / 冲突；同标签幂等
    items2 = [
        {"request_id": "nope", "label": 1},
        {"request_id": "p-0", "label": 9},
        {"request_id": "p-1", "label": 1 - labels[1]},
        {"request_id": "p-2", "label": labels[2]},
    ]
    j2 = repo.create_backfill_job(CARD, items2)
    backfills.submit(j2, CARD, items2)
    for _ in range(200):
        job2 = repo.get_backfill_job(j2)
        if job2["status"] == "finished":
            break
        time.sleep(0.02)
    assert job2["applied"] == 0 and job2["duplicates"] == 1
    assert len(job2["rejected"]) == 3

    from app.monitor.performance import performance_report
    labeled = repo.list_score_records(CARD, 1, labeled=True)
    rows_rep = [{"total_score": r["total_score"], "pd": r["pd"],
                 "label": r["label"]} for r in labeled]
    perf = performance_report(rows_rep, artifacts["score_baseline"])
    p = np.array([r["pd"] for r in rows_rep], dtype=np.float64)
    y = np.array([r["label"] for r in rows_rep], dtype=np.float64)
    assert perf["ks"] == pytest.approx(ks_stat(y, p), abs=1e-12)
    assert perf["auc"] == pytest.approx(roc_auc(y, p), abs=1e-12)
    assert sum(s["count"] for s in perf["segments"]) == len(rows_rep)


# ============================================================ 重启不变
def test_restart_new_repo_instance_same_results(repo, services, sample, dsn):
    _, _, auditor, _ = services
    _, parsed = sample
    _, artifacts = _build_version(repo, parsed)
    for i in range(50):
        auditor.score_single(CARD, {"features": FEAT, "request_id": f"s-{i}"})
    if repo._pool is not None:
        repo._pool.close()
        repo._pool = None
    fresh = PostgresRepository(dsn)
    fresh.init_schema()  # 重启后再跑一次建表：全部 IF NOT EXISTS，无副作用
    assert fresh.count_score_records(CARD, 1) == 50
    # 幂等在重启后仍然成立：原结果逐位取回
    from app.deps import RuntimeCache
    o = ScoreAudit(fresh, RuntimeCache(fresh)).score_single(
        CARD, {"features": FEAT, "request_id": "s-0"})
    assert o.status == "duplicate"
    from app.monitor.stability import stability_report
    rep = stability_report(artifacts, fresh.list_score_records(CARD, 1))
    assert rep["n_scored"] == 50
    fresh._pool.close()


# ============================================================ 平滑升级
def test_smooth_upgrade_from_legacy_schema(dsn, sample):
    """旧库只有 cards/jobs/versions 且已有数据：新 init_schema 只加表不毁数据。"""
    legacy = PostgresRepository(dsn)
    with legacy._p().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cards (
                    name TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now())
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id BIGSERIAL PRIMARY KEY, card_name TEXT NOT NULL,
                    status TEXT NOT NULL, request JSONB NOT NULL,
                    error TEXT, newton_history JSONB, version INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now())
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    card_name TEXT NOT NULL, version INTEGER NOT NULL,
                    job_id BIGINT NOT NULL, status TEXT NOT NULL,
                    summary JSONB NOT NULL, artifacts JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (card_name, version))
            """)
            cur.execute("INSERT INTO cards(name) VALUES ('legacy_card') "
                        "ON CONFLICT DO NOTHING")
        conn.commit()

    # 老版本产物：没有 score_baseline（升级前的真实形态）
    _, parsed = sample
    art = run_pipeline(parsed, BuildParams())
    art.pop("score_baseline", None)
    job_id = legacy.create_job("legacy_card", {"params": {}})
    version = legacy.save_version("legacy_card", art, job_id)
    legacy._pool.close()

    # ---- 升级：新代码在同一库上 init_schema ----
    upgraded = PostgresRepository(dsn)
    upgraded.init_schema()
    # 老数据完好
    cards = upgraded.list_cards()
    assert any(c["card_name"] == "legacy_card" for c in cards)
    loaded = upgraded.load_version("legacy_card", version)
    assert loaded["selected_features"] == art["selected_features"]
    assert loaded.get("score_baseline") is None
    # 新能力直接可用：打分留痕/幂等
    from app.deps import RuntimeCache
    auditor = ScoreAudit(upgraded, RuntimeCache(upgraded))
    auditor.score_single("legacy_card", {"features": FEAT, "request_id": "L1"})
    auditor.score_single("legacy_card", {"features": FEAT, "request_id": "L1"})
    assert upgraded.count_score_records("legacy_card", 1) == 1
    # 老版本特征层 PSI 照算、总分层不可用
    from app.monitor.stability import stability_report
    rep = stability_report(loaded, upgraded.list_score_records("legacy_card", 1))
    assert rep["features"] and rep["total_score"]["available"] is False
    upgraded._pool.close()
