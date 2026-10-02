"""跑在真实 PostgreSQL 16（docker compose 起的 db）上的监控集成测试。

运行方式：
    docker compose up -d db
    DATABASE_URL=postgresql://scorecard:scorecard@localhost:5432/scorecard \
        pytest tests/test_postgres_integration.py -q

检测不到数据库时全部 skip（普通 pytest 不依赖 DB）。这组用例与内存后端用同一
套业务服务（AuditService / MonitoringService / PerformanceService），验证：
- 内存与 PostgreSQL 两后端行为一致；
- 在「已有数据的老库」上重复 init_schema 平滑升级（建监控表、不丢老数据）；
- 幂等、跨事务并发重放只落一条记录；
- 监控查询、回填、KS/AUC 口径在真库上同样成立。
"""
from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PG_TESTS") != "1",
    reason="需要真实 PostgreSQL：docker compose up -d db 后 RUN_PG_TESTS=1 运行")

from app.audit.service import AuditService
from app.core.metrics import ks_stat as ks_fn
from app.core.metrics import roc_auc as auc_fn
from app.monitoring.service import MonitoringService
from app.performance.scheduler import BackfillScheduler, validate_items
from app.performance.service import PerformanceService
from app.scoring.engine import ScorecardRuntime
from app.storage.postgres import PostgresRepository

DSN = os.getenv("DATABASE_URL",
                "postgresql://scorecard:scorecard@localhost:5432/scorecard")

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "dev_sample.csv")


@pytest.fixture(scope="module")
def repo():
    r = PostgresRepository(DSN)
    try:
        r.init_schema(retries=2, delay=0.5)
    except Exception as exc:
        pytest.skip(f"PostgreSQL 不可用：{exc}")
    yield r


@pytest.fixture()
def card(repo):
    name = f"pg_it_{int(time.time()*1000)}_{threading.get_ident()}"
    repo.ensure_card(name)
    yield name


def _build_artifacts():
    """在内存里跑一遍建卡流水线，把产物直接存入 PG（绕过 HTTP/作业线程）。"""
    from app.core.pipeline import BuildParams, run_pipeline
    from app.core.sample import parse_csv
    with open(DATA, encoding="utf-8") as f:
        sample = parse_csv(f.read(), label_col="label")
    return run_pipeline(sample, BuildParams())


def _rows():
    with open(DATA, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _features(row):
    def num(v):
        v = (v or "").strip()
        return None if v == "" else float(v)
    return {"age": num(row.get("age")), "income": num(row.get("income")),
            "debt_ratio": num(row.get("debt_ratio")),
            "city": (row.get("city") or "").strip() or None,
            "job_grade": (row.get("job_grade") or "").strip() or None}


@pytest.fixture()
def built(repo, card):
    job_id = repo.create_job(card, {"smoke": True})
    art = _build_artifacts()
    version = repo.save_version(card, art, job_id)
    rt = ScorecardRuntime(art)
    return card, version, art, rt


# ------------------------------------------------------------ 平滑升级

def test_schema_init_idempotent_on_existing_db(repo, built):
    # 已有版本/作业数据的库上再次初始化：只 CREATE IF NOT EXISTS，数据原样保留
    before_versions = repo.list_versions(built[0])
    repo.init_schema(retries=2, delay=0.5)
    repo.init_schema(retries=2, delay=0.5)
    after_versions = repo.list_versions(built[0])
    assert before_versions == after_versions


# ------------------------------------------------------------ 留痕与幂等

def test_idempotent_replay_persists_single_log(repo, built):
    card, version, art, rt = built
    svc = AuditService(repo)
    feat = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}
    first, replayed = svc.score_single(card, version, rt, feat, "PG-R1")
    assert replayed is False
    for _ in range(20):
        out, rp = svc.score_single(card, version, rt, feat, "PG-R1")
        assert rp is True and out == first
    logs = repo.query_score_logs(card, version, None, None, 1000)
    assert len(logs) == 1 and logs[0]["request_id"] == "PG-R1"


def test_concurrent_replays_single_insert(repo, built):
    card, version, art, rt = built
    svc = AuditService(repo)
    feat = {"age": 33, "income": 7000, "city": "SH", "job_grade": "C"}
    results, errors = [], []

    def worker():
        try:
            results.append(svc.score_single(card, version, rt, feat, "PG-RACE"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    assert len({r[0]["total_score"] for r in results}) == 1
    assert sum(1 for _, rp in results if rp) == 31
    logs = repo.query_score_logs(card, version, None, None, 1000)
    assert [l["request_id"] for l in logs].count("PG-RACE") == 1


def test_conflict_rejected_and_preserved(repo, built):
    card, version, art, rt = built
    svc = AuditService(repo)
    feat = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}
    first, _ = svc.score_single(card, version, rt, feat, "PG-C")
    from app.audit.service import IdempotentConflict
    with pytest.raises(IdempotentConflict):
        svc.score_single(card, version, rt, {**feat, "age": 41}, "PG-C")
    again, rp = svc.score_single(card, version, rt, feat, "PG-C")
    assert rp is True and again == first


def test_batch_failures_not_logged_pg(repo, built):
    card, version, art, rt = built
    svc = AuditService(repo)
    out = svc.score_batch(card, version, rt, [
        {"applicant_id": "ok1", "features": {"age": 40, "income": 9000,
                                             "city": "BJ", "job_grade": "A"}},
        {"applicant_id": "bad", "features": {"age": "oops", "income": 1,
                                             "city": "BJ", "job_grade": "A"}},
    ])
    assert [x["ok"] for x in out] == [True, False]
    logs = repo.query_score_logs(card, version, None, None, 1000)
    assert len(logs) == 1


# ------------------------------------------------------------ 稳定性

def test_training_self_psi_zero_on_pg(repo, built):
    card, version, art, rt = built
    svc = AuditService(repo)
    rows = _rows()
    for i, row in enumerate(rows):
        res, rp = svc.score_single(card, version, rt, _features(row),
                                   f"PG-T{i}")
        assert rp is False
    st = MonitoringService(repo).stability(card, version, None, None)
    assert st["n_scored"] == len(rows)
    for f in st["features"]:
        assert f["psi"] == pytest.approx(0.0, abs=1e-12)
    assert st["total_score"]["psi"] == pytest.approx(0.0, abs=1e-12)
    assert st["total_score"]["out_of_range_count"] == 0
    assert [b["actual_count"] for b in st["total_score"]["bins"]] == \
        art["monitoring"]["score_baseline"]["counts"]


def test_versions_isolated_on_pg(repo, card):
    job_id = repo.create_job(card, {"v": "b"})
    art = _build_artifacts()
    v1 = repo.save_version(card, art, job_id)
    rt1 = ScorecardRuntime(art)
    job_id2 = repo.create_job(card, {"v": "b2"})
    art2 = _build_artifacts()
    v2 = repo.save_version(card, art2, job_id2)
    rt2 = ScorecardRuntime(art2)
    svc = AuditService(repo)
    feat = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}
    for _ in range(3):
        svc.score_single(card, v1, rt1, feat, None)
    for _ in range(5):
        svc.score_single(card, v2, rt2, feat, None)
    s1 = MonitoringService(repo).stability(card, v1, None, None)
    s2 = MonitoringService(repo).stability(card, v2, None, None)
    assert s1["n_scored"] == 3 and s2["n_scored"] == 5


def test_time_window_pg(repo, built):
    card, version, art, rt = built
    AuditService(repo).score_single(
        card, version, rt, {"age": 40, "income": 9000, "city": "BJ",
                            "job_grade": "A"}, None)
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    logs = repo.query_score_logs(
        card, version, now - timedelta(minutes=1), now + timedelta(minutes=1),
        100)
    assert len(logs) == 1
    future = repo.query_score_logs(
        card, version, now + timedelta(days=1), now + timedelta(days=2), 100)
    assert future == []


# ------------------------------------------------------------ 回填与表现

def test_backfill_and_metrics_on_pg(repo, built):
    card, version, art, rt = built
    svc = AuditService(repo)
    rows = _rows()
    scores = []
    for i, row in enumerate(rows):
        res, _ = svc.score_single(card, version, rt, _features(row),
                                  f"PG-P{i}")
        scores.append(res["pd"])
    valid, invalid = validate_items(
        [{"request_id": f"PG-P{i}", "label": int(rows[i]["label"])}
         for i in range(len(rows))]
        + [{"request_id": "ghost", "label": 1},
           {"request_id": "PG-P0", "label": "bad"}])
    assert invalid and not any(x["reason"] == "not_found" for x in invalid)

    sched = BackfillScheduler(repo)
    job_id = repo.create_backfill_job(card, {"items": valid})
    sched.submit(job_id, card, valid)
    for _ in range(300):
        j = repo.get_backfill_job(job_id)
        if j["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)
    assert j["status"] == "succeeded", j
    assert j["applied"] == len(rows)
    assert [x["reason"] for x in j["rejected"]] == ["not_found"]

    perf = PerformanceService(repo).report(card, version, None, None)
    assert perf["n_performed"] == len(rows)
    import numpy as np
    y = np.array([int(r["label"]) for r in rows], float)
    p = np.array(scores, float)
    assert perf["ks"] == ks_fn(y, p) == pytest.approx(art["metrics"]["ks"])
    assert perf["auc"] == auc_fn(y, p) == pytest.approx(art["metrics"]["auc"])
    assert sum(b["count"] for b in perf["score_bands"]) == len(rows)


def test_backfill_conflict_and_duplicate_on_pg(repo, built):
    card, version, art, rt = built
    svc = AuditService(repo)
    rows = _rows()
    svc.score_single(card, version, rt, _features(rows[0]), "PG-D0")
    lab0 = int(rows[0]["label"])
    r1 = repo.apply_perf_labels(1, card, [{"request_id": "PG-D0",
                                           "label": lab0}])
    assert r1["applied"] == 1
    r2 = repo.apply_perf_labels(2, card, [{"request_id": "PG-D0",
                                           "label": lab0}])
    assert r2["duplicated"] == 1 and r2["applied"] == 0
    r3 = repo.apply_perf_labels(3, card, [{"request_id": "PG-D0",
                                           "label": 1 - lab0}])
    assert r3["rejected"][0]["reason"] == "label_conflict"


def test_restart_invariant_on_pg(repo, built):
    """模拟服务重启：丢掉全部服务对象，重新建仓储连接后查询结果不变。"""
    card, version, art, rt = built
    svc = AuditService(repo)
    feat = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}
    for i in range(10):
        svc.score_single(card, version, rt, feat, f"PG-K{i}")
    st1 = MonitoringService(repo).stability(card, version, None, None)

    repo2 = PostgresRepository(DSN)  # 新连接池 = 进程重启后的仓储
    repo2.init_schema(retries=2, delay=0.5)
    st2 = MonitoringService(repo2).stability(card, version, None, None)
    assert st1["n_scored"] == st2["n_scored"] == 10
    assert st1["features"][0]["bins"] == st2["features"][0]["bins"]
    assert st1["total_score"]["psi"] == st2["total_score"]["psi"]
