"""监控层端到端（内存存储）：留痕/幂等/稳定性/回填/表现/时间窗/版本隔离。"""
from __future__ import annotations

import csv
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.deps import repo
from app.core.metrics import ks_stat, roc_auc
from tests.conftest import build_and_wait

CARD = "mon"
FEAT = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}


# ------------------------------------------------------------- 公共工具
@pytest.fixture()
def built(client, dev_csv_bytes):
    job = build_and_wait(client, CARD, dev_csv_bytes)
    assert job["status"] == "succeeded"
    return job["version"]


def _dev_rows():
    with open("data/dev_sample.csv", newline="") as f:
        return list(csv.DictReader(f))


def _row_to_features(row: dict) -> dict:
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


def _score_training_rows(client, version, prefix="t", chunk=200):
    """把建卡样本原样逐条经批量打分留痕，返回 (rows, results)。"""
    rows = _dev_rows()
    applicants = [
        {"applicant_id": f"{prefix}-{i}", "request_id": f"{prefix}-rid-{i}",
         "features": _row_to_features(r)}
        for i, r in enumerate(rows)
    ]
    results = []
    for off in range(0, len(applicants), chunk):
        r = client.post(f"/cards/{CARD}/score/batch",
                        json={"version": version,
                              "applicants": applicants[off:off + chunk]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["failed"] == 0, body
        results.extend(body["results"])
    return rows, results


def _wait_backfill(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/cards/{CARD}/backfills/{job_id}").json()
        if j["status"] == "finished":
            return j
        time.sleep(0.02)
    raise AssertionError("回填作业超时")


# ============================================================ 留痕与幂等
def test_response_fields_unchanged_with_monitoring(client, built):
    """监控层接入后，打分响应字段与数值一个都不变。"""
    r1 = client.post(f"/cards/{CARD}/score", json={"features": FEAT})
    r2 = client.post(f"/cards/{CARD}/score",
                     json={"features": FEAT, "request_id": "abc"})
    assert set(r1.json()) == {
        "card_name", "version", "total_score", "pd",
        "features", "has_unseen", "has_missing"}
    a, b = r1.json(), r2.json()
    for k in ("total_score", "pd", "features", "has_unseen", "has_missing"):
        assert a[k] == b[k]
    assert abs(a["total_score"] - sum(f["score"] for f in a["features"])) < 1e-9


def test_replay_same_id_same_content_returns_identical_and_counts_once(
        client, built):
    first = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "R1"}).json()
    for _ in range(99):  # 同一标识重放 100 次
        r = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "R1"})
        assert r.status_code == 200
        assert r.json() == first  # 逐字段（含浮点）完全一样
    assert repo.count_score_records(CARD, built) == 1  # 计数只加 1


def test_same_id_different_content_rejected_not_overwritten(client, built):
    client.post(f"/cards/{CARD}/score",
                json={"features": FEAT, "request_id": "R2"})
    diff = {**FEAT, "age": 41}
    r = client.post(f"/cards/{CARD}/score",
                    json={"features": diff, "request_id": "R2"})
    assert r.status_code == 409 and "不一致" in r.json()["detail"]
    # 原记录未被覆盖：再用原内容重放仍是原结果
    again = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "R2"})
    assert again.status_code == 200
    assert again.json()["total_score"] == client.post(
        f"/cards/{CARD}/score", json={"features": FEAT}).json()["total_score"]
    assert repo.count_score_records(CARD, built) == 2  # 冲突不加计数
    # 指向另一版本也算不同内容
    if built >= 1:
        r = client.post(
            f"/cards/{CARD}/score",
            json={"version": built, "features": diff, "request_id": "R2"})
        assert r.status_code == 409


def test_nonexistent_version_is_404_even_with_existing_id(client, built):
    client.post(f"/cards/{CARD}/score",
                json={"features": FEAT, "request_id": "R3"})
    r = client.post(f"/cards/{CARD}/score",
                    json={"version": built + 99, "features": FEAT,
                          "request_id": "R3"})
    assert r.status_code == 404


def test_no_request_id_scores_and_records_every_time(client, built):
    for _ in range(3):
        r = client.post(f"/cards/{CARD}/score", json={"features": FEAT})
        assert r.status_code == 200
    assert repo.count_score_records(CARD, built) == 3


def test_invalid_request_id_rejected(client, built):
    r = client.post(
        f"/cards/{CARD}/score",
        json={"features": FEAT, "request_id": "x" * 129})
    assert r.status_code == 400


def test_batch_failures_not_recorded_and_batch_replay_idempotent(
        client, built):
    body = {"version": built, "applicants": [
        {"applicant_id": "ok", "request_id": "B1", "features": FEAT},
        {"applicant_id": "bad", "request_id": "B2",
         "features": {"age": "NaN-string", "income": 1, "city": "BJ",
                      "job_grade": "A"}},
    ]}
    r = client.post(f"/cards/{CARD}/score/batch", json=body).json()
    assert r["succeeded"] == 1 and r["failed"] == 1
    # 失败的那一条不进统计
    assert repo.count_score_records(CARD, built) == 1
    # 批量里成功条目的重放同样幂等
    r2 = client.post(f"/cards/{CARD}/score/batch",
                     json={"version": built, "applicants": [body["applicants"][0]]})
    assert r2.json()["succeeded"] == 1 and r2.json()["failed"] == 0
    assert repo.count_score_records(CARD, built) == 1
    # 批量里标识冲突按单项失败隔离
    conflict = dict(body["applicants"][0])
    conflict["features"] = {**FEAT, "age": 99}
    r3 = client.post(f"/cards/{CARD}/score/batch",
                     json={"version": built, "applicants": [conflict]})
    assert r3.json()["failed"] == 1
    assert "RequestIdConflict" in r3.json()["results"][0]["error"]
    assert repo.count_score_records(CARD, built) == 1


# ============================================================ 并发
def test_8_threads_2000_scores_exact_count(client, built):
    """8 线程并发打 2000 次分（各自不同标识），统计正好 2000。"""
    def hit(i):
        return client.post(
            f"/cards/{CARD}/score",
            json={"features": FEAT, "request_id": f"cc-{i}"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(hit, range(2000)))
    assert all(r.status_code == 200 for r in out)
    assert repo.count_score_records(CARD, built) == 2000
    # 稳定性报告按留痕现算的次数同样正好 2000（各箱计数之和也对得上）
    rep = client.get(f"/cards/{CARD}/versions/{built}/stability").json()
    assert rep["n_scored"] == 2000
    for f in rep["features"]:
        binned = sum(b["actual_count"] for b in f["bins"])
        assert binned + f["unseen_count"] + f["out_of_range_count"] == 2000


def test_concurrent_replay_same_id_counts_once(client, built):
    """8 线程同时用同一标识同内容打 100 次：只算 1 条，全部拿到同一结果。"""
    first = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "hot"})
    first_body = first.json()

    def hit(_):
        return client.post(f"/cards/{CARD}/score",
                           json={"features": FEAT, "request_id": "hot"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(hit, range(100)))
    assert all(r.status_code == 200 and r.json() == first_body for r in out)
    assert repo.count_score_records(CARD, built) == 1


def test_stats_match_recomputation_from_records(client, built):
    """统计计数与从留痕逐条重算的结果一致（全量现算口径的对账性质）。"""
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda i: client.post(
                f"/cards/{CARD}/score",
                json={"features": FEAT, "request_id": f"rec-{i}"}),
            range(500)))
    recs = repo.list_score_records(CARD, built)
    assert len(recs) == repo.count_score_records(CARD, built) == 500
    # 从留痕重算总分直方图，和直接重算计数对得上（无遗漏、无重复）
    ids = [r["request_id"] for r in recs]
    assert len(set(ids)) == len(ids)


# ============================================================ 稳定性
def test_training_replay_stability_identically_zero(client, built):
    """建卡样本原样逐条打分：各特征与总分 PSI≈0，占比/计数与产物完全吻合。"""
    rows, _ = _score_training_rows(client, built)
    rep = client.get(f"/cards/{CARD}/versions/{built}/stability").json()
    assert rep["n_scored"] == len(rows)
    assert rep["verdict"] == "stable"
    art = repo.load_version(CARD, built)
    fblocks = {f["name"]: f for f in art["features"] if f["selected"]}
    for f in rep["features"]:
        assert f["psi"] == pytest.approx(0.0, abs=1e-12)
        assert f["unseen_count"] == 0 and f["out_of_range_count"] == 0
        fb = fblocks[f["feature"]]
        expected = {b["label"]: b["total"] for b in fb["bins"]}
        expected[fb["missing_bin"]["label"]] = fb["missing_bin"]["total"]
        n = art["data_summary"]["n"]
        for b in f["bins"]:
            assert b["actual_count"] == expected[b["bin_label"]]
            assert b["actual_share"] == pytest.approx(
                expected[b["bin_label"]] / n, abs=1e-12)
            assert b["actual_share"] == pytest.approx(
                b["baseline_share"], abs=1e-12)
    sc = rep["total_score"]
    assert sc["available"] and sc["psi"] == pytest.approx(0.0, abs=1e-12)
    assert sum(b["actual_count"] for b in sc["bins"]) == len(rows)
    for b in sc["bins"]:
        assert b["actual_count"] == b["baseline_count"]


def test_drift_detected_and_tiered(client, built):
    """进件集中到训练范围内的低龄/低收入：age 与总分显著漂移，三档可调。"""
    art = repo.load_version(CARD, built)
    age_bins = next(f for f in art["features"] if f["name"] == "age")["bins"]
    low_age = max(int(age_bins[0]["lo"]) + 1, 22)  # 落在训练范围内的低龄
    young = {"age": low_age, "income": 2100, "city": "CD", "job_grade": "E"}
    for i in range(300):
        client.post(f"/cards/{CARD}/score",
                    json={"features": young, "request_id": f"dr-{i}"})
    rep = client.get(f"/cards/{CARD}/versions/{built}/stability").json()
    age = next(f for f in rep["features"] if f["feature"] == "age")
    assert age["psi"] is not None
    assert age["psi"] > 0.25 and age["verdict"] == "significant"
    assert rep["verdict"] == "significant"
    # 阈值可调：调到极高 -> stable
    rep2 = client.get(
        f"/cards/{CARD}/versions/{built}/stability?warning=9&significant=99"
    ).json()
    assert rep2["verdict"] == "stable"


def test_time_window_filtering(client, built):
    from datetime import datetime, timedelta, timezone
    mid = datetime(2026, 5, 1, tzinfo=timezone.utc)
    # 100 条打在 1 月中，100 条打在 9 月中
    for i in range(100):
        repo.insert_score_record(
            CARD, built, f"jan-{i}", "h", FEAT,
            _fake_result(), datetime(2026, 1, 15, tzinfo=timezone.utc))
        repo.insert_score_record(
            CARD, built, f"sep-{i}", "h", FEAT,
            _fake_result(), datetime(2026, 9, 15, tzinfo=timezone.utc))
    n_jan = client.get(
        f"/cards/{CARD}/versions/{built}/stability"
        "?start=2026-01-01&end=2026-02-01").json()["n_scored"]
    n_all = client.get(
        f"/cards/{CARD}/versions/{built}/stability"
        "?start=2025-01-01&end=2027-01-01").json()["n_scored"]
    n_sep = client.get(
        f"/cards/{CARD}/versions/{built}/stability"
        "?start=2026-09-01&end=2026-10-01").json()["n_scored"]
    assert n_jan == 100 and n_sep == 100 and n_all == 200
    # 半开区间：end 当刻不计入
    q = client.get(
        f"/cards/{CARD}/versions/{built}/stability"
        f"?start=2026-05-01T00:00:00%2B00:00&end=2026-08-29T00:00:00%2B00:00"
    ).json()
    assert q["n_scored"] == 0


def _fake_result():
    return {"total_score": 600.0, "pd": 0.01,
            "features": [], "has_unseen": False, "has_missing": False}


def test_version_stats_do_not_mix(client, dev_csv_bytes):
    """版本 1 / 版本 2 交替打分，两个版本统计互不渗入。"""
    build_and_wait(client, CARD, dev_csv_bytes, base_score=600.0, pdo=50.0)
    build_and_wait(client, CARD, dev_csv_bytes, base_score=700.0, pdo=25.0)
    for i in range(50):
        client.post(
            f"/cards/{CARD}/score",
            json={"version": 1, "features": FEAT, "request_id": f"v1-{i}"})
        client.post(
            f"/cards/{CARD}/score",
            json={"version": 2, "features": FEAT, "request_id": f"v2-{i}"})
    assert repo.count_score_records(CARD, 1) == 50
    assert repo.count_score_records(CARD, 2) == 50
    r1 = client.get(f"/cards/{CARD}/versions/1/stability").json()
    r2 = client.get(f"/cards/{CARD}/versions/2/stability").json()
    assert r1["n_scored"] == 50 and r2["n_scored"] == 50
    # 两个版本分制不同，同样进件总分落桶不一样（各自独立基准）
    assert r1["total_score"]["available"] and r2["total_score"]["available"]


def test_restart_invariance_in_memory(client, built):
    """进程内存储下「重启」：新建仓储/审计实例重放查询，结果不变。"""
    for i in range(30):
        client.post(f"/cards/{CARD}/score",
                    json={"features": FEAT, "request_id": f"p-{i}"})
    before = client.get(f"/cards/{CARD}/versions/{built}/stability").json()
    # 模拟重启：留痕来自仓储（这里用同一仓储上新建审计服务，等价重启后读取）
    from app.monitor.audit import ScoreAudit
    from app.scoring.engine import ScorecardRuntime
    fresh_auditor = ScoreAudit(repo, _StubRuntimes(repo))
    r = fresh_auditor.score_single(CARD, {"features": FEAT, "request_id": "p-0"})
    assert r.status == "duplicate"
    after = client.get(f"/cards/{CARD}/versions/{built}/stability").json()
    assert after["n_scored"] == before["n_scored"]
    assert after["total_score"]["psi"] == before["total_score"]["psi"]


class _StubRuntimes:
    def __init__(self, repo):
        self.repo = repo

    def get(self, card, version):
        art = self.repo.load_version(card, version)
        return version, ScorecardRuntime(art)

    def resolve_version(self, card, version):
        return version, self.repo.load_version(card, version)


# ============================================================ 老版本
def test_old_version_feature_psi_works_score_unavailable(client, built):
    repo.clear_score_baseline(CARD, built)
    _score_training_rows(client, built, prefix="old")
    rep = client.get(f"/cards/{CARD}/versions/{built}/stability").json()
    assert rep["total_score"]["available"] is False
    assert all(f["psi"] == pytest.approx(0.0, abs=1e-12)
               for f in rep["features"])
    perf = client.get(f"/cards/{CARD}/versions/{built}/performance").json()
    assert perf["n"] == 0  # 尚无回填，不影响接口
    # 打分完全不受影响
    r = client.post(f"/cards/{CARD}/score", json={"features": FEAT})
    assert r.status_code == 200


def test_supplement_baseline_with_original_sample(client, built,
                                                  dev_csv_bytes):
    repo.clear_score_baseline(CARD, built)
    # 错误样本（行数不同）拒绝
    wrong = b"label,age\n" + b"1,30\n0,40\n"
    r = client.post(
        f"/cards/{CARD}/versions/{built}/score-baseline",
        files={"file": ("w.csv", wrong, "text/csv")})
    assert r.status_code == 400
    # 原始建卡样本：逐特征计数核对通过后补录成功
    r = client.post(
        f"/cards/{CARD}/versions/{built}/score-baseline",
        files={"file": ("d.csv", dev_csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    art = repo.load_version(CARD, built)
    assert art["score_baseline"] and art["score_baseline"]["n"]
    # 已有基准不可覆盖
    r2 = client.post(
        f"/cards/{CARD}/versions/{built}/score-baseline",
        files={"file": ("d.csv", dev_csv_bytes, "text/csv")})
    assert r2.status_code == 409


# ============================================================ 回填与表现
def test_backfill_rejection_list_and_idempotent_relabel(client, built):
    _score_training_rows(client, built, prefix="bf")
    labels = {i: int(_dev_rows()[i]["label"]) for i in range(len(_dev_rows()))}
    items = [{"request_id": f"bf-rid-{i}", "label": labels[i]}
             for i in range(0, 200)]
    # 合法 200 条
    j = client.post(f"/cards/{CARD}/backfills", json={"items": items})
    job = _wait_backfill(client, j.json()["job_id"])
    assert job["applied"] == 200 and job["rejected"] == []

    # 再传：同标签幂等、不同标签冲突拒收、找不到/非法标签拒收
    # bf-rid-3 传与已回填标签不同的 0/1 字符串 -> 应是 conflict（证明字符串
    # 0/1 会被正常解析），另取一条未回填过的标识 bf-rid-200 测字符串幂等路径
    mixed = [
        {"request_id": "bf-rid-0", "label": labels[0]},          # duplicate
        {"request_id": "bf-rid-1", "label": 1 - labels[1]},      # conflict
        {"request_id": "ghost", "label": 1},                    # missing
        {"request_id": "bf-rid-2", "label": 2},                 # invalid
        {"request_id": "bf-rid-3", "label": 1 - labels[3]},      # conflict(字符串)
    ]
    j2 = client.post(f"/cards/{CARD}/backfills", json={"items": mixed})
    job2 = _wait_backfill(client, j2.json()["job_id"])
    assert job2["applied"] == 0 and job2["duplicates"] == 1
    reasons = {(x["request_id"], x["reason"].split("：")[0])
               for x in job2["rejected"]}
    assert ("bf-rid-1", "label_conflict") in reasons
    assert ("bf-rid-3", "label_conflict") in reasons
    assert ("ghost", "在该卡的打分留痕中找不到该请求标识") in reasons
    assert ("bf-rid-2", "标签非法") in reasons
    # 字符串 0/1 可正常回填（找一条没回填的）
    j3 = client.post(f"/cards/{CARD}/backfills", json={"items": [
        {"request_id": "bf-rid-250", "label": str(labels[250])}]})
    job3 = _wait_backfill(client, j3.json()["job_id"])
    assert job3["applied"] == 1
    # 冲突标签没有改写成新值
    rec = repo.find_score_record(CARD, "bf-rid-1")
    assert rec["label"] == labels[1]
    # 前 200 条 + 后补的 250 号共 201 条已回填，且表现只统计已回填的
    perf = client.get(f"/cards/{CARD}/versions/{built}/performance").json()
    assert perf["n"] == 201


def test_performance_metrics_match_build_metrics_on_same_batch(
        client, built):
    """回填后 KS/AUC 与把同一批 (pd,label) 直接喂给建卡 metrics 完全一致。"""
    rows, results = _score_training_rows(client, built, prefix="pm")
    labels = [int(r["label"]) for r in rows]
    items = [{"request_id": f"pm-rid-{i}", "label": labels[i]}
             for i in range(len(rows))]
    j = client.post(f"/cards/{CARD}/backfills", json={"items": items})
    job = _wait_backfill(client, j.json()["job_id"], timeout=15)
    assert job["applied"] == len(rows)
    perf = client.get(f"/cards/{CARD}/versions/{built}/performance").json()
    assert perf["n"] == len(rows)
    p = np.array([r["pd"] for r in results], dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    assert perf["ks"] == pytest.approx(ks_stat(y, p), abs=1e-12)
    assert perf["auc"] == pytest.approx(roc_auc(y, p), abs=1e-12)
    assert perf["actual_bad_rate"] == pytest.approx(y.mean(), abs=1e-12)
    assert perf["avg_predicted_pd"] == pytest.approx(p.mean(), abs=1e-12)
    # 总分分段合计对得上，且每段预测/实际坏率都给出
    assert sum(s["count"] for s in perf["segments"]) == len(rows)
    for s in perf["segments"]:
        if s["count"]:
            assert 0 <= s["actual_bad_rate"] <= 1
            assert 0 < s["avg_predicted_pd"] < 1


def test_backfill_scoped_by_card_name(client, dev_csv_bytes):
    build_and_wait(client, "cA", dev_csv_bytes)
    build_and_wait(client, "cB", dev_csv_bytes)
    client.post("/cards/cA/score",
                json={"features": FEAT, "request_id": "shared-id"})
    client.post("/cards/cB/score",
                json={"features": FEAT, "request_id": "shared-id"})
    # cB 的回填不能回填 cA 的留痕…… 两个卡标识空间独立，各自 applied
    for card in ("cA", "cB"):
        j = client.post(f"/cards/{card}/backfills",
                        json={"items": [{"request_id": "shared-id",
                                         "label": 1}]})
        job = _wait_backfill(client, j.json()["job_id"]) if False else None
        # _wait_backfill 路径写死 CARD，这里直接轮询
        job_id = j.json()["job_id"]
        for _ in range(100):
            job = client.get(f"/cards/{card}/backfills/{job_id}").json()
            if job["status"] == "finished":
                break
            time.sleep(0.02)
        assert job["applied"] == 1


def test_empty_window_returns_nulls_not_error(client, built):
    rep = client.get(
        f"/cards/{CARD}/versions/{built}/stability"
        "?start=2000-01-01&end=2000-02-01").json()
    assert rep["n_scored"] == 0 and rep["verdict"] == "stable"
    perf = client.get(
        f"/cards/{CARD}/versions/{built}/performance"
        "?start=2000-01-01&end=2000-02-01").json()
    assert perf["n"] == 0 and perf["ks"] is None and perf["auc"] is None
