"""人群稳定性（PSI）测试。

覆盖需求中的硬性质：
- 建卡样本原样逐条打分后，特征与总分 PSI 全为 0（浮点误差内），
  逐箱占比与建卡产物完全吻合；
- 未见类别 / 越界单独报占比，不并入箱；
- 总分基准只存在于新版本：老版本总分部分 unavailable、特征层照常、打分照常；
- 版本之间互不渗入；
- 时间区间过滤、三档阈值、与从记录逐条重算一致；
- 8 线程并发 2000 次打分，统计次数正好 2000。
"""
from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.deps import audit, repo, runtimes
from app.monitoring.psi import PSI_FLOOR, psi_index
from tests.conftest import build_and_wait
from tests.helpers import load_dev_rows, row_to_features, score_training_sample

CARD = "card_psi"


def _stability(client, card=CARD, version=1, **params):
    return client.get(
        f"/cards/{card}/versions/{version}/stability", params=params).json()


def _artifact(client, card=CARD, version=1):
    return client.get(f"/cards/{card}/versions/{version}/artifact").json()


# ------------------------------------------------------------ PSI 口径

def test_psi_zero_and_symmetric():
    assert psi_index([0.25] * 4, [0.25] * 4) == pytest.approx(0.0, abs=1e-12)
    a = [0.1, 0.4, 0.5]
    b = [0.3, 0.3, 0.4]
    # 交换两侧 PSI 不变
    assert psi_index(a, b) == pytest.approx(psi_index(b, a))
    # 一侧某箱为 0：不爆炸，且另一侧归零也有限
    v = psi_index([1.0, 0.0], [0.5, 0.5])
    assert math.isfinite(v) and v > 0
    # 地板口径：两侧都为 0 的箱贡献 0
    assert psi_index([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-12)


def test_psi_zero_side_uses_floor():
    # expected 箱占比 0、actual 占比 1 时：(1-FLOOR)·ln(1/FLOOR) 量级有限
    v = psi_index([1.0], [0.0])
    expected = (1 - PSI_FLOOR) * math.log(1 / PSI_FLOOR)
    assert v == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------ 自比为零

def test_training_sample_self_comparison_psi_zero(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    rows = score_training_sample(client, CARD)
    art = _artifact(client)
    st = _stability(client)
    assert st["n_scored"] == len(rows) == art["data_summary"]["n"]

    # 特征：逐箱计数与建卡产物完全一致，PSI 为 0
    for f in st["features"]:
        fb = next(x for x in art["features"] if x["name"] == f["name"])
        exp = [b["bad"] + b["good"] for b in fb["bins"]]
        exp.append(fb["missing_bin"]["bad"] + fb["missing_bin"]["good"])
        assert [b["actual_count"] for b in f["bins"]] == exp
        assert [b["expected_count"] for b in f["bins"]] == exp
        for b in f["bins"]:
            assert b["actual_pct"] == pytest.approx(b["expected_pct"], abs=1e-12)
        assert f["psi"] == pytest.approx(0.0, abs=1e-12)
        assert f["rating"] == "stable"
        assert f["unseen_count"] == 0 and f["out_of_range_count"] == 0

    # 总分：与新版本建卡时留存的基准一致
    sc = st["total_score"]
    assert sc["status"] == "ok"
    assert sc["psi"] == pytest.approx(0.0, abs=1e-12)
    assert sc["out_of_range_count"] == 0
    base_counts = art["monitoring"]["score_baseline"]["counts"]
    assert [b["actual_count"] for b in sc["bins"]] == base_counts
    assert st["overall_rating"] == "stable"


# ------------------------------------------------------------ 漂移、未见、越界

def test_drift_detected_and_rated(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    # 极端人群：全是年轻、E 级、SZ（训练里可能未见）
    drift_feats = [
        {"age": 20 + i % 3, "income": 3000 + i, "city": "SZ", "job_grade": "E"}
        for i in range(300)
    ]
    for f in drift_feats:
        client.post(f"/cards/{CARD}/score", json={"features": f})
    st = _stability(client)
    assert st["n_scored"] == 300
    by_name = {f["name"]: f for f in st["features"]}
    assert by_name["job_grade"]["psi"] > 0.25
    assert by_name["job_grade"]["rating"] == "drift"
    assert st["overall_rating"] == "drift"
    # city=SZ 是否未见取决于样本；未见占比必须独立报出且不进 PSI 分母
    city = by_name["city"]
    binned = sum(b["actual_count"] for b in city["bins"])
    assert binned + city["unseen_count"] == 300
    if city["unseen_count"]:
        # PSI 的 actual 分母只含落箱记录；全部未见时逐箱占比和为 0
        binned_sum = sum(b["actual_pct"] for b in city["bins"])
        if binned:
            assert binned_sum == pytest.approx(1.0, abs=1e-9)
        else:
            assert city["psi"] is None


def test_unseen_and_out_of_range_reported_separately(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    feats = [
        {"age": -500, "income": 9000, "city": "MARS", "job_grade": "A"},
        {"age": 99999, "income": 9000, "city": "MARS", "job_grade": "A"},
        {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"},
    ]
    for f in feats:
        client.post(f"/cards/{CARD}/score", json={"features": f})
    st = _stability(client)
    by_name = {f["name"]: f for f in st["features"]}
    assert by_name["city"]["unseen_pct"] == pytest.approx(2 / 3)
    assert by_name["age"]["out_of_range_pct"] == pytest.approx(2 / 3)
    # 越界仍夹到首/末箱算占比；未见不落任何箱
    binned_age = sum(b["actual_count"] for b in by_name["age"]["bins"])
    assert binned_age == 3
    binned_city = sum(b["actual_count"] for b in by_name["city"]["bins"])
    assert binned_city == 1


def test_warning_rating_band(client, dev_csv_bytes):
    # 阈值可配置：psi_rating 的阈值来自参数（settings 默认 0.10/0.25）
    from app.monitoring.psi import DRIFT, STABLE, WARNING, psi_rating
    assert psi_rating(0.05) == STABLE
    assert psi_rating(0.10) == WARNING
    assert psi_rating(0.24) == WARNING
    assert psi_rating(0.25) == DRIFT
    assert psi_rating(0.05, stable_max=0.04, warning_max=0.20) == WARNING

    build_and_wait(client, CARD, dev_csv_bytes)
    # 明显偏离建卡人群的进件 -> 总体进入 warning 或 drift
    for i in range(200):
        client.post(f"/cards/{CARD}/score", json={
            "features": {"age": 22, "income": 3100, "city": "SZ",
                         "job_grade": "E"}})
    st = _stability(client)
    assert st["overall_rating"] in {"warning", "drift"}


# ------------------------------------------------------------ 时间区间 / 版本隔离

def test_time_window_filtering(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    now = datetime.now(timezone.utc)
    client.post(f"/cards/{CARD}/score", json={"features": {
        "age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}})
    # [start, end) 语义：明天开始的窗口为空，空窗口不报错
    future = (now + timedelta(days=1)).isoformat()
    far = (now + timedelta(days=2)).isoformat()
    st = _stability(client, start=future, end=far)
    assert st["n_scored"] == 0
    for f in st["features"]:
        assert f["psi"] is None and f["rating"] is None
    # 覆盖当前的窗口能查到
    past = (now - timedelta(minutes=1)).isoformat()
    st2 = _stability(client, start=past, end=future)
    assert st2["n_scored"] == 1
    bad = client.get(f"/cards/{CARD}/versions/1/stability",
                     params={"start": far, "end": past})
    assert bad.status_code == 400


def test_versions_do_not_mix(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes,
                   base_score=600, base_odds=20, pdo=50)
    build_and_wait(client, CARD, dev_csv_bytes,
                   base_score=650, base_odds=10, pdo=25)
    feat = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}
    for _ in range(3):
        client.post(f"/cards/{CARD}/score",
                    json={"version": 1, "features": feat})
    for _ in range(5):
        client.post(f"/cards/{CARD}/score",
                    json={"version": 2, "features": feat})
    assert _stability(client, version=1)["n_scored"] == 3
    assert _stability(client, version=2)["n_scored"] == 5


# ------------------------------------------------------------ 老版本降级

def test_legacy_version_score_unavailable_but_features_ok(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    feat = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}
    for _ in range(20):
        client.post(f"/cards/{CARD}/score",
                    json={"version": 1, "features": feat})
    # 模拟升级前产物：直接抹掉版本 1 的 monitoring 块
    repo._versions[CARD][1]["artifacts"].pop("monitoring", None)
    st = _stability(client)
    assert st["total_score"]["status"] == "unavailable"
    assert "边际" in st["total_score"]["reason"] or "基准" in st["total_score"]["reason"]
    assert all(f["psi"] is not None for f in st["features"])
    # 打分照常
    r = client.post(f"/cards/{CARD}/score",
                    json={"version": 1, "features": feat})
    assert r.status_code == 200


# ------------------------------------------------------------ 并发计数

def test_concurrent_2000_scores_exact_count(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    ver, rt = runtimes.get(CARD, 1)
    pool = [{"age": 20 + i % 40, "income": 3000 + (i % 50) * 100,
             "city": ["BJ", "SH", "GZ", "CD", "WH", "XA", "SZ"][i % 7],
             "job_grade": ["A", "B", "C", "D", "E"][i % 5]}
            for i in range(250)]
    errors: list[Exception] = []

    def worker(t):
        for k in range(250):
            i = (t * 250 + k) % 250
            try:
                audit.score_single(CARD, ver, rt, pool[i], None)
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    logs = repo.query_score_logs(CARD, 1, None, None, 10_000_000)
    assert len(logs) == 2000
    # 与监控服务统计一致
    st = _stability(client)
    assert st["n_scored"] == 2000
    counted = 0
    for f in st["features"]:
        counted = max(counted, sum(b["actual_count"] for b in f["bins"])
                      + f["unseen_count"])
    assert counted == 2000


def test_recompute_from_logs_matches_service(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    # 混合：无标识 + 有标识打分
    for i in range(50):
        client.post(f"/cards/{CARD}/score", json={"features": {
            "age": 20 + i % 35, "income": 3000 + i * 7,
            "city": ["BJ", "SH", "GZ"][i % 3],
            "job_grade": ["A", "C", "E"][i % 3]}})
        client.post(f"/cards/{CARD}/score", json={"features": {
            "age": 45 + i % 10, "income": 8000 + i, "city": "CD",
            "job_grade": "B"}, "request_id": f"M{i}"})
    st = _stability(client)
    logs = repo.query_score_logs(CARD, 1, None, None, 10_000_000)
    assert st["n_scored"] == len(logs) == 100
    # 从记录逐条重算 age 各状态计数，与服务结果一致
    by_name = {f["name"]: f for f in st["features"]}
    age = by_name["age"]
    manual_bins = {b["bin_label"]: 0 for b in age["bins"]}
    manual_unseen = manual_oor = 0
    for log in logs:
        entry = next(f for f in log["feature_bins"] if f["name"] == "age")
        if entry["status"] == "unseen":
            manual_unseen += 1
        else:
            manual_bins[entry["bin_label"]] += 1
            if entry["status"] == "out_of_range":
                manual_oor += 1
    for b in age["bins"]:
        assert b["actual_count"] == manual_bins[b["bin_label"]]
    assert age["unseen_count"] == manual_unseen
    assert age["out_of_range_count"] == manual_oor
