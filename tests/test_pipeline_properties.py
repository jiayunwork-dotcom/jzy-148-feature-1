"""端到端建卡产物性质（需求列出的性质逐条验证）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.binning import bin_feature
from app.core.pipeline import BuildParams, run_pipeline
from app.core.sample import parse_csv
from app.core.scoring import pd_from_score
from app.scoring.engine import ScorecardRuntime


@pytest.fixture(scope="module")
def dev_bytes():
    with open("data/dev_sample.csv", "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def artifacts(dev_bytes):
    s = parse_csv(dev_bytes)
    return run_pipeline(s, BuildParams())


@pytest.fixture(scope="module")
def sample(dev_bytes):
    return parse_csv(dev_bytes)


# 1) 平均预测违约概率 = 样本违约率（1e-6）
def test_avg_predicted_pd_equals_bad_rate(artifacts):
    v = artifacts["validation"]
    assert v["abs_error"] < 1e-6
    assert v["avg_predicted_pd"] == pytest.approx(v["sample_bad_rate"], abs=1e-9)


# 2) 标签取反：WOE 变号、IV 不变（端到端，含筛选/回归之外的分箱产物）
def test_label_inversion_e2e(dev_csv_bytes):
    s1 = parse_csv(dev_csv_bytes)
    s2 = parse_csv(dev_csv_bytes)
    s2.label = 1.0 - s2.label
    a1 = run_pipeline(s1, BuildParams(iv_threshold=0.05))
    a2 = run_pipeline(s2, BuildParams(iv_threshold=0.05))
    f1 = {f["name"]: f for f in a1["features"]}
    f2 = {f["name"]: f for f in a2["features"]}
    for name in a1["selected_features"]:
        b1 = {b["label"]: b for b in f1[name]["bins"]}
        b2 = {b["label"]: b for b in f2[name]["bins"]}
        assert set(b1) == set(b2)
        for lab in b1:
            assert b1[lab]["woe"] == pytest.approx(-b2[lab]["woe"], abs=1e-12)
        assert f1[name]["iv"] == pytest.approx(f2[name]["iv"], abs=1e-12)
    # 取反后校准性质同样成立
    assert a2["validation"]["abs_error"] < 1e-6


# 3) 样本复制：分箱、WOE、系数都不变
def test_duplicate_sample_e2e(dev_csv_bytes):
    s1 = parse_csv(dev_csv_bytes)
    lines = dev_csv_bytes.decode().strip().splitlines()
    doubled = (lines[0] + "\n" + "\n".join(lines[1:] + lines[1:]) + "\n").encode()
    s2 = parse_csv(doubled)
    a1 = run_pipeline(s1, BuildParams())
    a2 = run_pipeline(s2, BuildParams())
    assert a1["selected_features"] == a2["selected_features"]
    f1 = {f["name"]: f for f in a1["features"]}
    f2 = {f["name"]: f for f in a2["features"]}
    for name in a1["selected_features"]:
        assert len(f1[name]["bins"]) == len(f2[name]["bins"])
        for b1, b2 in zip(f1[name]["bins"], f2[name]["bins"]):
            assert b1["woe"] == pytest.approx(b2["woe"], abs=1e-12)
    assert np.allclose(a1["regression"]["beta"], a2["regression"]["beta"], atol=1e-10)


# 4) 各箱分值之和 = 总分
def test_total_score_is_sum_of_bin_scores(artifacts):
    rt = ScorecardRuntime(artifacts)
    applicant = {"age": 35, "income": 8000, "debt_ratio": 0.4,
                 "city": "SH", "job_grade": "B"}
    r = rt.score(applicant)
    assert r["total_score"] == pytest.approx(
        sum(f["score"] for f in r["features"]), abs=1e-9)


# 5) 总分 +PDO，odds 翻倍
def test_pdo_doubles_odds(artifacts):
    rt = ScorecardRuntime(artifacts)
    pdo = artifacts["params"]["pdo"]
    base_odds = artifacts["params"]["base_odds"]
    base_score = artifacts["params"]["base_score"]

    def odds(score):
        pd = pd_from_score(score, _tables(rt))
        return (1 - pd) / pd

    # 基准分处 odds = 基准 odds
    assert odds(base_score) == pytest.approx(base_odds, abs=1e-9)
    for s0 in (450, 600, 720):
        assert odds(s0 + pdo) / odds(s0) == pytest.approx(2.0, abs=1e-9)


# 6) PD 随总分严格单调下降且始终在 (0,1)
def test_pd_strictly_decreasing_within_unit(artifacts):
    rt = ScorecardRuntime(artifacts)
    scores = np.linspace(100, 1100, 2001)
    pds = np.array([pd_from_score(float(s), _tables(rt)) for s in scores])
    assert np.all(pds > 0) and np.all(pds < 1)
    assert np.all(np.diff(pds) < 0)


# 7) 数值特征合并后 WOE 序列确实单调
def test_numeric_woe_monotone_after_merge(artifacts):
    for f in artifacts["features"]:
        if f["type"] != "numeric" or not f["selected"]:
            continue
        ws = [b["woe"] for b in f["bins"]]
        d = f["direction"]
        assert d in (-1, 1)
        diffs = np.diff(ws)
        if d == -1:
            assert np.all(diffs <= 1e-12)
        else:
            assert np.all(diffs >= -1e-12)


# 额外：在线 PD 与回归模型 predict_proba 一致
def test_online_pd_matches_model(artifacts, sample):
    rt = ScorecardRuntime(artifacts)
    beta = np.array(artifacts["regression"]["beta"])
    name2f = {f["name"]: f for f in artifacts["features"]}
    sel = artifacts["selected_features"]

    def woe_of(name, val):
        fb = name2f[name]
        if val is None:
            return fb["missing_bin"]["woe"]
        if fb["type"] == "numeric":
            for b in fb["bins"]:
                if (b["lo"] <= val <= b["hi"]) if b["right_closed"] else (b["lo"] <= val < b["hi"]):
                    return b["woe"]
        else:
            for b in fb["bins"]:
                if val in b["categories"]:
                    return b["woe"]
        return 0.0

    max_diff = 0.0
    for i in range(sample.n):
        X = [1.0]
        row = {}
        for name in sample.feature_names:
            v = sample.features[name][i]
            if sample.types[name] == "numeric" and np.isnan(v):
                v = None
            row[name] = v
        for name in sel:
            X.append(woe_of(name, row[name]))
        model_pd = 1 / (1 + math.exp(-float(np.dot(beta, X))))
        online_pd = rt.score(row)["pd"]
        max_diff = max(max_diff, abs(model_pd - online_pd))
    assert max_diff < 1e-12


def _tables(rt):
    from app.core.scoring import ScoringTables
    return ScoringTables(
        rt.selected, np.array([]), rt.factor, rt.base_score,
        rt.base_odds, 0, 0, {}, {})
