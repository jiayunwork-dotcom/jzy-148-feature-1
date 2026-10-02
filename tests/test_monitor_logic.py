"""PSI 与稳定性/表现纯函数的单测（不依赖 HTTP 与存储）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.monitor.stability import (
    VERDICT_SIGNIFICANT, VERDICT_STABLE, VERDICT_WARNING,
    psi_index, verdict_for, stability_report,
)
from app.monitor.baseline import build_score_baseline, classify_score, quantile_edges
from app.monitor.performance import performance_report
from app.core.metrics import ks_stat, roc_auc


# ------------------------------------------------------------- PSI 公式
def test_psi_identical_distributions_is_zero():
    counts = [100, 200, 300, 50]
    assert psi_index(counts, counts) == pytest.approx(0.0, abs=1e-15)


def test_psi_symmetric_scale_invariant_and_nonneg():
    base = [500, 300, 200]
    shifted = [200, 400, 400]
    p1 = psi_index(base, shifted)
    p2 = psi_index(shifted, base)
    assert p1 == pytest.approx(p2, abs=1e-12)     # 交换两侧结果一致
    assert p1 > 0.1


def test_psi_empty_side_returns_none():
    assert psi_index([], [1, 2]) is None
    assert psi_index([1, 2], [0, 0]) is None


def test_psi_zero_side_smoothed_finite_and_monotone():
    """新箱只在实际侧出现：0.5 平滑后有限；占比越大值越大。"""
    small = psi_index([500, 0], [499, 1])
    big = psi_index([500, 0], [100, 400])
    assert math.isfinite(small) and math.isfinite(big)
    assert 0 < small < big
    # 两侧都为 0 的箱不贡献
    assert psi_index([500, 0, 0], [500, 0, 0]) == pytest.approx(0.0, abs=1e-15)
    # 与样本量同阶：同样 1% 的罕见新箱，小样本下 0.5 平滑会压住它，
    # 不会把 1/100 的新箱夸大成比 100/1000 更严重
    p_small = psi_index([99, 0], [99 - 1, 1])
    p_large = psi_index([999, 0], [999 - 10, 10])
    assert p_small < p_large


def test_psi_known_handcalc():
    # 两箱：基准 50/50，实际 60/40（用计数 100/100 vs 120/80）
    pb, pa = 0.5, 0.6
    qb, qa = 0.5, 0.4
    expected = (pa - pb) * math.log(pa / pb) + (qa - qb) * math.log(qa / qb)
    assert psi_index([100, 100], [120, 80]) == pytest.approx(expected, abs=1e-15)


def test_verdict_thresholds():
    assert verdict_for(0.0, 0.1, 0.25) == VERDICT_STABLE
    assert verdict_for(0.0999, 0.1, 0.25) == VERDICT_STABLE
    assert verdict_for(0.1, 0.1, 0.25) == VERDICT_WARNING
    assert verdict_for(0.2499, 0.1, 0.25) == VERDICT_WARNING
    assert verdict_for(0.25, 0.1, 0.25) == VERDICT_SIGNIFICANT


# ------------------------------------------------------------- 总分基准分箱
def test_quantile_edges_and_classifier():
    vals = [float(i) for i in range(100)]
    edges = quantile_edges(vals, n_bins=10)
    assert edges[0] == 0.0 and edges[-1] == 99.0
    assert len(edges) == 11
    for v in vals:
        b = classify_score(v, edges)
        assert b not in ("below", "above")
        assert 0 <= int(b) <= 9
    assert classify_score(-1.0, edges) == "below"
    assert classify_score(99.0001, edges) == "above"
    # 末箱右端封闭
    assert classify_score(99.0, edges) == "9"
    # 首值入首箱（== edges[0] 不算越界）
    assert classify_score(0.0, edges) == "0"


def test_quantile_edges_ties_not_split():
    # 只有 3 个不同取值且大量打结：非空箱数 <= 3
    vals = [1.0] * 40 + [2.0] * 40 + [3.0] * 40
    edges = quantile_edges(vals, n_bins=10)
    assert edges == [1.0, 2.0, 3.0]


def test_build_score_baseline_counts_partition_all_rows():
    artifacts, rows = _tiny_artifacts_and_rows()
    base = build_score_baseline(artifacts, rows, n_bins=5)
    assert sum(base["counts"]) == len(rows)
    assert base["n"] == len(rows)
    # 训练样本分数全部落进基准箱，无越界
    from app.monitor.baseline import score_training_samples
    for s in score_training_samples(artifacts, rows):
        assert classify_score(s, base["edges"]) not in ("below", "above")


def _tiny_artifacts_and_rows():
    """用真实建卡流水线在开发样本上产产物 + 行样本。"""
    from app.core.pipeline import BuildParams, run_pipeline, _sample_rows_as_applicants
    from app.core.sample import parse_csv
    sample = parse_csv(open("data/dev_sample.csv", "rb").read())
    art = run_pipeline(sample, BuildParams())
    return art, _sample_rows_as_applicants(sample)


# ----------------------------------------------------- 稳定性报告组装
def test_stability_report_training_replay_all_zero():
    from app.monitor.baseline import score_training_samples
    art, rows = _tiny_artifacts_and_rows()
    base = art["score_baseline"]
    # 构造「留痕」：训练行经在线引擎打分的结果
    rt_rows = [
        {"total_score": s, "pd": 0.0, "features": _features_for(art, row)}
        for row, s in zip(rows, score_training_samples(art, rows))
    ]
    rep = stability_report(art, rt_rows)
    assert rep["n_scored"] == len(rows)
    assert rep["verdict"] == VERDICT_STABLE
    for f in rep["features"]:
        assert f["psi"] == pytest.approx(0.0, abs=1e-12)
        assert f["unseen_count"] == 0 and f["out_of_range_count"] == 0
        # 占比之和为 1（含缺失箱）
        assert sum(b["actual_share"] for b in f["bins"]) == pytest.approx(1.0)
        # 实际占比与基准占比逐位吻合
        for b in f["bins"]:
            assert b["actual_share"] == pytest.approx(b["baseline_share"], abs=1e-12)
            assert b["actual_count"] == b["baseline_count"]
    sc = rep["total_score"]
    assert sc["available"] and sc["psi"] == pytest.approx(0.0, abs=1e-12)
    for b in sc["bins"]:
        assert b["actual_share"] == pytest.approx(b["baseline_share"], abs=1e-12)
        assert b["actual_count"] == b["baseline_count"]


def test_stability_report_missing_baseline_total_score_unavailable():
    art, rows = _tiny_artifacts_and_rows()
    art = {**art, "score_baseline": None}
    rep = stability_report(art, [])
    assert rep["total_score"]["available"] is False
    assert rep["features"], "特征层稳定性必须照常可算"


def test_unseen_and_out_of_range_reported_separately():
    art, rows = _tiny_artifacts_and_rows()
    recs = []
    # 未见类别
    recs.append({"total_score": 600.0, "pd": 0.01,
                 "features": _features_for(art, {"age": 40, "income": 9000,
                                                 "city": "MARS",
                                                 "job_grade": "A"})})
    # 数值越界（age 打一个极端值）
    recs.append({"total_score": 500.0, "pd": 0.02,
                 "features": _features_for(art, {"age": 99999, "income": 9000,
                                                 "city": "BJ",
                                                 "job_grade": "A"})})
    rep = stability_report(art, recs)
    by = {f["feature"]: f for f in rep["features"]}
    assert by["city"]["unseen_count"] == 1 and by["city"]["unseen_rate"] == 0.5
    assert by["age"]["out_of_range_count"] == 1
    assert by["age"]["out_of_range_rate"] == 0.5
    # 这两类不并进任何箱：city 常规箱只计入另一条 BJ，未见的 MARS 不进箱
    city_binned = sum(b["actual_count"] for b in by["city"]["bins"])
    age_binned = sum(b["actual_count"] for b in by["age"]["bins"])
    assert city_binned == 1 and age_binned == 1


def _features_for(art, row):
    from app.scoring.engine import ScorecardRuntime
    return ScorecardRuntime(art).score(row)["features"]


# ----------------------------------------------------- 表现分析口径
def test_performance_reuses_build_metrics_exactly():
    """KS/AUC 与把同一批 (y,p) 直接交给建卡 metrics 完全一致。"""
    rng = np.random.default_rng(7)
    n = 800
    y = (rng.random(n) > 0.8).astype(int)
    p = np.clip(0.05 + 0.8 * y + rng.normal(0, 0.1, n), 1e-6, 1 - 1e-6)
    rows = [{"total_score": 700 - 400 * float(p[i]), "pd": float(p[i]),
             "label": int(y[i])} for i in range(n)]
    rep = performance_report(rows, score_baseline=None)
    assert rep["ks"] == pytest.approx(ks_stat(y.astype(float), p), abs=1e-15)
    assert rep["auc"] == pytest.approx(roc_auc(y.astype(float), p), abs=1e-15)
    assert rep["actual_bad_rate"] == pytest.approx(y.mean(), abs=1e-15)
    assert rep["avg_predicted_pd"] == pytest.approx(p.mean(), abs=1e-15)
    # 分段合计等于总体
    assert sum(s["count"] for s in rep["segments"]) == n
    assert sum(s["bad"] for s in rep["segments"]) == int(y.sum())


def test_performance_empty_and_single_class():
    rep = performance_report([], None)
    assert rep["n"] == 0 and rep["ks"] is None and rep["segments"] == []
    rows = [{"total_score": 600.0, "pd": 0.1, "label": 0} for _ in range(10)]
    rep = performance_report(rows, None)
    assert rep["ks"] is None and rep["auc"] is None   # 单类无法算，如实报 None
    assert rep["actual_bad_rate"] == 0.0
