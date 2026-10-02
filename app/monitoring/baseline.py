"""建卡时留存的监控基准。

特征层基准不需要额外存储：版本产物 features[].bins / missing_bin 里的好坏计数
就是建卡样本的逐箱计数（特征 PSI 用 total=bad+good），老版本同样具备，因此
**老版本的特征层稳定性照常可算**。

总分的联合分布无法从各特征边际计数推出，故新版本在建卡流水线中逐条打分、
留下总分的等频分箱基准（score baseline）。升级前已经存在的老版本产物里没有
这一块，查询时 score 基准返回 unavailable（原因写明），不影响特征层监控与打分。

等频切箱
--------
取 m 个等频分位的秩位置（floor(q·n)）映射到实际总分、去重后得到严格递增的
切点（打结不拆开），因此每箱非空；不同总分值数 <= m 时每个取值一箱。
训练总分一定落在 [首箱 lo, 末箱 hi] 内（基准自身越界数恒为 0）。
"""
from __future__ import annotations

import numpy as np

from ..core.config import settings
from ..scoring.engine import ScorecardRuntime


def _quantile_knots(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """等频切点（含首尾），吸附到实际取值，保证每箱非空。"""
    s = np.sort(scores, kind="mergesort")
    n = len(s)
    knots = [s[0]]
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for q in qs:
        idx = int(np.floor(float(q) * n))
        idx = min(max(idx, 1), n - 1)
        v = s[idx]
        if v > knots[-1]:
            knots.append(v)
    if s[-1] > knots[-1]:
        knots.append(s[-1])
    return np.asarray(knots, dtype=np.float64)


def build_score_baseline(scores: np.ndarray,
                         n_bins: int = settings.score_baseline_bins) -> dict:
    """把一批总分切成等频箱并计数，返回可 JSON 化的基准块。"""
    s = np.asarray(scores, dtype=np.float64)
    knots = _quantile_knots(s, n_bins)
    m = len(knots) - 1
    bins = []
    counts = [0] * m
    lo_first, hi_last = float(knots[0]), float(knots[-1])
    for i in range(m):
        lo, hi = float(knots[i]), float(knots[i + 1])
        right_closed = i == m - 1
        if lo == hi:
            label = f"{lo:g}"
        elif right_closed:
            label = f"[{lo:g}, {hi:g}]"
        else:
            label = f"[{lo:g}, {hi:g})"
        bins.append({"lo": lo, "hi": hi, "right_closed": right_closed,
                     "label": label})

    oor = 0
    for v in s:
        idx = assign_score_bin(bins, float(v))
        if idx is None:
            oor += 1
        else:
            counts[idx] += 1
    return {
        "bins": bins,
        "counts": counts,
        "n": int(len(s)),
        "lo": lo_first,
        "hi": hi_last,
        "out_of_range_count": oor,
    }


def assign_score_bin(bins: list[dict], score: float) -> int | None:
    """总分落箱索引；越出基准范围返回 None（调用方单独计入越界，不夹边箱）。"""
    if not bins:
        return None
    if score < bins[0]["lo"] or score > bins[-1]["hi"]:
        return None
    for i, b in enumerate(bins):
        if b["lo"] == b["hi"]:
            if score == b["lo"]:
                return i
        elif b["right_closed"]:
            if b["lo"] < score <= b["hi"]:
                return i
        else:
            if b["lo"] <= score < b["hi"]:
                return i
    return len(bins) - 1


def build_monitoring_baseline(sample, params, artifacts: dict) -> dict:
    """建卡流水线收尾时调用：用建好的卡对建卡样本逐条打分，留存总分基准。"""
    rt = ScorecardRuntime(artifacts)
    selected = artifacts["selected_features"]
    scores = np.empty(sample.n, dtype=np.float64)
    for i in range(sample.n):
        applicant = {name: _raw(sample, name, i) for name in selected}
        scores[i] = rt.score(applicant)["total_score"]
    score_baseline = build_score_baseline(scores)
    # 自检：训练样本逐条打分必全部落在基准范围内、计数和为 n
    assert sum(score_baseline["counts"]) == sample.n
    assert score_baseline["out_of_range_count"] == 0
    return {
        "version": 1,
        "policy": (
            "新版本建卡时对建卡样本逐条打分，留存总分等频基准；"
            "特征层基准直接取版本产物的逐箱好坏计数。"
            "打分越出总分基准范围的记录单独计入 out_of_range，不并入边箱。"
        ),
        "score_baseline": score_baseline,
    }


def _raw(sample, name: str, i: int):
    v = sample.features[name][i]
    if sample.types[name] == "numeric":
        return None if np.isnan(v) else float(v)
    return v
