"""总分分布基准：建卡时把训练样本逐条经**在线打分引擎**打分后留存。

为什么必须逐行在线打分而不是从分箱边际计数推：总分是各特征 WOE 的联合函数，
仅靠各特征各箱的边际计数推不出联合分布。因此：
- 新版本：建卡流水线在产物 artifacts["score_baseline"] 中直接留存总分基准
  （等频边界 + 各箱计数 + n），零信息损失，可重放校验；
- 升级前已存在的老版本：产物里没有这个基准，系统无法替它补（原始样本未必
  还在）。特征层 PSI 只依赖分箱计数，老版本照常能算；总分层返回
  unavailable。若老卡的原始建卡样本还找得到，可用管理接口
  POST .../score-baseline 用样本补基准（校验逐特征分箱计数与产物完全一致，
  防止用错样本）。

分箱方式与建卡数值分箱同源：按训练总分取等频分位点、打结不拆开，切出
2..n_bins 个非空箱，边界首尾为训练总分的最小/最大值，最后一箱右端封闭；
落箱规则也与在线引擎一致（小于最小边界 = 低于训练范围，单独成桶）。
这样把训练样本原样逐条再打分时，总分 PSI 严格为 0、各箱占比逐位吻合。
"""
from __future__ import annotations

from typing import Any

from ..scoring.engine import ScorecardRuntime


def quantile_edges(scores: list[float], n_bins: int = 10) -> list[float]:
    """等频边界：首尾 min/max + 内部 (n_bins-1) 个分位点，吸附到实际取值且
    打结不拆开；相邻边界去重，因此返回长度可能小于 n_bins+1（非空箱数随之减少）。
    """
    if not scores:
        return []
    ordered = sorted(scores)
    n = len(ordered)
    edges = [ordered[0]]
    for k in range(1, n_bins):
        target = k * n / n_bins
        # 最小下标使 ordered[idx] >= target 分位（等价于分位点吸附）
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if mid < target:
                lo = mid + 1
            else:
                hi = mid
        edge = ordered[lo]
        if edge > edges[-1]:
            edges.append(edge)
    if ordered[-1] > edges[-1]:
        edges.append(ordered[-1])
    return edges


def classify_score(score: float, edges: list[float]) -> str:
    """总分落桶：箱下标（"0".."m-1"，最后一箱右端封闭），或 'below'/'above'。

    与在线数值落箱同规则：score <= edges[0] 入首箱；其余 [lo, hi)；
    最后一箱 [lo, hi] 闭；严格低于 edges[0] 为 below、严格高于 edges[-1] 为 above。
    """
    if score < edges[0]:
        return "below"
    if score > edges[-1]:
        return "above"
    if score <= edges[0]:
        return "0"
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            hit = lo <= score <= hi
        else:
            hit = lo <= score < hi
        if hit:
            return str(i)
    return str(len(edges) - 2)


def score_training_samples(
    artifacts: dict, rows: list[dict[str, Any]]
) -> list[float]:
    """用与线上完全相同的引擎给训练样本逐条打分，返回总分列表（行序保持）。"""
    rt = ScorecardRuntime(artifacts)
    return [float(rt.score(row)["total_score"]) for row in rows]


def build_score_baseline(
    artifacts: dict, rows: list[dict[str, Any]], n_bins: int = 10
) -> dict:
    """生成总分基准：edges + 各箱计数 + n。"""
    scores = score_training_samples(artifacts, rows)
    edges = quantile_edges(scores, n_bins=n_bins)
    counts = [0] * max(len(edges) - 1, 0)
    for s in scores:
        b = classify_score(s, edges)
        counts[int(b)] += 1
    return {"edges": edges, "counts": counts, "n": len(scores)}


def feature_baseline_counts(artifacts: dict) -> dict[str, dict]:
    """从建卡产物摘出入模特征的基准分布（常规箱 + 缺失箱计数）。"""
    out: dict[str, dict] = {}
    for fb in artifacts["features"]:
        if not fb["selected"]:
            continue
        out[fb["name"]] = {
            "bin_labels": [b["label"] for b in fb["bins"]],
            "counts": [int(b["total"]) for b in fb["bins"]],
            "missing_label": fb["missing_bin"]["label"],
            "missing_count": int(fb["missing_bin"]["total"]),
            "n": int(sum(b["total"] for b in fb["bins"])
                     + fb["missing_bin"]["total"]),
        }
    return out


def verify_feature_counts(
    artifacts: dict, rows: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """用给定样本逐行在线落箱，核对每个入模特征每箱（含缺失箱）计数是否与
    建卡产物完全一致——用于老版本补总分基准时防止用错样本。

    返回 (是否一致, 不一致说明清单)。
    """
    rt = ScorecardRuntime(artifacts)
    expected = feature_baseline_counts(artifacts)
    actual: dict[str, dict[str, int]] = {
        name: {"missing": 0, **{lab: 0 for lab in e["bin_labels"]}}
        for name, e in expected.items()
    }
    for row in rows:
        scored = rt.score(row)
        for fs in scored["features"]:
            if fs["name"] not in actual:
                continue
            if fs["status"] in ("unseen", "out_of_range"):
                # 未见/越界在训练重放里不应出现；出现即样本不对
                actual[fs["name"]][fs["status"]] = (
                    actual[fs["name"]].get(fs["status"], 0) + 1)
                continue
            lab = fs["bin_label"]
            if lab == expected[fs["name"]]["missing_label"]:
                actual[fs["name"]]["missing"] += 1
            else:
                actual[fs["name"]][lab] += 1

    diffs: list[str] = []
    for name, e in expected.items():
        for lab, cnt in zip(e["bin_labels"], e["counts"]):
            got = actual[name].get(lab, 0)
            if got != cnt:
                diffs.append(f"{name} 箱 {lab!r}：样本计数 {got} != 建卡计数 {cnt}")
        if actual[name].get("missing", 0) != e["missing_count"]:
            diffs.append(
                f"{name} 缺失箱：样本计数 {actual[name].get('missing', 0)} "
                f"!= 建卡计数 {e['missing_count']}")
        for odd in ("unseen", "out_of_range"):
            if actual[name].get(odd, 0):
                diffs.append(
                    f"{name}：建卡样本重放出现 {odd} {actual[name][odd]} 条")
    return (not diffs, diffs)
