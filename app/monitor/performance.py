"""表现回填后的区分能力复核。

硬约束：这里的 KS / AUC 必须与「把同一批分数与标签直接交给建卡时用的那套
评估」完全一致——因此本模块不另写任何排序/分组口径，直接 import 建卡用的
app.core.metrics.ks_stat / roc_auc（两者都以「违约概率/坏样本得分」为方向：
值越大越坏）。入参为留痕里的 pd（=建卡回归 predict_proba 的同一位结果）
与回填的 0/1 标签。

分段：优先用总分基准的等频箱（基准各段样本量均衡，便于投产前后对比）；
老版本没有基准时，按**本批实际总分**等频现切（仅用于本段对比，不产生
漂移结论）。低于/高于基准范围的分别单列 below/above 两段。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.metrics import ks_stat, roc_auc
from .baseline import classify_score, quantile_edges


def _seg(label: str, rows: list[dict]) -> dict:
    n = len(rows)
    bad = sum(1 for r in rows if r["label"] == 1)
    return {
        "segment": label,
        "count": n,
        "bad": bad,
        "good": n - bad,
        "actual_bad_rate": bad / n if n else None,
        "avg_predicted_pd": float(sum(r["pd"] for r in rows) / n) if n else None,
    }


def segment_report(
    rows: list[dict[str, Any]], score_baseline: dict | None
) -> list[dict]:
    """rows: 已回填标签、在查询区间内的留痕（含 total_score / pd / label）。"""
    if not rows:
        return []
    if score_baseline:
        edges = score_baseline["edges"]
        buckets: dict[str, list[dict]] = {
            "below": [], "above": [],
            **{str(i): [] for i in range(len(edges) - 1)},
        }
        for r in rows:
            buckets[classify_score(float(r["total_score"]), edges)].append(r)
        out = [_seg("低于训练范围", buckets["below"])] if buckets["below"] else []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            lab = f"[{lo:g}, {hi:g}{']' if i == len(edges) - 2 else ')'}"
            out.append(_seg(lab, buckets[str(i)]))
        if buckets["above"]:
            out.append(_seg("高于训练范围", buckets["above"]))
        return out

    # 无基准：按本批总分等频现切（仅分段对比用）
    scores = sorted(float(r["total_score"]) for r in rows)
    edges = quantile_edges(scores, n_bins=10)
    n_bin = max(len(edges) - 1, 1)
    if len(edges) < 2:  # 本批总分全部相同：一个零宽度箱
        v = scores[0]
        edges, n_bin = [v, v], 1
    buckets: dict[str, list[dict]] = {str(i): [] for i in range(n_bin)}
    for r in rows:
        b = classify_score(float(r["total_score"]), edges)
        if b in ("below", "above"):  # 首尾即本批 min/max，不会发生
            b = "0" if b == "below" else str(n_bin - 1)
        buckets[b].append(r)
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lab = f"[{lo:g}, {hi:g}{']' if i == len(edges) - 2 else ')'}"
        out.append(_seg(lab, buckets[str(i)]))
    return out


def performance_report(
    rows: list[dict[str, Any]], score_baseline: dict | None
) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "actual_bad_rate": None, "avg_predicted_pd": None,
                "ks": None, "auc": None, "segments": []}
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    p = np.array([r["pd"] for r in rows], dtype=np.float64)
    bad = int(y.sum())
    # 与建卡完全同一套函数；单类时 metrics 会抛错，这里如实报 None（无法复核）
    try:
        ks = ks_stat(y, p)
        auc = roc_auc(y, p)
    except ValueError:
        ks = auc = None
    return {
        "n": n,
        "bad": bad,
        "good": n - bad,
        "actual_bad_rate": bad / n,
        "avg_predicted_pd": float(p.mean()),
        "ks": ks,
        "auc": auc,
        "segments": segment_report(rows, score_baseline),
    }
