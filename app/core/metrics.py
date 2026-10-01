"""模型评估：AUC、KS 全部自实现，只用 numpy 排序。"""
from __future__ import annotations

import numpy as np


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC：随机取一个坏样本、一个好样本，坏样本分数更高的概率（分数相同算 0.5）。

    用平均秩公式：AUC = (坏样本秩和 - n1(n1+1)/2) / (n0*n1)。
    """
    y = y.astype(np.int64)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        raise ValueError("标签全为同一类，无法计算 AUC")

    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sp = p[order]
    # 打结赋予平均秩（秩从 1 开始）
    starts = np.flatnonzero(np.r_[True, sp[1:] != sp[:-1]])
    ends = np.r_[starts[1:], len(sp)]
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + 1 + e) / 2.0

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    """KS = 各阈值下坏样本与好样本累计分布之差的绝对值最大值。

    分数相同的样本视为同一阈值，一并计入累计后再比较。
    """
    y = y.astype(np.int64)
    B = int(y.sum())
    G = int(len(y) - B)
    if B == 0 or G == 0:
        raise ValueError("标签全为同一类，无法计算 KS")

    order = np.argsort(p, kind="mergesort")
    sp, sy = p[order], y[order]
    cum_b = cum_g = 0
    best = 0.0
    i = 0
    n = len(sp)
    while i < n:
        j = i
        while j < n and sp[j] == sp[i]:
            j += 1
        chunk = sy[i:j]
        cum_b += int(chunk.sum())
        cum_g += int((chunk == 0).sum())
        d = abs(cum_b / B - cum_g / G)
        if d > best:
            best = d
        i = j
    return float(best)
