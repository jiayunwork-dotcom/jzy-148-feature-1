"""把原始特征值映射为 WOE，并构造逻辑回归设计矩阵。

训练阶段所有类别取值都属于某一合并后的类别箱；打分阶段未见类别由 scoring
模块单独处理（不静默归入某一箱）。
"""
from __future__ import annotations

import numpy as np

from .binning import FeatureBinning, assign_bin


def values_to_woe(fb: FeatureBinning, values: np.ndarray,
                  unseen_woe: float | None = None) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float64)
    if fb.ftype == "numeric":
        miss = np.isnan(values)
    else:
        miss = np.array([x is None for x in values], dtype=bool)
    for i, val in enumerate(values):
        if miss[i]:
            out[i] = fb.missing_bin.woe
            continue
        try:
            out[i] = assign_bin(fb, val).woe
        except KeyError:
            if unseen_woe is None:
                raise
            out[i] = unseen_woe
    return out


def build_design_matrix(
    sample,
    binnings: dict[str, FeatureBinning],
    feature_names: list[str],
) -> np.ndarray:
    n = len(sample.label)
    X = np.ones((n, len(feature_names) + 1), dtype=np.float64)
    for j, name in enumerate(feature_names, start=1):
        X[:, j] = values_to_woe(binnings[name], sample.features[name])
    return X
