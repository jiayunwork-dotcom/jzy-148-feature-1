"""指标自实现的核对：AUC 与一个手算小例对比，KS 用阶梯 CDF 性质。"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.metrics import ks_stat, roc_auc


def test_auc_handcalc():
    # 坏样本分数 0.9/0.8，好样本 0.7/0.6/0.1
    y = np.array([1, 1, 0, 0, 0], dtype=float)
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    # 6 个好坏对全部排对 -> AUC=1
    assert roc_auc(y, p) == pytest.approx(1.0)

    # 一对并列：坏 0.5，好 0.5/0.2 -> 两对中一对平(0.5)一对赢(0.2) => 0.75
    y2 = np.array([1, 0, 0], dtype=float)
    p2 = np.array([0.5, 0.5, 0.2])
    assert roc_auc(y2, p2) == pytest.approx(0.75)


def test_auc_random_order_half():
    rng = np.random.default_rng(2)
    # 分数与标签完全无关：AUC 期望 0.5
    y = np.array([0] * 1000 + [1] * 1000, dtype=float)
    p = rng.permutation(np.linspace(0, 1, 2000))
    assert abs(roc_auc(y, p) - 0.5) < 0.05


def test_ks_bounds_and_perfect():
    rng = np.random.default_rng(3)
    y = (rng.random(2000) > 0.7).astype(float)
    p = y * 0.8 + rng.random(2000) * 0.1
    ks = ks_stat(y, p)
    assert 0.0 <= ks <= 1.0
    assert ks > 0.5

    # 完全分离 -> KS=1
    p_perfect = y.astype(float)
    assert ks_stat(y, p_perfect) == pytest.approx(1.0)


def test_ks_monotone_piecewise():
    # 构造两段分明分布：坏集中低分、好集中高分，KS 在切点处达到 0.8
    y = np.array([1] * 40 + [1] * 10 + [0] * 10 + [0] * 40, dtype=float)
    # 坏: 40 个 0.1 + 10 个 0.5；好: 10 个 0.5 + 40 个 0.9
    p = np.array([0.1] * 40 + [0.5] * 10 + [0.5] * 10 + [0.9] * 40)
    ks = ks_stat(y, p)
    # 阈值取 0.1 后：累计坏 40/50=0.8，累计好 0 -> 0.8
    assert ks == pytest.approx(0.8)
