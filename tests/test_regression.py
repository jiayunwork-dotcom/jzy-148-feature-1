"""逻辑回归（自研牛顿）性质：收敛、平均预测=违约率、完全分离失败、复制不变。"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import BuildError
from app.core.regression import fit_logistic, predict_proba


def test_newton_matches_known_separation_problem():
    """两特征二维可分问题：拟合概率均值=正例率，且概率随 x1 降、x2 升。"""
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = -0.5 - 1.3 * x1 + 0.8 * x2
    p = 1 / (1 + np.exp(-eta))
    y = (rng.random(n) < p).astype(float)

    X = np.column_stack([np.ones(n), x1, x2])
    beta, n_iter, history = fit_logistic(X, y)
    assert n_iter < 100
    assert beta[1] == pytest.approx(-1.3, abs=0.15)
    assert beta[2] == pytest.approx(0.8, abs=0.15)

    p_hat = predict_proba(X, beta)
    assert abs(p_hat.mean() - y.mean()) < 1e-6
    # 迭代历史已留存
    assert history[-1]["iter"] == n_iter
    assert history[-1]["grad_max_abs"] is not None


def test_intercept_only_calibration():
    """只有截距时，预测概率恒等于样本违约率（解析解 beta0=ln(r/(1-r)))。"""
    y = np.array([1] * 300 + [0] * 700, dtype=float)
    X = np.ones((1000, 1))
    beta, _, _ = fit_logistic(X, y)
    r = 0.3
    assert beta[0] == pytest.approx(np.log(r / (1 - r)), abs=1e-6)
    assert abs(predict_proba(X, beta).mean() - r) < 1e-9


def test_complete_separation_fails_with_reason():
    """完全分离（y=1[x>0]，无截距以外重叠）牛顿迭代发散，必须失败并记录原因。"""
    x = np.linspace(-5, 5, 600)
    y = (x > 0).astype(float)
    X = np.column_stack([np.ones_like(x), x])
    with pytest.raises(BuildError) as ei:
        fit_logistic(X, y, max_iter=100)
    assert "不收敛" in str(ei.value) or "发散" in str(ei.value)
    assert getattr(ei.value, "history", None)


def test_duplicate_rows_coefficients_invariant():
    """设计矩阵原样复制一倍，收敛系数不变（梯度/海森只差标量倍数）。"""
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=(n, 2))
    eta = 0.4 + 0.9 * x[:, 0] - 0.6 * x[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    X = np.column_stack([np.ones(n), x])
    b1, _, _ = fit_logistic(X, y)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([y, y])
    b2, _, _ = fit_logistic(X2, y2)
    assert np.max(np.abs(b1 - b2)) < 1e-8
