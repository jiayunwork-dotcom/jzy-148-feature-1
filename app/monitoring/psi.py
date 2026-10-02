"""群体稳定性指标 PSI（Population Stability Index），手写实现。

    PSI = Σ over bins (p_actual - p_expected) * ln(p_actual / p_expected)

口径约定（监管检查时按此说明）：
- expected 为建卡样本的基准占比，actual 为查询区间内实际进件占比。
- 交换两侧不改变指标：每个分项 (a-e)·ln(a/e) 关于交换对称
  （交换后 (e-a)·ln(e/a) = (a-e)·ln(a/e)），故 PSI 无方向性。
- 零占比箱处理：PSI 在某一侧占比为 0 时数学上发散，本实现采用评分卡行业
  常见的「最小占比地板」——两侧占比都先做
      p' = max(p, FLOOR)，FLOOR = 1e-4（万分之一）
  再代入公式。含义：占比低于万分之一的箱视为万分之一，既避免 ln(0) 爆炸，
  又对该箱给出有限但显著的惩罚（单侧完全消失的一箱贡献约 0.0088，
  若干箱同时消失 PSI 即显著上升）。该地板只用于 PSI 求和，不改动返回的
  原始占比；占比为 0 的箱在对比表里仍如实显示 0。
- 未见类别（unseen）、越界（out_of_range）不并进任何箱：不参与 PSI，
  其占比在响应中单独报出。PSI 的分母只是「落入正常箱 + 缺失箱」的记录数。

三档阈值（可在 app/core/config.py 配置）：
  stable   : PSI <  0.10
  warning  : 0.10 <= PSI < 0.25
  drift    : PSI >= 0.25
"""
from __future__ import annotations

import math

from ..core.config import settings

#: 零占比地板：见模块 docstring
PSI_FLOOR = 1e-4

STABLE = "stable"
WARNING = "warning"
DRIFT = "drift"


def psi_term(p_actual: float, p_expected: float) -> float:
    """单个箱的 PSI 贡献。"""
    a = max(p_actual, PSI_FLOOR)
    e = max(p_expected, PSI_FLOOR)
    return (a - e) * math.log(a / e)


def psi_index(actual: list[float], expected: list[float]) -> float:
    """两组等长占比序列的 PSI。两侧序列应分别归一（和为 1）。"""
    if len(actual) != len(expected):
        raise ValueError(
            f"PSI 两侧箱数不一致：actual={len(actual)} expected={len(expected)}"
        )
    total = 0.0
    for a, e in zip(actual, expected):
        total += psi_term(a, e)
    return total


def psi_rating(value: float,
               stable_max: float = settings.psi_stable_max,
               warning_max: float = settings.psi_warning_max) -> str:
    if value < stable_max:
        return STABLE
    if value < warning_max:
        return WARNING
    return DRIFT


def distribution_compare(actual_counts: list[int],
                         expected_counts: list[int]
                         ) -> dict:
    """两组等长箱计数 -> 逐箱占比对比 + PSI。

    actual 侧计数全为 0（区间内没有任何正常落箱记录）时 PSI 无法定义，
    返回 None，由上层决定如何呈现（特征层仍返回逐箱占比）。
    """
    n_act = sum(actual_counts)
    n_exp = sum(expected_counts)
    rows = []
    for c_act, c_exp in zip(actual_counts, expected_counts):
        rows.append({
            "expected_count": c_exp,
            "actual_count": c_act,
            "expected_pct": (c_exp / n_exp) if n_exp else 0.0,
            "actual_pct": (c_act / n_act) if n_act else 0.0,
        })
    value = None
    if n_act > 0 and n_exp > 0:
        value = psi_index(
            [r["actual_pct"] for r in rows],
            [r["expected_pct"] for r in rows],
        )
    return {
        "bins": rows,
        "n_expected": n_exp,
        "n_actual": n_act,
        "psi": value,
        "rating": psi_rating(value) if value is not None else None,
    }
