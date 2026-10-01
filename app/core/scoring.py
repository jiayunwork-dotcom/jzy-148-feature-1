"""评分换算。

约定：odds = 好/坏 = P(好)/P(坏) = (1-PD)/PD，分数越高越安全。
基准 odds（好/坏）处得基准分；总分每增加一个 PDO，odds 翻倍。标准公式：

    factor = PDO / ln2
    Score  = base_score - factor*ln(base_odds) + factor*ln(odds)
           = base_score + factor*ln(odds / base_odds)

由 logit(PD) = b0 + Σ b_j·WOE_j，ln(odds) = -logit(PD)：

    Score = base_score - factor*ln(base_odds) - factor*(b0 + Σ b_j·WOE_j)

截距分摊：常量 base_score - factor*ln(base_odds) - factor*b0 不单独成项，
按入模特征数 k 等分到各特征，因此

    每箱分值 = C/k - factor*b_j*WOE
    某申请人总分 = Σ 各特征落箱分值
且在 odds=base_odds 时总分恰为 base_score。

未见类别（打分时出现训练未见的类别取值）：WOE 视为 0，分值等于该特征的
常量分摊 C/k，并在打分结果中标注 status="unseen"，不静默归入任何箱。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoringTables:
    feature_names: list[str]
    beta: np.ndarray            # [b0, b1, ..., bk]
    factor: float
    base_score: float
    base_odds: float
    constant_total: float       # base_score - factor*ln(base_odds) - factor*b0
    intercept_part: float       # 每特征分摊 constant_total / k
    score_of_bin: dict[str, list[float]]   # 特征 -> 各常规箱分值
    missing_scores: dict[str, float]


def build_scoring_tables(
    beta: np.ndarray,
    feature_names: list[str],
    binnings,
    base_score: float,
    base_odds: float,
    pdo: float,
) -> ScoringTables:
    k = len(feature_names)
    factor = pdo / np.log(2.0)
    constant_total = (
        float(base_score) - factor * np.log(float(base_odds))
        - factor * float(beta[0])
    )
    intercept_part = constant_total / k

    score_of_bin: dict[str, list[float]] = {}
    missing_scores: dict[str, float] = {}
    for j, name in enumerate(feature_names, start=1):
        fb = binnings[name]
        score_of_bin[name] = [
            float(intercept_part - factor * beta[j] * b.woe) for b in fb.bins
        ]
        missing_scores[name] = float(
            intercept_part - factor * beta[j] * fb.missing_bin.woe
        )

    return ScoringTables(
        feature_names=feature_names,
        beta=np.asarray(beta, dtype=np.float64),
        factor=float(factor),
        base_score=float(base_score),
        base_odds=float(base_odds),
        constant_total=constant_total,
        intercept_part=float(intercept_part),
        score_of_bin=score_of_bin,
        missing_scores=missing_scores,
    )


def pd_from_score(total_score: float, tables: ScoringTables) -> float:
    """由总分反算 PD。

    Score = base_score + factor*(ln(odds) - ln(base_odds))
    => logit(PD) = -ln(odds) = -ln(base_odds) - (Score - base_score)/factor。
    """
    logit_pd = (
        -np.log(tables.base_odds)
        - (total_score - tables.base_score) / tables.factor
    )
    logit_pd = min(700.0, max(-700.0, float(logit_pd)))
    return float(1.0 / (1.0 + np.exp(-logit_pd)))
