"""建卡流水线：把一个开发样本从原始 CSV 变成可版本化、可打分的完整产物。

流水线是纯函数式编排：所有中间状态只存在于本次调用栈与返回的 artifacts 中，
并发作业互不共享任何可变对象，天然隔离。

返回的 artifacts 是可 JSON 序列化的 dict，直接作为版本产物落库：
- params            建卡参数
- data_summary      样本量、好坏数、违约率、缺失率
- features[].binning 各特征分箱、好坏计数、WOE/IV、合并日志（中间结果）
- selected_features 入模特征及入选原因
- regression        beta、WOE 设计矩阵摘要、牛顿迭代历史
- scoring           factor、基准分、截距分摊说明、每箱分值表
- metrics           KS、AUC、各特征 IV
- validation        收敛校验：平均预测概率 vs 样本违约率
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .binning import FeatureBinning, bin_feature
from .config import settings
from .exceptions import BuildError, ValidationError
from .metrics import ks_stat, roc_auc
from ..monitor.baseline import build_score_baseline
from .regression import fit_logistic, predict_proba
from .sample import Sample
from .scoring import build_scoring_tables
from .transform import build_design_matrix, values_to_woe


@dataclass
class BuildParams:
    label_col: str = "label"
    features: list[str] | None = None        # 指定入模清单；None 按 IV 阈值自动
    iv_threshold: float = settings.default_iv_threshold
    max_bins: int = settings.default_max_bins
    min_bin_pct: float = settings.default_min_bin_pct
    base_score: float = settings.default_base_score
    base_odds: float = settings.default_base_odds
    pdo: float = settings.default_pdo

    def validate(self, sample: Sample) -> None:
        if self.min_bin_pct <= 0 or self.min_bin_pct >= 1:
            raise ValidationError("min_bin_pct 必须在 (0,1) 之间")
        if self.max_bins < 2:
            raise ValidationError("max_bins 至少为 2")
        if self.pdo <= 0:
            raise ValidationError("PDO 必须为正数")
        if self.base_odds <= 0:
            raise ValidationError("基准 odds 必须为正数")
        if self.iv_threshold < 0:
            raise ValidationError("IV 阈值不能为负")
        if self.features:
            missing = [f for f in self.features if f not in sample.features]
            if missing:
                raise ValidationError(
                    f"入模特征在样本中不存在：{missing}；可用特征：{sample.feature_names}"
                )
            if len(set(self.features)) != len(self.features):
                raise ValidationError("入模特征清单存在重复")
        if sample.n < settings.min_samples:
            raise ValidationError(
                f"开发样本仅 {sample.n} 行，少于最小要求 {settings.min_samples} 行"
            )
        uniq = set(sample.label.tolist())
        if uniq - {0.0, 1.0}:
            raise ValidationError("标签不是 0/1")
        if len(uniq) < 2:
            raise ValidationError("标签全为同一类，无法建卡")


def binning_to_json(fb: FeatureBinning) -> dict:
    return {
        "name": fb.name,
        "type": fb.ftype,
        "iv": fb.iv,
        "direction": fb.direction,
        "total_bad": fb.total_bad,
        "total_good": fb.total_good,
        "bins": [vars(b) for b in fb.bins],
        "missing_bin": vars(fb.missing_bin),
        "journal": fb.journal,
    }


def _missing_rate(sample: Sample, name: str) -> float:
    v = sample.features[name]
    if sample.types[name] == "numeric":
        return float(np.mean(np.isnan(v)))
    return float(np.mean([x is None for x in v]))


def _sample_rows_as_applicants(sample: Sample) -> list[dict]:
    """把训练样本转成在线打分入参形态（数值 NaN -> None，类别 None 保留）。

    这样总分基准由**在线引擎逐行打分**得到，与投产真实调用逐位同路径；
    建卡样本原样回放时总分 PSI 严格为 0 才有保证。
    """
    rows: list[dict] = [{} for _ in range(sample.n)]
    for name in sample.feature_names:
        col = sample.features[name]
        if sample.types[name] == "numeric":
            for i, v in enumerate(col):
                rows[i][name] = None if np.isnan(v) else float(v)
        else:
            for i, v in enumerate(col):
                rows[i][name] = v
    return rows


def run_pipeline(sample: Sample, params: BuildParams) -> dict:
    params.validate(sample)

    all_features = sample.feature_names
    if not all_features:
        raise BuildError("样本中没有可用特征列")

    # 1) 全特征分箱（IV 表需要全量结果，中间结果全部保留）
    binnings: dict[str, FeatureBinning] = {}
    for name in all_features:
        binnings[name] = bin_feature(
            name, sample.types[name], sample.features[name], sample.label,
            max_bins=params.max_bins, min_bin_pct=params.min_bin_pct,
        )

    # 2) 入模特征：指定清单优先，否则按 IV 阈值自动筛选
    iv_table = {name: binnings[name].iv for name in all_features}
    if params.features:
        selected = list(params.features)
        reasons = {n: "用户指定" for n in selected}
        zero_iv = [n for n in selected if binnings[n].iv <= 0.0]
        if zero_iv:
            raise BuildError(
                f"指定入模特征 {zero_iv} 的 IV 为 0（分箱后好坏比例与总体完全相同），"
                "该特征在 WOE 逻辑回归中是常数列，无法拟合"
            )
    else:
        selected = sorted(
            (n for n in all_features if iv_table[n] >= params.iv_threshold),
            key=lambda n: (-iv_table[n], n),
        )
        reasons = {n: f"IV={iv_table[n]:.5f} >= 阈值 {params.iv_threshold:g}"
                   for n in selected}
        if not selected:
            raise BuildError(
                f"IV 阈值 {params.iv_threshold:g} 下没有特征入选"
                f"（最大 IV={max(iv_table.values()):.5f}）"
            )

    # 3) WOE 设计矩阵 + 自研牛顿逻辑回归
    X = build_design_matrix(sample, binnings, selected)
    beta, n_iter, history = fit_logistic(
        X, sample.label,
        max_iter=settings.newton_max_iter, tol=settings.newton_tol,
    )
    p_hat = predict_proba(X, beta)

    avg_p = float(np.mean(p_hat))
    bad_rate = float(np.mean(sample.label))
    calib_err = abs(avg_p - bad_rate)
    if calib_err > 1e-6:
        raise BuildError(
            f"收敛校验失败：平均预测 PD={avg_p:.8f} 与样本违约率={bad_rate:.8f} "
            f"相差 {calib_err:.2e} > 1e-6"
        )

    # 4) 评分换算
    tables = build_scoring_tables(
        beta, selected, binnings,
        base_score=params.base_score,
        base_odds=params.base_odds,
        pdo=params.pdo,
    )

    # 5) 训练指标
    ks = ks_stat(sample.label, p_hat)
    auc = roc_auc(sample.label, p_hat)

    feature_blocks = []
    for name in all_features:
        fb = binnings[name]
        block = binning_to_json(fb)
        block["missing_rate"] = _missing_rate(sample, name)
        block["selected"] = name in selected
        block["selection_reason"] = reasons.get(name, "")
        if name in selected:
            j = selected.index(name) + 1
            block["beta"] = float(beta[j])
        feature_blocks.append(block)

    artifacts = {
        "params": {
            "label_col": params.label_col,
            "features": params.features,
            "iv_threshold": params.iv_threshold,
            "max_bins": params.max_bins,
            "min_bin_pct": params.min_bin_pct,
            "base_score": params.base_score,
            "base_odds": params.base_odds,
            "pdo": params.pdo,
        },
        "data_summary": {
            "n": int(sample.n),
            "bad": int(sample.label.sum()),
            "good": int(sample.n - sample.label.sum()),
            "bad_rate": bad_rate,
        },
        "features": feature_blocks,
        "selected_features": selected,
        "regression": {
            "beta": [float(v) for v in beta],
            "feature_names": ["(intercept)"] + selected,
            "n_iter": n_iter,
            "history": history,
            "tol": settings.newton_tol,
        },
        "scoring": {
            "base_score": tables.base_score,
            "base_odds": tables.base_odds,
            "factor": tables.factor,
            "constant_total": tables.constant_total,
            "intercept_part_per_feature": tables.intercept_part,
            "intercept_policy": (
                "Score = base_score + factor*ln(odds/base_odds)，"
                "factor = PDO/ln2；常量 "
                "base_score - factor*ln(base_odds) - factor*b0 = "
                f"{tables.constant_total:.6f} 按入模特征数 {len(selected)} 等分"
                f"（每特征 {tables.intercept_part:.6f} 分），总分=Σ各特征落箱分值，"
                "odds=base_odds 时总分恰为 base_score；缺失使用缺失箱 WOE；"
                "未见类别 WOE 视为 0，分值=常量分摊分并标注 unseen"
            ),
            "bin_scores": tables.score_of_bin,
            "missing_scores": tables.missing_scores,
        },
        "metrics": {
            "ks": ks,
            "auc": auc,
            "iv": {n: iv_table[n] for n in all_features},
        },
        "validation": {
            "avg_predicted_pd": avg_p,
            "sample_bad_rate": bad_rate,
            "abs_error": calib_err,
            "passed": calib_err <= 1e-6,
        },
    }
    # 总分分布基准：训练样本经**在线引擎**逐条打分后按等频箱留存。
    # 边际分箱计数推不出总分联合分布，故新版本建卡时必须在此一次性留好。
    artifacts["score_baseline"] = build_score_baseline(
        artifacts, _sample_rows_as_applicants(sample),
        n_bins=settings.score_baseline_bins,
    )
    return artifacts
