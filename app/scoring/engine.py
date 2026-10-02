"""在线打分：接收原始特征值，自动落箱并汇总总分/PD。

- 缺失（键缺失或 null/空字符串/缺失记号）落「缺失箱」。
- 类别取值在训练中没见过 -> 不落任何箱，按「未见类别」处理：WOE 视为 0，
  分值=该特征截距分摊分，结果中 status="unseen" 并给出该取值，绝不静默归入某箱。
- 数值越出训练范围：夹到首/末箱，status 标注 "out_of_range"。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.sample import MISSING_TOKENS


@dataclass
class FeatureScore:
    name: str
    bin_label: str | None
    woe: float | None
    score: float
    status: str                 # ok | missing | unseen | out_of_range
    raw_value: object = None
    detail: str = ""
    bin_index: int | None = None   # 常规箱下标（缺失=-1，未见/越界=None）


class ScorecardRuntime:
    def __init__(self, artifacts: dict):
        p = artifacts["params"]
        self.card_type_index = {f["name"]: f for f in artifacts["features"]}
        self.selected = artifacts["selected_features"]
        beta = artifacts["regression"]["beta"]
        sc = artifacts["scoring"]
        self.base_score = float(sc["base_score"])
        self.factor = float(sc["factor"])
        self.base_odds = float(sc.get("base_odds", p.get("base_odds", 1.0)))
        self.constant_total = float(
            sc.get("constant_total",
                   self.base_score - self.factor * math.log(self.base_odds)))
        self.intercept_part = float(sc["intercept_part_per_feature"])
        self.pdo = float(p["pdo"])
        self.base_odds = float(p["base_odds"])
        self.bin_scores = sc["bin_scores"]
        self.missing_scores = sc["missing_scores"]
        # 名称 -> 系数
        self.beta = {name: float(beta[i + 1]) for i, name in enumerate(self.selected)}

    def _coerce_numeric(self, raw) -> float:
        if isinstance(raw, bool):
            return float(raw)
        if isinstance(raw, (int, float)):
            return float(raw)
        raise ValueError(f"数值特征收到非数值 {raw!r}")

    def _score_one_feature(self, name: str, raw) -> FeatureScore:
        fblock = self.card_type_index[name]
        ftype = fblock["type"]
        present = not (raw is None or (
            isinstance(raw, str) and raw.strip().lower() in MISSING_TOKENS))

        if not present:
            b = fblock["missing_bin"]
            return FeatureScore(
                name=name, bin_label=b["label"], woe=b["woe"],
                score=self.missing_scores[name], status="missing", raw_value=raw,
                bin_index=-1,
            )

        if ftype == "numeric":
            x = self._coerce_numeric(raw)
            if math.isnan(x):
                b = fblock["missing_bin"]
                return FeatureScore(
                    name=name, bin_label=b["label"], woe=b["woe"],
                    score=self.missing_scores[name], status="missing", raw_value=raw,
                    bin_index=-1,
                )
            bins = fblock["bins"]
            if not bins:
                # 训练时该特征全缺失：非缺失输入无处可落，按未见处理
                return FeatureScore(
                    name=name, bin_label=None, woe=0.0,
                    score=self.intercept_part, status="unseen", raw_value=raw,
                    detail="该特征训练样本全部为缺失，非缺失值无处落箱，按未见处理",
                )
            if x < bins[0]["lo"]:
                idx = 0
                status, detail = "out_of_range", f"{x:g} 低于训练最小值 {bins[0]['lo']:g}，夹到首箱"
            elif x > bins[-1]["hi"]:
                idx = len(bins) - 1
                status, detail = "out_of_range", f"{x:g} 高于训练最大值 {bins[-1]['hi']:g}，夹到末箱"
            elif x <= bins[0]["lo"]:
                idx = 0
                status, detail = "ok", ""
            else:
                idx = None
                for k, b in enumerate(bins):
                    lo, hi = b["lo"], b["hi"]
                    if b["right_closed"]:
                        hit = lo < x <= hi
                    else:
                        hit = lo <= x < hi
                    if hit:
                        idx = k
                        break
                if idx is None:
                    idx = len(bins) - 1
                status, detail = "ok", ""
            b = bins[idx]
            return FeatureScore(
                name=name, bin_label=b["label"], woe=b["woe"],
                score=self.bin_scores[name][idx],
                status=status, raw_value=raw, detail=detail,
                bin_index=idx,
            )

        # categorical
        sval = str(raw)
        for idx, b in enumerate(fblock["bins"]):
            if sval in b["categories"]:
                return FeatureScore(
                    name=name, bin_label=b["label"], woe=b["woe"],
                    score=self.bin_scores[name][idx],
                    status="ok", raw_value=raw, bin_index=idx,
                )
        return FeatureScore(
            name=name, bin_label=None, woe=0.0,
            score=self.intercept_part, status="unseen", raw_value=raw,
            detail=f"训练未见类别 {sval!r}，WOE 按 0 处理（不落任何箱）",
        )

    def score(self, applicant: dict) -> dict:
        feats = []
        total = 0.0  # 常量已按特征分摊：总分 = Σ 各箱分值
        for name in self.selected:
            if name not in applicant:
                raw = None
            else:
                raw = applicant[name]
            fs = self._score_one_feature(name, raw)
            feats.append(fs)
            total += fs.score

        logit_pd = (
            -math.log(self.base_odds)
            - (total - self.base_score) / self.factor
        )
        logit_pd = min(700.0, max(-700.0, logit_pd))
        pd = 1.0 / (1.0 + math.exp(-logit_pd))
        return {
            "total_score": total,
            "pd": pd,
            "features": [vars(f) for f in feats],
            "has_unseen": any(f.status == "unseen" for f in feats),
            "has_missing": any(f.status == "missing" for f in feats),
        }
