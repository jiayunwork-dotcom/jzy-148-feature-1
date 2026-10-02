"""人群稳定性：各特征与总分的分布对比 + PSI（Population Stability Index）。

PSI 是评分卡行业通用的稳定性指标：

    PSI = Σ_b (实际占比_b - 基准占比_b) · ln(实际占比_b / 基准占比_b)

两侧占比完全一致时每一项为 0；偏移越大值越大。三档结论沿用行业惯例
（阈值可在查询参数或 app.core.config.settings 中配置）：

    PSI <  0.10  稳定
    0.10 <= PSI < 0.25  需关注
    PSI >= 0.25  显著漂移

零侧处理（本服务的明确约定，与建卡 WOE 的单边平滑保持同一惯例）：
某一箱在基准侧或实际侧占比为 0 时，对该侧该箱**按 0.5 条样本**平滑——
即用 (该侧该箱计数 + 0.5) / 该侧总样本数 替代占比，再代入公式。
其余箱不动。这样：
- 基准侧为 0、实际侧新出现的箱（新箱涌入）贡献为正且随样本量收敛，不会
  因直接取 0/0 而被跳过、也不会给出无穷大；
- 与样本量同阶（小样本下新箱不会被夸大成极端漂移）；
- 两侧同时为 0 时不贡献（不计入求和）。

未见类别 / 数值越界**绝不并入任何箱**：单独报其在实际进件中的占比，
不进 PSI 求和（基准侧两者占比恒为 0，平滑进 PSI 会与「分箱分布漂移」语义
混淆；监管看的是两个独立信号：箱内迁移 vs. 覆盖不了的进件）。
缺失箱是训练时就存在的正式箱，正常参与 PSI。
"""
from __future__ import annotations

import math
from typing import Any

from ..core.config import settings
from .baseline import classify_score

VERDICT_STABLE = "stable"
VERDICT_WARNING = "warning"
VERDICT_SIGNIFICANT = "significant"

_RANK = {VERDICT_STABLE: 0, VERDICT_WARNING: 1, VERDICT_SIGNIFICANT: 2}
_LABEL = {0: VERDICT_STABLE, 1: VERDICT_WARNING, 2: VERDICT_SIGNIFICANT}


def verdict_for(psi: float, warn: float, alarm: float) -> str:
    if psi < warn:
        return VERDICT_STABLE
    if psi < alarm:
        return VERDICT_WARNING
    return VERDICT_SIGNIFICANT


def psi_index(counts_base: list[int], counts_actual: list[int]) -> float | None:
    """两组箱计数上的 PSI。空侧（总计数为 0）返回 None。

    零侧平滑：某个箱任一侧计数为 0 时，该箱两侧计数各按 +0.5 代入占比
    （注意 0.5 只补在零侧；与 WOE 平滑惯例一致）。
    两侧都为 0 的箱跳过。
    """
    nb, na = sum(counts_base), sum(counts_actual)
    if nb == 0 or na == 0:
        return None
    total = 0.0
    for cb, ca in zip(counts_base, counts_actual):
        if cb == 0 and ca == 0:
            continue
        pb = (cb + 0.5) / nb if cb == 0 else cb / nb
        pa = (ca + 0.5) / na if ca == 0 else ca / na
        total += (pa - pb) * math.log(pa / pb)
    return float(total)


def _share(part: int, total: int) -> float:
    return part / total if total else 0.0


def feature_distribution(
    feature_block: dict, records: list[dict[str, Any]]
) -> dict:
    """单个入模特征的稳定性对比（records 为该版本区间内的留痕）。

    聚合按箱**下标** bin_index 对齐（在线落箱时写入留痕），不按 label
    展示字符串——数值箱 label 用 %g 格式化，相邻箱在边界上可能显示成同一串。
    """
    name = feature_block["name"]
    bins = feature_block["bins"]
    missing_label = feature_block["missing_bin"]["label"]
    labels = [b["label"] for b in bins]
    base_counts = [int(b["total"]) for b in bins]
    base_missing = int(feature_block["missing_bin"]["total"])
    n_base = sum(base_counts) + base_missing

    act_counts = [0] * len(bins)
    act_missing = 0
    unseen = out_of_range = 0
    for rec in records:
        fs = _find_feature(rec, name)
        if fs is None:
            continue
        st = fs["status"]
        if st == "missing":
            act_missing += 1
        elif st == "unseen":
            unseen += 1
        elif st == "out_of_range":
            out_of_range += 1
        else:  # status=ok：按落箱下标计数
            idx = fs.get("bin_index")
            if isinstance(idx, int) and 0 <= idx < len(act_counts):
                act_counts[idx] += 1

    n_act = sum(act_counts) + act_missing + unseen + out_of_range

    # 缺失箱作为正式箱参与 PSI
    base_all = base_counts + [base_missing]
    act_all = act_counts + [act_missing]
    psi = psi_index(base_all, act_all)

    bins_out = []
    for i, (cb, ca) in enumerate(zip(base_counts, act_counts)):
        bins_out.append({
            "bin_label": labels[i],
            "bin_index": i,
            "baseline_share": _share(cb, n_base),
            "actual_share": _share(ca, n_act),
            "baseline_count": cb,
            "actual_count": ca,
        })
    bins_out.append({
        "bin_label": missing_label,
        "bin_index": -1,
        "baseline_share": _share(base_missing, n_base),
        "actual_share": _share(act_missing, n_act),
        "baseline_count": base_missing,
        "actual_count": act_missing,
    })

    return {
        "feature": name,
        "psi": psi,
        "n_actual": n_act,
        "bins": bins_out,
        "unseen_rate": _share(unseen, n_act),
        "out_of_range_rate": _share(out_of_range, n_act),
        "unseen_count": unseen,
        "out_of_range_count": out_of_range,
    }


def _find_feature(record: dict[str, Any], name: str) -> dict | None:
    for fs in record.get("features", []):
        if fs["name"] == name:
            return fs
    return None


def score_distribution(
    score_baseline: dict | None, records: list[dict[str, Any]]
) -> dict:
    """总分分布对比。老版本没有总分基准时 unavailable=True。"""
    if not score_baseline:
        return {"available": False,
                "reason": "该版本建卡时未留存总分基准（升级前的老版本）；"
                          "特征层稳定性仍可正常计算，可用原始建卡样本通过 "
                          "score-baseline 接口补基准",
                "n_actual": len(records)}
    edges = score_baseline["edges"]
    base_counts = list(score_baseline["counts"])
    n_base = int(score_baseline["n"])

    n_bin = len(base_counts)
    act_counts = [0] * n_bin
    below = above = 0
    for rec in records:
        bucket = classify_score(float(rec["total_score"]), edges)
        if bucket == "below":
            below += 1
        elif bucket == "above":
            above += 1
        else:
            act_counts[int(bucket)] += 1
    n_act = sum(act_counts) + below + above

    psi = psi_index(base_counts, act_counts)
    bins_out = []
    for i, (cb, ca) in enumerate(zip(base_counts, act_counts)):
        lo, hi = edges[i], edges[i + 1]
        bins_out.append({
            "bin_label": f"[{lo:g}, {hi:g}{']' if i == n_bin - 1 else ')'}",
            "lo": lo, "hi": hi, "right_closed": i == n_bin - 1,
            "baseline_share": _share(cb, n_base),
            "actual_share": _share(ca, n_act),
            "baseline_count": cb,
            "actual_count": ca,
        })
    return {
        "available": True,
        "psi": psi,
        "n_actual": n_act,
        "bins": bins_out,
        "below_range_rate": _share(below, n_act),
        "above_range_rate": _share(above, n_act),
        "below_range_count": below,
        "above_range_count": above,
    }


def stability_report(
    artifacts: dict,
    records: list[dict[str, Any]],
    warn: float = settings.psi_warning,
    alarm: float = settings.psi_significant,
) -> dict:
    """组装一份完整稳定性报告：逐入模特征 + 总分 + 三档结论。"""
    features_out = []
    worst = 0
    for fb in artifacts["features"]:
        if not fb["selected"]:
            continue
        row = feature_distribution(fb, records)
        if row["psi"] is not None:
            row["verdict"] = verdict_for(row["psi"], warn, alarm)
            worst = max(worst, _RANK[row["verdict"]])
        else:
            row["verdict"] = None
        features_out.append(row)

    score_out = score_distribution(artifacts.get("score_baseline"), records)
    if score_out.get("available") and score_out["psi"] is not None:
        score_out["verdict"] = verdict_for(score_out["psi"], warn, alarm)
        worst = max(worst, _RANK[score_out["verdict"]])

    return {
        "n_scored": len(records),
        "thresholds": {"warning": warn, "significant": alarm},
        "features": features_out,
        "total_score": score_out,
        "verdict": _LABEL[worst] if records else VERDICT_STABLE,
    }
