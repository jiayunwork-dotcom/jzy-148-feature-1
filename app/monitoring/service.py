"""人群稳定性服务：查询时从打分记录现算，不做增量计数。

取舍（详见 README「监控统计的实现取舍」）
-----------------------------------------
打分记录 score_logs 是唯一事实源，所有统计在查询时按
(card, version, [start,end)) 现算：
- 写入开销恒定（打分路径只多一条 insert），不维护任何计数桶；
- 支持任意起止时间，不被分桶粒度限制；
- 任何时候都能用打分记录逐条重算对账（测试即以此验证一致性）；
- 服务重启结果不变（无内存状态）；
- 代价：查询随区间内打分量线性增长，靠 (card_name, version, scored_at)
  索引与 monitor_query_limit 兜底，月度监控量级下毫秒到几十毫秒。

特征基准取版本产物逐箱 total=bad+good（升级前的老版本同样具备）；
总分基准仅新版本在 artifacts.monitoring.score_baseline 中留存，
老版本查不到时 score 部分返回 status="unavailable" 并说明原因。
"""
from __future__ import annotations

from datetime import datetime

from ..core.config import settings
from ..storage.repository import Repository
from .baseline import assign_score_bin
from .psi import distribution_compare


class MonitoringService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def stability(self, card_name: str, version: int,
                  start: datetime | None, end: datetime | None) -> dict:
        artifact = self.repo.load_version(card_name, version)
        logs = self.repo.query_score_logs(
            card_name, version, start, end, settings.monitor_query_limit)

        features_out = []
        for fblock in artifact["features"]:
            if not fblock.get("selected", False):
                continue
            features_out.append(self._feature_stability(fblock, logs))

        score_out = self._score_stability(artifact, logs)

        psi_values = [f["psi"] for f in features_out if f["psi"] is not None]
        if score_out.get("psi") is not None:
            psi_values.append(score_out["psi"])
        from .psi import psi_rating
        overall = psi_rating(max(psi_values)) if psi_values else None

        return {
            "card_name": card_name,
            "version": version,
            "window": {"start": _iso(start), "end": _iso(end)},
            "n_scored": len(logs),
            "truncated": len(logs) >= settings.monitor_query_limit,
            "query_limit": settings.monitor_query_limit,
            "features": features_out,
            "total_score": score_out,
            "overall_rating": overall,
        }

    # ---------------------------------------------------------- 特征层
    @staticmethod
    def _feature_stability(fblock: dict, logs: list[dict]) -> dict:
        n_bins = len(fblock["bins"])
        # 常规箱 0..n_bins-1，缺失箱单独一档；unseen/oor 只报占比不进 PSI
        actual = [0] * (n_bins + 1)
        expected = [0] * (n_bins + 1)
        for i, b in enumerate(fblock["bins"]):
            expected[i] = b["bad"] + b["good"]
        expected[n_bins] = (fblock["missing_bin"]["bad"]
                            + fblock["missing_bin"]["good"])

        n_unseen = n_oor = 0
        name = fblock["name"]
        for log in logs:
            entry = _find_feature(log, name)
            status, label = entry["status"], entry["bin_label"]
            if status == "unseen":
                n_unseen += 1
                continue
            if status == "out_of_range":
                n_oor += 1
            idx = _bin_index(fblock, status, label, n_bins)
            actual[idx] += 1

        comp = distribution_compare(actual, expected)
        rows = comp["bins"]
        # 给每行带上可读标签（最后一档为缺失箱）
        for i, row in enumerate(rows):
            row["bin_label"] = (fblock["bins"][i]["label"] if i < n_bins
                                else fblock["missing_bin"]["label"])
        n = len(logs)
        return {
            "name": name,
            "type": fblock["type"],
            "bins": rows,
            "psi": comp["psi"],
            "rating": comp["rating"],
            "unseen_pct": (n_unseen / n) if n else 0.0,
            "out_of_range_pct": (n_oor / n) if n else 0.0,
            "unseen_count": n_unseen,
            "out_of_range_count": n_oor,
        }

    # ---------------------------------------------------------- 总分层
    @staticmethod
    def _score_stability(artifact: dict, logs: list[dict]) -> dict:
        mon = artifact.get("monitoring")
        if not mon or "score_baseline" not in mon:
            return {
                "status": "unavailable",
                "reason": (
                    "该版本在监控功能上线前建卡，产物中没有建卡样本总分基准，"
                    "且总分联合分布无法由各特征边际计数回推；特征层稳定性不受影响。"
                    "可重新建卡（新版本）获取总分基准。"
                ),
                "n_scored": len(logs),
            }
        base = mon["score_baseline"]
        sbins = base["bins"]
        actual = [0] * len(sbins)
        n_oor = 0
        for log in logs:
            idx = assign_score_bin(sbins, log["total_score"])
            if idx is None:
                n_oor += 1
            else:
                actual[idx] += 1
        comp = distribution_compare(actual, base["counts"])
        rows = [{**row, "bin_label": sb["label"]}
                for row, sb in zip(comp["bins"], sbins)]
        n = len(logs)
        return {
            "status": "ok",
            "bins": rows,
            "psi": comp["psi"],
            "rating": comp["rating"],
            "out_of_range_count": n_oor,
            "out_of_range_pct": (n_oor / n) if n else 0.0,
            "baseline_range": {"lo": base["lo"], "hi": base["hi"]},
        }


def _find_feature(log: dict, name: str) -> dict:
    for f in log["feature_bins"]:
        if f["name"] == name:
            return f
    raise KeyError(f"打分记录缺少特征 {name}，卡版本产物与记录不一致")


def _bin_index(fblock: dict, status: str, label: str | None,
               n_bins: int) -> int:
    if status == "missing":
        return n_bins
    for i, b in enumerate(fblock["bins"]):
        if b["label"] == label:
            return i
    # 越界记录会夹到首/末箱，标签必然匹配；匹配不到属于数据损坏
    raise KeyError(
        f"特征 {fblock['name']} 的落箱标签 {label!r} 在版本产物中找不到")


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None
