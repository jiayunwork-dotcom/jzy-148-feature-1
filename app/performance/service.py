"""表现统计：在已有实际标签的放款上计算实际坏率、平均预测 PD、KS、AUC，
并按总分分段给出预测/实际坏率对比。

KS / AUC 直接调用建卡时的同一套实现 app.core.metrics（ks_stat / roc_auc），
输入就是这批放款打分记录里逐条保存的 PD 与回填的 0/1 标签，不另写口径。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from ..core.config import settings
from ..core.metrics import ks_stat, roc_auc
from ..monitoring.baseline import assign_score_bin, build_score_baseline
from ..storage.repository import Repository


class PerformanceService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def report(self, card_name: str, version: int,
               start: datetime | None, end: datetime | None,
               n_bands: int = 10) -> dict:
        artifact = self.repo.load_version(card_name, version)
        rows = self.repo.query_performance(
            card_name, version, start, end, settings.monitor_query_limit)

        labels = np.array([r["label"] for r in rows], dtype=np.float64)
        pds = np.array([r["pd"] for r in rows], dtype=np.float64)
        scores = np.array([r["total_score"] for r in rows], dtype=np.float64)

        n = len(rows)
        out: dict = {
            "card_name": card_name,
            "version": version,
            "window": {"start": _iso(start), "end": _iso(end)},
            "n_performed": n,
            "bad_count": int(labels.sum()) if n else 0,
            "good_count": int(n - labels.sum()) if n else 0,
            "actual_bad_rate": None,
            "avg_predicted_pd": None,
            "ks": None,
            "auc": None,
            "score_bands": [],
            "truncated": n >= settings.monitor_query_limit,
            "query_limit": settings.monitor_query_limit,
            "note": None,
        }
        if n == 0:
            out["note"] = "区间内没有已回填表现的打分记录"
            return out

        actual_bad_rate = float(labels.mean())
        avg_pd = float(pds.mean())
        out["actual_bad_rate"] = actual_bad_rate
        out["avg_predicted_pd"] = avg_pd

        # 与建卡完全同一套评估口径；标签单一类时 KS/AUC 无定义
        if out["bad_count"] == 0 or out["good_count"] == 0:
            out["note"] = (
                f"这批放款标签全为 {int(labels[0])}，KS/AUC 需要好坏两类，"
                "按建卡评估同样的规则返回 null"
            )
        else:
            out["ks"] = float(ks_stat(labels, pds))
            out["auc"] = float(roc_auc(labels, pds))

        out["score_bands"] = self._score_bands(artifact, scores, pds, labels,
                                               n_bands)
        out["truncated"] = n >= settings.monitor_query_limit
        out["query_limit"] = settings.monitor_query_limit
        return out

    # ---------------------------------------------------------- 分段
    @staticmethod
    def _score_bands(artifact: dict, scores: np.ndarray, pds: np.ndarray,
                     labels: np.ndarray, n_bands: int) -> list[dict]:
        """分段优先用建卡总分基准箱（与稳定性同口径）；老版本没有基准时，
        用这批实际分数现场切等频箱（标注 baseline_source="observed"）。"""
        mon = artifact.get("monitoring") or {}
        base = mon.get("score_baseline")
        observed_mode = base is None
        if observed_mode:
            ob = build_score_baseline(scores, n_bins=n_bands)
            bins = ob["bins"]
            range_note = None
        else:
            bins = base["bins"]

        n_bins = len(bins)
        agg = [{"n": 0, "bad": 0, "pd_sum": 0.0} for _ in range(n_bins)]
        oor = {"n": 0, "bad": 0, "pd_sum": 0.0}
        for s, p, y in zip(scores, pds, labels):
            idx = assign_score_bin(bins, float(s))
            bucket = oor if idx is None else agg[idx]
            bucket["n"] += 1
            bucket["bad"] += int(y)
            bucket["pd_sum"] += float(p)

        bands = []
        for i, (b, a) in enumerate(zip(bins, agg)):
            bands.append({
                "bin_label": b["label"],
                "lo": b["lo"], "hi": b["hi"],
                "right_closed": b["right_closed"],
                "count": a["n"],
                "bad_count": a["bad"],
                "actual_bad_rate": (a["bad"] / a["n"]) if a["n"] else None,
                "avg_predicted_pd": (a["pd_sum"] / a["n"]) if a["n"] else None,
            })
        if oor["n"]:
            bands.append({
                "bin_label": "越界（总分超出基准范围）",
                "lo": None, "hi": None, "right_closed": False,
                "count": oor["n"], "bad_count": oor["bad"],
                "actual_bad_rate": oor["bad"] / oor["n"],
                "avg_predicted_pd": oor["pd_sum"] / oor["n"],
            })
        return bands


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None
