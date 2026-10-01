"""全局配置。所有可调阈值集中在这里。"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://scorecard:scorecard@localhost:5432/scorecard"
    )
    min_samples: int = 500                # 作业开始前的最小样本量
    default_max_bins: int = 20            # 数值特征等频初始箱上限
    default_min_bin_pct: float = 0.05     # 每箱样本占比下限
    default_iv_threshold: float = 0.02    # 未指定入模特征时的 IV 筛选阈值
    default_base_score: float = 600.0
    default_base_odds: float = 20.0       # 基准分处的 好/坏 odds
    default_pdo: float = 50.0
    newton_max_iter: int = 100
    newton_tol: float = 1e-10
    batch_size_limit: int = 10_000


settings = Settings()
