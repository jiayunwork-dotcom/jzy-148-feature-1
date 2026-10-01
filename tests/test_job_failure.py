"""建卡作业失败路径：失败时作业 failed 且原因、迭代历史可查。"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import BuildError
from app.core.pipeline import BuildParams, run_pipeline
from app.core.sample import Sample

from tests.conftest import build_and_wait


def test_build_failed_job_records_reason_and_history(client):
    # 两个类别箱各 300 行、坏率 100%/0%：完全对称的极端 WOE，IV 平滑后为 0，
    # 作业在入模前被拒绝（或回归不收敛），两种失败都必须留下明确原因
    lines = ["label,grp"]
    for i in range(600):
        g = "A" if i < 300 else "B"
        y = 1 if i < 300 else 0
        lines.append(f"{y},{g}")
    content = ("\n".join(lines) + "\n").encode()

    job = build_and_wait(client, "sep_card", content, features="grp")
    assert job["status"] == "failed"
    assert job["error"]
    assert ("不收敛" in job["error"] or "发散" in job["error"]
            or "IV 为 0" in job["error"])
    # 失败不产生版本
    assert client.get("/cards/sep_card/versions").json() == []


def test_numeric_separation_reaches_regression_and_fails():
    """三个三值特征：每个边际箱好坏都混合（不会被非零合并消箱），
    但标签由 WOE 线性组合完美切开（联合完全分离），牛顿迭代必须失败并带历史。"""
    n = 2700
    g = np.array([i % 27 for i in range(n)])
    a = (g % 3).astype(float)
    b = ((g // 3) % 3).astype(float)
    c = ((g // 9) % 3).astype(float)
    y = (a + b + c >= 4).astype(float)
    # 每个边际取值都有好有坏
    for col in (a, b, c):
        rates = [y[col == k].mean() for k in (0, 1, 2)]
        assert 0.0 < min(rates) and max(rates) < 1.0

    sample = Sample(
        label=y,
        features={"a": a, "b": b, "c": c},
        types={"a": "numeric", "b": "numeric", "c": "numeric"},
    )
    with pytest.raises(BuildError) as ei:
        run_pipeline(sample, BuildParams(
            features=["a", "b", "c"], min_bin_pct=0.05))
    assert "不收敛" in str(ei.value) or "发散" in str(ei.value)
    assert getattr(ei.value, "history", None)
