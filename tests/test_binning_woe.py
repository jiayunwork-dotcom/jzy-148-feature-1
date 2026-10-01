"""分箱与 WOE/IV 的手算核对、约束检查、标签取反、样本复制不变性。

手算样本 data/handcalc_sample.csv（28 行，11 坏 17 好）：
  x_num 三个取值箱：1(6坏4好), 2(3坏6好), 3(1坏6好)，缺失箱(1坏1好)
  x_cat 三个类别箱：A(5坏3好), B(3坏4好), C(1坏6好)，缺失箱(2坏4好)

WOE = ln((坏数/总坏)/(好数/总好))，IV = Σ (坏占比-好占比)*WOE。
"""
from __future__ import annotations

import math

import numpy as np

from app.core.binning import bin_feature
from app.core.sample import parse_csv

B, G = 11, 17


def _load():
    sample = parse_csv(
        open("data/handcalc_sample.csv", "rb").read())
    return sample


def test_handcalc_numeric_woe_and_iv():
    s = _load()
    # 手算样本只有 28 行，直接调分箱（min_bin_pct 放宽，避免小样本合并）
    fb = bin_feature("x_num", s.types["x_num"], s.features["x_num"], s.label,
                     max_bins=20, min_bin_pct=0.0)
    by_lo = {b.lo: b for b in fb.bins}

    def woe(b, g):
        return math.log((b / B) / (g / G))

    assert by_lo[1.0].bad == 6 and by_lo[1.0].good == 4
    assert by_lo[2.0].bad == 3 and by_lo[2.0].good == 6
    assert by_lo[3.0].bad == 1 and by_lo[3.0].good == 6
    assert by_lo[1.0].woe == pytest_approx(woe(6, 4))
    assert by_lo[2.0].woe == pytest_approx(woe(3, 6))
    assert by_lo[3.0].woe == pytest_approx(woe(1, 6))

    # 缺失箱始终单独：1 坏 1 好
    assert fb.missing_bin.is_missing
    assert fb.missing_bin.bad == 1 and fb.missing_bin.good == 1
    assert fb.missing_bin.woe == pytest_approx(woe(1, 1))

    iv_reg = sum((b.bad / B - b.good / G) * b.woe for b in fb.bins)
    iv_miss = (1 / B - 1 / G) * fb.missing_bin.woe
    assert fb.iv == pytest_approx(iv_reg + iv_miss, abs=1e-12)
    assert fb.iv == pytest_approx(0.650858, abs=1e-5)


def test_handcalc_categorical_woe_and_iv():
    s = _load()
    fb = bin_feature("x_cat", s.types["x_cat"], s.features["x_cat"], s.label,
                     max_bins=20, min_bin_pct=0.0)
    by_cat = {b.categories[0]: b for b in fb.bins}

    def woe(b, g):
        return math.log((b / B) / (g / G))

    assert by_cat["A"].bad == 5 and by_cat["A"].good == 3
    assert by_cat["B"].bad == 3 and by_cat["B"].good == 4
    assert by_cat["C"].bad == 1 and by_cat["C"].good == 6
    assert by_cat["A"].woe == pytest_approx(woe(5, 3))
    assert by_cat["B"].woe == pytest_approx(woe(3, 4))
    assert by_cat["C"].woe == pytest_approx(woe(1, 6))
    # 类别按坏样率排序后 WOE 严格递增
    woe_seq = [b.woe for b in fb.bins]
    assert all(woe_seq[i + 1] - woe_seq[i] > 0 for i in range(len(woe_seq) - 1))
    # 缺失箱 2 坏 4 好
    assert fb.missing_bin.bad == 2 and fb.missing_bin.good == 4
    assert fb.iv == pytest_approx(0.637844, abs=1e-5)


def pytest_approx(v, abs=1e-9):
    import pytest
    return pytest.approx(v, abs=abs)


# ---------------------------------------------------------------- 约束

def test_numeric_bins_satisfy_constraints(dev_csv_bytes):
    s = parse_csv(dev_csv_bytes)
    n = s.n
    for name in ("age", "income", "debt_ratio"):
        v = s.features[name]
        present = v[~np.isnan(v)]
        fb = bin_feature(name, "numeric", v, s.label,
                         max_bins=20, min_bin_pct=0.05)
        # 覆盖全部非缺失样本且互不重叠（箱数计数和=非缺失行数）
        assert sum(b.total for b in fb.bins) == len(present)
        # 缺失单独成箱
        assert fb.missing_bin.is_missing
        assert fb.missing_bin.total == n - len(present)
        # 每箱好坏非零
        assert all(b.bad > 0 and b.good > 0 for b in fb.bins)
        # 每箱占比 >= 5%（分母为非缺失样本数）
        assert all(b.total / len(present) >= 0.05 - 1e-9 for b in fb.bins)
        # WOE 单调（按数据方向；非严格）
        ws = [b.woe for b in fb.bins]
        if fb.direction == -1:
            assert all(ws[i + 1] <= ws[i] + 1e-12 for i in range(len(ws) - 1))
        elif fb.direction == 1:
            assert all(ws[i + 1] >= ws[i] - 1e-12 for i in range(len(ws) - 1))
        # 箱数不超过初始上限
        assert len(fb.bins) <= 20


def test_categorical_missing_always_separate(dev_csv_bytes):
    s = parse_csv(dev_csv_bytes)
    fb = bin_feature("job_grade", "categorical", s.features["job_grade"], s.label)
    cats_in_bins = {c for b in fb.bins for c in b.categories}
    assert None not in cats_in_bins
    assert fb.missing_bin.total > 0  # job_grade 5% 缺失


def test_label_inversion_woe_sign_and_iv(dev_csv_bytes):
    """标签整体取反：各箱 WOE 变号、IV 不变。"""
    s = parse_csv(dev_csv_bytes)
    for name in ("age", "city"):
        ftype = s.types[name]
        fb = bin_feature(name, ftype, s.features[name], s.label)
        fb_inv = bin_feature(name, ftype, s.features[name], 1.0 - s.label)

        # 取反后排序方向相反，但箱成员应一致；按边界/类别对齐后 WOE 变号、IV 不变
        if ftype == "numeric":
            key = lambda b: (b.lo, b.hi)
        else:
            key = lambda b: b.categories
        a = {key(b): b for b in fb.bins}
        bb = {key(b): b for b in fb_inv.bins}
        assert set(a) == set(bb)
        for k in a:
            assert a[k].woe + bb[k].woe == pytest_approx(0.0, abs=1e-12)
            assert a[k].bad == bb[k].good and a[k].good == bb[k].bad
        assert fb.iv == pytest_approx(fb_inv.iv, abs=1e-12)
        # 缺失箱同样变号、IV 不变
        assert (fb.missing_bin.woe + fb_inv.missing_bin.woe
                == pytest_approx(0.0, abs=1e-12))
        m1 = fb.missing_bin.iv_contrib
        m2 = fb_inv.missing_bin.iv_contrib
        assert m1 == pytest_approx(m2, abs=1e-12)


def test_duplicate_sample_invariant_binning(dev_csv_bytes):
    """样本原样复制一份：分箱、WOE 不变。"""
    s = parse_csv(dev_csv_bytes)
    doubled_text = dev_csv_bytes.decode().strip() + "\n"
    doubled_text += "\n".join(
        dev_csv_bytes.decode().strip().splitlines()[1:]) + "\n"
    s2 = parse_csv(doubled_text.encode())
    assert s2.n == 2 * s.n
    for name in ("age", "income", "city", "job_grade"):
        f1 = bin_feature(name, s.types[name], s.features[name], s.label)
        f2 = bin_feature(name, s2.types[name], s2.features[name], s2.label)
        assert len(f1.bins) == len(f2.bins)
        for b1, b2 in zip(f1.bins, f2.bins):
            assert b1.label == b2.label
            assert b1.woe == pytest_approx(b2.woe, abs=1e-12)
            assert b1.bad_rate == pytest_approx(b2.bad_rate, abs=1e-12)
        assert f1.iv == pytest_approx(f2.iv, abs=1e-12)
