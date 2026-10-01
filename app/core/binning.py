"""分箱引擎（全部自研，仅用 numpy 做计数/排序）。

数值特征
--------
1. 不同取值数 <= max_bins 时每个取值一个初始箱；否则按等频分位点切出不超过
   max_b1ins 个非空初始箱（打结值不拆开）。
2. 相邻合并分三阶段，每阶段合并后重新计算好坏计数与 WOE：
   a. 非零：任一箱坏或好为 0 时，与其坏样率最接近的相邻箱合并；
   b. 占比：样本占比 < min_bin_pct 的箱中取最小箱，与坏样率最接近的相邻箱合并；
   c. 单调：方向由数据决定（WOE 与特征值的加权协方差符号），沿该方向把
      逆序的相邻箱中 WOE 落差最小的一对合并，直到严格单调（容差 1e-12）。
合并保持相邻性与覆盖完整性，因此不会出现空箱，且 b、c 阶段不会破坏前面的约束。

类别特征
--------
按坏样率升序排列（坏样率相同按类别名），WOE 是坏样率 r 的严格增函数
(log(r/(1-r)) + const)，故排序后天然单调；同样执行非零与占比合并，
每次合并后重新排序。

缺失值
------
始终单独成一箱，不参与任何合并。训练集该特征无缺失时该箱计数为 0、WOE 记 0。
合并阶段某箱出现 0 好或 0 坏（极端情形的兜底）时，WOE/IV 用 0.5 对称平滑，
保证标签整体取反时 WOE 变号、IV 不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MONO_TOL = 1e-12


# ---------------------------------------------------------------- 数据结构

@dataclass
class Bin:
    index: int
    total: int
    bad: int
    good: int
    bad_rate: float
    woe: float
    iv_contrib: float
    lo: float | None = None          # 数值箱左边界（闭）
    hi: float | None = None          # 数值箱右边界（最后一箱为闭，其余开）
    right_closed: bool = False       # 是否最后一个数值箱（右端闭）
    categories: tuple[str, ...] | None = None
    is_missing: bool = False
    label: str = ""


@dataclass
class _Group:
    """合并过程中的临时箱。"""
    bad: int
    good: int
    lo: float | None = None
    hi: float | None = None
    cats: tuple[str, ...] | None = None

    @property
    def total(self) -> int:
        return self.bad + self.good

    @property
    def bad_rate(self) -> float:
        n = self.total
        return self.bad / n if n else 0.0

    def key(self) -> tuple[float, ...] | tuple[str, ...]:
        if self.cats is not None:
            return tuple(sorted(self.cats))
        return (self.lo,)


def woe_iv(bad: int, good: int, B: int, G: int) -> tuple[float, float]:
    """单箱 WOE 与 IV 贡献。

    空箱：WOE=0、IV=0。非空但单边为 0：对零的一侧补 0.5（对称平滑，
    标签取反后 WOE 恰好变号、IV 不变）。
    """
    n = bad + good
    if n == 0:
        return 0.0, 0.0
    b, g = float(bad), float(good)
    if good == 0 or bad == 0:
        b += 0.5
        g += 0.5
    w = np.log((b / B) / (g / G))
    iv = (b / B - g / G) * w
    return float(w), float(iv)


@dataclass
class FeatureBinning:
    name: str
    ftype: str                      # "numeric" | "categorical"
    bins: list[Bin] = field(default_factory=list)
    missing_bin: Bin | None = None
    iv: float = 0.0
    direction: int = 0              # +1: WOE 随值递增；-1 递减；0 无区分度
    total_bad: int = 0
    total_good: int = 0
    journal: list[str] = field(default_factory=list)  # 分箱合并全过程日志

    def regular_bins_sorted(self) -> list[Bin]:
        return self.bins


# ---------------------------------------------------------------- 数值初箱

def _numeric_value_counts(v: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (取值, 各取值坏样本数, 各取值好样本数)。"""
    order = np.argsort(v, kind="mergesort")
    sv, sy = v[order], y[order]
    change = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    vals = sv[change]
    bad = np.add.reduceat(sy, change).astype(np.int64)
    counts = np.diff(np.r_[change, len(sv)])
    good = (counts - bad).astype(np.int64)
    return vals, bad, good


def numeric_initial_groups(v: np.ndarray, y: np.ndarray, max_bins: int) -> list[_Group]:
    vals, bad, good = _numeric_value_counts(v, y)
    n = len(v)

    if len(vals) <= max_bins:
        # 每个不同取值一箱
        groups: list[_Group] = []
        for i, val in enumerate(vals):
            lo = float(val)
            hi = float(vals[i + 1]) if i + 1 < len(vals) else float(val)
            groups.append(_Group(int(bad[i]), int(good[i]), lo=lo, hi=hi))
        if groups:
            groups[-1].hi = float(vals[-1])
        return groups

    # 等频切点：先对全部样本取分位点，再吸附到最近的实际取值，打结值不拆开。
    # counts[i] = 取值 vals[i] 的样本数；cum_before[i] = 严格小于 vals[i] 的样本数。
    counts = (bad + good).astype(np.int64)
    cum_before = np.zeros(len(vals), dtype=np.int64)
    np.cumsum(counts[:-1], out=cum_before[1:])
    qs = np.linspace(0, 1, max_bins + 1)[1:-1]
    cuts: list[float] = []
    for q in qs:
        target = q * n
        # 最小取值索引，使严格小于该取值的样本数达到 target
        idx = int(np.searchsorted(cum_before, target, side="left"))
        idx = min(max(idx, 1), len(vals) - 1)
        cut = float(vals[idx])
        if not cuts or cut > cuts[-1]:
            cuts.append(cut)
    edges = [float(vals[0])] + cuts + [float(vals[-1])]

    # 按边界聚合各取值的计数
    groups = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if k < len(edges) - 2:
            mask = (v >= lo) & (v < hi)
        else:
            mask = (v >= lo) & (v <= hi)
        b = int(y[mask].sum())
        groups.append(_Group(b, int(mask.sum() - b), lo=lo, hi=hi))
    groups = [g for g in groups if g.total > 0]
    groups[-1].hi = float(vals[-1])
    return groups


# ---------------------------------------------------------------- 类别初箱

def categorical_initial_groups(v: np.ndarray, y: np.ndarray) -> list[_Group]:
    levels: dict[str, list[int]] = {}
    for cat, label in zip(v, y):
        rec = levels.setdefault(cat, [0, 0])
        rec[int(label)] += 1
    groups = [
        _Group(bad=rec[1], good=rec[0], cats=(cat,))
        for cat, rec in levels.items()
    ]
    groups.sort(key=lambda g: (g.bad_rate, _cat_key(g)))
    return groups


def _cat_key(g: _Group) -> tuple[str, ...]:
    return tuple(sorted(g.cats))  # type: ignore[arg-type]


# ---------------------------------------------------------------- 合并原语

def _merge(a: _Group, b: _Group) -> _Group:
    if a.cats is not None:
        return _Group(a.bad + b.bad, a.good + b.good,
                      cats=tuple(sorted(set(a.cats) | set(b.cats))))  # type: ignore[arg-type]
    return _Group(a.bad + b.bad, a.good + b.good, lo=a.lo, hi=b.hi)


def _group_woe(g: _Group, B: int, G: int) -> float:
    w, _ = woe_iv(g.bad, g.good, B, G)
    return w


def _merge_positivity_and_size(
    groups: list[_Group],
    n_total: int,
    min_bin_pct: float,
    categorical: bool,
    journal: list[str],
) -> list[_Group]:
    """非零与占比两阶段合并。数值组保持值序；类别组始终按坏样率排序。"""

    def resort() -> None:
        if categorical:
            groups.sort(key=lambda g: (g.bad_rate, _cat_key(g)))

    def desc(i: int) -> str:
        g = groups[i]
        if categorical:
            return "{" + ",".join(_cat_key(g)) + "}"
        return f"[{g.lo:g},{g.hi:g}]"

    resort()

    # 阶段 a：消除 0 好 / 0 坏
    while any(g.bad == 0 or g.good == 0 for g in groups):
        if len(groups) == 1:
            break
        best: tuple[float, object, int] | None = None
        for i, g in enumerate(groups):
            if g.bad != 0 and g.good != 0:
                continue
            for j in (i - 1, i + 1):
                if not (0 <= j < len(groups)):
                    continue
                other = groups[j]
                dist = abs(g.bad_rate - other.bad_rate)
                canon = _canonical(groups, min(i, j), categorical)
                cand = (dist, canon, min(i, j))
                if best is None or cand[:2] < best[:2]:
                    best = cand
        assert best is not None
        i = best[2]
        journal.append(
            f"非零合并：{desc(i)} + {desc(i + 1)}（坏样率最接近）"
        )
        groups[i] = _merge(groups[i], groups[i + 1])
        del groups[i + 1]
        resort()

    # 阶段 b：占比下限
    while groups and min(g.total for g in groups) / n_total < min_bin_pct:
        if len(groups) == 1:
            break
        sizes = [g.total for g in groups]
        i = min(range(len(groups)), key=lambda k: (sizes[k], k))
        g = groups[i]
        best_pair: tuple[float, object, int] | None = None
        for j in (i - 1, i + 1):
            if not (0 <= j < len(groups)):
                continue
            dist = abs(g.bad_rate - groups[j].bad_rate)
            canon = _canonical(groups, min(i, j), categorical)
            cand = (dist, canon, min(i, j))
            if best_pair is None or cand[:2] < best_pair[:2]:
                best_pair = cand
        assert best_pair is not None
        k = best_pair[2]
        journal.append(
            f"占比合并：{desc(k)} + {desc(k + 1)}（最小箱占比 "
            f"{groups[k].total}/{n_total} < {min_bin_pct:g}）"
        )
        groups[k] = _merge(groups[k], groups[k + 1])
        del groups[k + 1]
        resort()

    return groups


def _canonical(groups: list[_Group], i: int, categorical: bool):
    """合并对的确定性次序键：类别用合并后类别集合的字典序，数值用左边界。"""
    merged = _merge(groups[i], groups[i + 1])
    if categorical:
        return _cat_key(merged)
    return merged.lo


def _numeric_direction(groups: list[_Group], B: int, G: int) -> int:
    ws = np.array([_group_woe(g, B, G) for g in groups])
    mids = np.array([(g.lo + (g.hi if g.hi is not None else g.lo)) / 2.0 for g in groups])
    ns = np.array([g.total for g in groups], dtype=float)
    wm = np.average(ws, weights=ns)
    mm = np.average(mids, weights=ns)
    cov = np.sum(ns * (mids - mm) * (ws - wm))
    if cov > MONO_TOL:
        return 1
    if cov < -MONO_TOL:
        return -1
    return 0


def _merge_monotonic_numeric(groups: list[_Group], B: int, G: int,
                             direction: int, journal: list[str]) -> list[_Group]:
    """沿 direction 合并逆序相邻箱，直到 WOE 序列在该方向上非严格单调
    （严格好箱之间允许相等；合并以逆序落差最小为优先，保证确定性）。"""
    if direction == 0:
        return groups
    while len(groups) > 1:
        ws = [_group_woe(g, B, G) for g in groups]
        viol: list[tuple[float, float, int]] = []
        for i in range(len(groups) - 1):
            delta = ws[i + 1] - ws[i]
            if direction == 1 and delta < -MONO_TOL:
                viol.append((abs(delta), groups[i].lo, i))
            elif direction == -1 and delta > MONO_TOL:
                viol.append((abs(delta), groups[i].lo, i))
        if not viol:
            break
        viol.sort()
        i = viol[0][2]
        a, b = groups[i], groups[i + 1]
        journal.append(
            f"单调合并(direction={direction:+d})：[{a.lo:g},{a.hi:g}] + "
            f"[{b.lo:g},{b.hi:g}]（WOE 逆序落差 {ws[i + 1] - ws[i]:+.4f} 最小）"
        )
        groups[i] = _merge(groups[i], groups[i + 1])
        del groups[i + 1]
    return groups


# ---------------------------------------------------------------- 对外入口

def _missing_mask(v: np.ndarray, ftype: str) -> np.ndarray:
    if ftype == "numeric":
        return np.isnan(v)
    return np.array([x is None for x in v], dtype=bool)


def bin_feature(
    name: str,
    ftype: str,
    values: np.ndarray,
    label: np.ndarray,
    max_bins: int = 20,
    min_bin_pct: float = 0.05,
) -> FeatureBinning:
    miss = _missing_mask(values, ftype)
    present_v = values[~miss]
    present_y = label[~miss]
    missing_y = label[miss]

    B = int(label.sum())
    G = int(len(label) - B)
    journal: list[str] = []

    if len(present_v) == 0:
        # 该特征训练样本全缺失：没有常规箱，只有缺失箱
        groups = []
        direction = 0
        journal.append("训练样本该特征全部缺失：无常规箱，仅缺失箱")
    elif ftype == "numeric":
        groups = numeric_initial_groups(present_v, present_y, max_bins)
        journal.append(f"等频初始箱：{len(groups)} 个（上限 {max_bins}）")
        groups = _merge_positivity_and_size(
            groups, len(present_v), min_bin_pct, False, journal)
        direction = _numeric_direction(groups, B, G) if groups else 0
        groups = _merge_monotonic_numeric(groups, B, G, direction, journal)
        groups.sort(key=lambda g: g.lo)
    else:
        groups = categorical_initial_groups(present_v, present_y)
        journal.append(f"按坏样率排序初始箱：{len(groups)} 个类别")
        groups = _merge_positivity_and_size(
            groups, len(present_v), min_bin_pct, True, journal)
        direction = 1 if groups else 0
    journal.append(f"最终常规箱：{len(groups)} 个，方向 direction={direction:+d}")

    bins: list[Bin] = []
    iv_total = 0.0
    for idx, g in enumerate(groups):
        w, iv = woe_iv(g.bad, g.good, B, G)
        iv_total += iv
        if ftype == "numeric":
            last = idx == len(groups) - 1
            lab = f"[{g.lo:g}, {g.hi:g}{']' if last else ')'}"
            bins.append(Bin(
                index=idx, total=g.total, bad=g.bad, good=g.good,
                bad_rate=g.bad_rate, woe=w, iv_contrib=iv,
                lo=g.lo, hi=g.hi, right_closed=last, label=lab,
            ))
        else:
            cats = _cat_key(g)
            bins.append(Bin(
                index=idx, total=g.total, bad=g.bad, good=g.good,
                bad_rate=g.bad_rate, woe=w, iv_contrib=iv,
                categories=cats, label=" | ".join(cats),
            ))

    m_bad = int(missing_y.sum())
    m_good = int(len(missing_y) - m_bad)
    if m_bad + m_good == 0:
        m_woe, m_iv = 0.0, 0.0
    else:
        m_woe, m_iv = woe_iv(m_bad, m_good, B, G)
    iv_total += m_iv
    missing_bin = Bin(
        index=-1, total=m_bad + m_good, bad=m_bad, good=m_good,
        bad_rate=(m_bad / (m_bad + m_good)) if (m_bad + m_good) else 0.0,
        woe=m_woe, iv_contrib=m_iv, is_missing=True, label="缺失",
    )

    return FeatureBinning(
        name=name, ftype=ftype, bins=bins, missing_bin=missing_bin,
        iv=float(iv_total), direction=direction, total_bad=B, total_good=G,
        journal=journal,
    )


def bin_all_features(
    sample,
    feature_names: list[str],
    max_bins: int = 20,
    min_bin_pct: float = 0.05,
) -> dict[str, FeatureBinning]:
    out = {}
    for name in feature_names:
        out[name] = bin_feature(
            name, sample.types[name], sample.features[name], sample.label,
            max_bins=max_bins, min_bin_pct=min_bin_pct,
        )
    return out


# ---------------------------------------------------------------- 落箱

def assign_bin(fb: FeatureBinning, value) -> Bin:
    """把一个原始特征值落到训练得到的箱。调用方负责先判缺失/未见。

    数值超出训练范围时夹到首/末箱（线性外推的常见做法）。
    """
    if fb.ftype == "numeric":
        x = float(value)
        reg = fb.bins
        if not reg:
            # 训练时该特征全缺失：任何非缺失值都无法落箱
            raise KeyError(str(value))
        if x <= reg[0].lo:
            return reg[0]
        for b in reg:
            if b.right_closed:
                if b.lo <= x <= b.hi:
                    return b
            elif b.lo <= x < b.hi:
                return b
        return reg[-1]

    sval = str(value)
    if not fb.bins:
        raise KeyError(sval)
    for b in fb.bins:
        if sval in b.categories:  # type: ignore[operator]
            return b
    raise KeyError(sval)
