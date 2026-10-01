"""样本解析与类型识别。

- CSV 解析使用标准库 csv（RFC4180 风格），不依赖任何机器学习库。
- 标签列必须是 0/1；缺失记号（空串、NA、N/A、null、None、NaN 等）统一识别为缺失。
- 特征类型按数据推断：所有非缺失值都能解析为 float 即视为数值型，否则类别型。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np

from .exceptions import ValidationError

MISSING_TOKENS = frozenset({"", "na", "n/a", "null", "none", "nan", "nat"})


@dataclass
class Sample:
    label: np.ndarray                     # shape (n,) float64，取值 0/1
    features: dict[str, np.ndarray] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.label)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.keys())


def _read_rows(content: bytes | str) -> tuple[list[str], list[list[str]]]:
    if isinstance(content, bytes):
        text = content.decode("utf-8-sig")
    else:
        text = content
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if row and not all(c.strip() == "" for c in row)]
    if len(rows) < 2:
        raise ValidationError("CSV 至少需要表头加一行数据")
    header = [h.strip() for h in rows[0]]
    if len(set(header)) != len(header):
        raise ValidationError("CSV 表头存在重复列名")
    body = rows[1:]
    for i, row in enumerate(body):
        if len(row) != len(header):
            raise ValidationError(
                f"第 {i + 2} 行列数为 {len(row)}，与表头 {len(header)} 不一致"
            )
    return header, body


def _is_numeric_column(values: list[str]) -> bool:
    saw_value = False
    for raw in values:
        token = raw.strip()
        if token.lower() in MISSING_TOKENS:
            continue
        try:
            float(token)
        except ValueError:
            return False
        saw_value = True
    return saw_value


def parse_csv(content: bytes | str, label_col: str = "label") -> Sample:
    header, body = _read_rows(content)
    if label_col not in header:
        raise ValidationError(f"标签列 {label_col!r} 不存在，实际列：{header}")
    label_idx = header.index(label_col)

    n = len(body)
    label = np.zeros(n, dtype=np.float64)
    for i, row in enumerate(body):
        token = row[label_idx].strip()
        if token.lower() in MISSING_TOKENS:
            raise ValidationError(f"第 {i + 2} 行标签缺失，标签不允许缺失")
        if token not in ("0", "1"):
            raise ValidationError(
                f"第 {i + 2} 行标签为 {token!r}，标签必须是 0/1"
            )
        label[i] = float(token)

    features: dict[str, np.ndarray] = {}
    types: dict[str, str] = {}
    for j, name in enumerate(header):
        if j == label_idx:
            continue
        raw_col = [row[j] for row in body]
        if _is_numeric_column(raw_col):
            arr = np.full(n, np.nan, dtype=np.float64)
            for i, raw in enumerate(raw_col):
                token = raw.strip()
                if token.lower() not in MISSING_TOKENS:
                    arr[i] = float(token)
            features[name] = arr
            types[name] = "numeric"
        else:
            arr = np.array(
                [None if r.strip().lower() in MISSING_TOKENS else r.strip()
                 for r in raw_col],
                dtype=object,
            )
            features[name] = arr
            types[name] = "categorical"

    return Sample(label=label, features=features, types=types)
