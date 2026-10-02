"""测试辅助：把开发样本 CSV 行转成打分接口接受的 JSON 特征。

打分接口的数值特征收 JSON 数字（字符串数字按非数值拒绝，是既有行为），
因此这里必须把数值列从字符串转 float / 空串转 None。
"""
from __future__ import annotations

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_dev_rows():
    with open(os.path.join(DATA_DIR, "dev_sample.csv"), newline="") as f:
        return list(csv.DictReader(f))


NUMERIC = {"age", "income", "debt_ratio"}
CATEGORICAL = {"city", "job_grade"}


def row_to_features(row: dict) -> dict:
    out = {}
    for name in NUMERIC:
        token = (row.get(name) or "").strip()
        out[name] = None if token == "" else float(token)
    for name in CATEGORICAL:
        token = (row.get(name) or "").strip()
        out[name] = None if token == "" else token
    return out


def score_training_sample(client, card: str, version: int = 1,
                          id_prefix: str = "train") -> list[dict]:
    """把建卡样本原样逐条打分（带请求标识），返回行字典列表。"""
    rows = load_dev_rows()
    for i, row in enumerate(rows):
        resp = client.post(
            f"/cards/{card}/score",
            json={"version": version, "features": row_to_features(row),
                  "request_id": f"{id_prefix}-{i}"},
        )
        assert resp.status_code == 200, resp.text
    return rows


def wait_backfill(client, job_id: int):
    import time
    for _ in range(500):
        j = client.get(f"/performance/backfills/{job_id}").json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.01)
    raise AssertionError("回填作业长时间未结束")
