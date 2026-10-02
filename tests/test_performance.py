"""表现回填与投产后区分能力测试。

覆盖：
- 回填后台作业可查进度与结果；not_found / 非法标签进拒收清单，不影响其它条目；
- 同标识同标签回填两次只算一遍（duplicated），不同标签拒收且首次不被覆盖；
- 实际坏率、平均预测 PD、KS、AUC 与把同一批分数标签直接交给建卡用的
  app.core.metrics 算出的结果逐位一致；
- 总分分段的计数/坏率、单类标签时 KS/AUC 为 null；
- 幂等打分在回填时仍只算一次（先有唯一记录才能关联标签）。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.metrics import ks_stat, roc_auc
from tests.conftest import build_and_wait
from tests.helpers import (load_dev_rows, row_to_features, score_training_sample,
                           wait_backfill)

CARD = "card_perf"


def _backfill(client, items):
    r = client.post(f"/cards/{CARD}/performance/backfill", json={"items": items})
    assert r.status_code == 202, r.text
    ack = r.json()
    return ack, wait_backfill(client, ack["job_id"])


def _performance(client, **params):
    return client.get(f"/cards/{CARD}/versions/1/performance",
                      params=params).json()


def _score_and_label_training(client, flip: dict | None = None):
    rows = score_training_sample(client, CARD)
    labels = {f"train-{i}": int(r["label"]) for i, r in enumerate(rows)}
    if flip:
        for rid, lab in flip.items():
            labels[rid] = lab
    return rows, labels


# ------------------------------------------------------------ 回填主流程

def test_backfill_happy_path_and_rejections(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    rows = score_training_sample(client, CARD)
    items = [{"request_id": f"train-{i}", "label": int(rows[i]["label"])}
             for i in range(50)]
    # 混入：找不到的标识、非法标签、布尔（不是 0/1）、空标识
    items += [
        {"request_id": "ghost", "label": 1},
        {"request_id": "train-0", "label": 2},
        {"request_id": "train-1", "label": "1"},
        {"request_id": "train-2", "label": True},
        {"request_id": "   ", "label": 0},
    ]
    ack, job = _backfill(client, items)
    assert job["status"] == "succeeded"
    assert job["applied"] == 50
    # not_found 是作业期判定，落在作业拒收清单
    assert [x["request_id"] for x in job["rejected"]] == ["ghost"]
    assert job["rejected"][0]["reason"] == "not_found"
    # 非法标签/标识在入队前剔除，随提交响应返回，不进作业
    invalid = {x["request_id"]: x["reason"] for x in ack["invalid_rejected_items"]}
    assert invalid == {"train-0": "invalid_label",
                       "train-1": "invalid_label",
                       "train-2": "invalid_label",
                       "   ": "invalid_request_id"}
    assert ack["invalid_rejected"] == 4
    assert ack["total"] == 51  # 50 合法 + ghost（作业期才知道不存在）
    # 作业可列出
    listed = client.get("/performance/backfills", params={"card_name": CARD}).json()
    assert any(j["id"] == job["id"] for j in listed)


def test_duplicate_same_label_idempotent(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    score_training_sample(client, CARD)
    items = [{"request_id": "train-0", "label": 1}]
    ack1, j1 = _backfill(client, items)
    ack2, j2 = _backfill(client, items)
    assert j1["applied"] == 1
    assert j2["applied"] == 0 and j2["duplicated"] == 1
    perf = _performance(client)
    assert perf["n_performed"] == 1  # 只算一遍


def test_conflicting_label_rejected_and_first_kept(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    score_training_sample(client, CARD)
    first_label = int(load_dev_rows()[0]["label"])
    other = 1 - first_label
    _, j1 = _backfill(client, [{"request_id": "train-0", "label": first_label}])
    assert j1["applied"] == 1
    _, j2 = _backfill(client, [{"request_id": "train-0", "label": other}])
    assert j2["applied"] == 0 and j2["duplicated"] == 0
    rej = j2["rejected"][0]
    assert rej["reason"] == "label_conflict"
    assert rej["existing_label"] == first_label
    # 首次标签仍然有效
    perf = _performance(client)
    assert perf["bad_count"] == first_label


def test_duplicates_within_one_batch(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    score_training_sample(client, CARD)
    lab = int(load_dev_rows()[3]["label"])
    # 同一批次内同一标识出现三次：第一次落库，后两次算 duplicated
    _, job = _backfill(client, [
        {"request_id": "train-3", "label": lab},
        {"request_id": "train-3", "label": lab},
        {"request_id": "train-3", "label": lab},
    ])
    assert job["applied"] == 1 and job["duplicated"] == 2
    assert _performance(client)["n_performed"] == 1


# ------------------------------------------------------------ 指标口径一致

def test_ks_auc_match_build_metrics_exactly(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    art = client.get(f"/cards/{CARD}/versions/1/artifact").json()
    rows = score_training_sample(client, CARD)
    items = [{"request_id": f"train-{i}", "label": int(rows[i]["label"])}
             for i in range(len(rows))]
    _, job = _backfill(client, items)
    assert job["applied"] == len(rows)

    perf = _performance(client)
    assert perf["n_performed"] == len(rows)
    assert perf["actual_bad_rate"] == pytest.approx(
        art["data_summary"]["bad_rate"])
    # 平均预测 PD = 训练违约率（带截距逻辑回归性质）
    assert perf["avg_predicted_pd"] == pytest.approx(
        art["data_summary"]["bad_rate"], abs=1e-9)

    # 把同一批分数与标签直接交给建卡时的评估函数
    logs = client.get(f"/cards/{CARD}/versions/1/score-logs",
                      params={"limit": 5000}).json()["logs"]
    by_id = {l["request_id"]: l for l in logs}
    y = np.array([by_id[f"train-{i}"]["label"] for i in range(len(rows))], float)
    p = np.array([by_id[f"train-{i}"]["pd"] for i in range(len(rows))], float)
    assert perf["ks"] == ks_stat(y, p)
    assert perf["auc"] == roc_auc(y, p)
    # 与建卡产物上的训练指标一致（同一批数据、同一套模型输出）
    assert perf["ks"] == pytest.approx(art["metrics"]["ks"])
    assert perf["auc"] == pytest.approx(art["metrics"]["auc"])


def test_score_bands_counts_and_rates(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    rows = score_training_sample(client, CARD)
    items = [{"request_id": f"train-{i}", "label": int(rows[i]["label"])}
             for i in range(len(rows))]
    _backfill(client, items)
    perf = _performance(client)
    bands = perf["score_bands"]
    assert sum(b["count"] for b in bands) == len(rows)
    assert sum(b["bad_count"] for b in bands) == perf["bad_count"]
    for b in bands:
        assert b["count"] > 0
        assert 0 <= b["actual_bad_rate"] <= 1
        assert 0 < b["avg_predicted_pd"] < 1
    # 分段按总分有序（等频箱 lo 单调）
    los = [b["lo"] for b in bands]
    assert los == sorted(los)


def test_single_class_labels_yield_null_ks_auc(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    score_training_sample(client, CARD)
    items = [{"request_id": f"train-{i}", "label": 0} for i in range(30)]
    _backfill(client, items)
    perf = _performance(client)
    assert perf["bad_count"] == 0 and perf["good_count"] == 30
    assert perf["ks"] is None and perf["auc"] is None
    assert "KS/AUC" in perf["note"]


def test_empty_performance_window(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    score_training_sample(client, CARD)
    perf = _performance(client)
    assert perf["n_performed"] == 0
    assert perf["ks"] is None and perf["score_bands"] == []


def test_version_isolation_in_performance(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes,
                   base_score=600, base_odds=20, pdo=50)
    build_and_wait(client, CARD, dev_csv_bytes,
                   base_score=650, base_odds=10, pdo=25)
    rows = load_dev_rows()
    f0 = row_to_features(rows[0])
    client.post(f"/cards/{CARD}/score",
                json={"version": 1, "features": f0, "request_id": "v1-0"})
    client.post(f"/cards/{CARD}/score",
                json={"version": 2, "features": f0, "request_id": "v2-0"})
    _backfill(client, [{"request_id": "v1-0", "label": int(rows[0]["label"])}])
    p1 = client.get(f"/cards/{CARD}/versions/1/performance").json()
    p2 = client.get(f"/cards/{CARD}/versions/2/performance").json()
    assert p1["n_performed"] == 1 and p2["n_performed"] == 0
