"""HTTP 接口测试：建卡作业、版本、打分、批量、校验、并发隔离。"""
from __future__ import annotations

import time

from tests.conftest import build_and_wait, latest_artifact


CARD = "card_a"


def _score(client, card, features, version=None):
    body = {"features": features}
    if version is not None:
        body["version"] = version
    return client.post(f"/cards/{card}/score", json=body)


# ------------------------------------------------------------ 建卡主流程

def test_build_job_lifecycle_and_version(client, dev_csv_bytes):
    job = build_and_wait(client, CARD, dev_csv_bytes)
    assert job["status"] == "succeeded"
    assert job["version"] == 1
    cards = client.get("/cards").json()
    assert {"card_name": CARD, "latest_version": 1, "version_count": 1} in cards

    art = latest_artifact(client, CARD)
    assert art["metrics"]["ks"] > 0.3
    assert art["metrics"]["auc"] > 0.6
    # 中间结果齐全
    assert art["regression"]["history"]
    for f in art["features"]:
        assert f["journal"]
        for b in f["bins"]:
            assert {"lo", "hi", "bad", "good", "woe", "iv_contrib"} <= set(b) \
                or {"categories", "bad", "good", "woe", "iv_contrib"} <= set(b)


def test_same_card_multiple_versions(client, dev_csv_bytes):
    j1 = build_and_wait(client, CARD, dev_csv_bytes)
    j2 = build_and_wait(client, CARD, dev_csv_bytes)
    j3 = build_and_wait(client, CARD, dev_csv_bytes)
    assert (j1["version"], j2["version"], j3["version"]) == (1, 2, 3)
    versions = client.get(f"/cards/{CARD}/versions").json()
    assert [v["version"] for v in versions] == [1, 2, 3]


def test_explicit_feature_list(client, dev_csv_bytes):
    job = build_and_wait(client, CARD, dev_csv_bytes, features="age,city")
    assert job["status"] == "succeeded"
    art = latest_artifact(client, CARD)
    assert art["selected_features"] == ["age", "city"]


def test_iv_auto_filter_excludes_noise(client, dev_csv_bytes):
    job = build_and_wait(client, CARD, dev_csv_bytes)
    art = latest_artifact(client, CARD)
    # debt_ratio 是确定性噪声，IV 极低，不应入选
    assert "debt_ratio" not in art["selected_features"]
    assert "age" in art["selected_features"]


# ------------------------------------------------------------ 打分

def test_single_scoring_fields(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    r = _score(client, CARD, {"age": 40, "income": 9000,
                              "city": "BJ", "job_grade": "A"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 1
    assert 0 < body["pd"] < 1
    assert abs(body["total_score"]
               - sum(f["score"] for f in body["features"])) < 1e-9
    names = {f["name"] for f in body["features"]}
    assert {"age", "city", "job_grade", "income"} <= names


def test_unseen_category_flagged_not_silently_binned(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    r = _score(client, CARD, {"age": 40, "income": 9000,
                              "city": "NEVER_SEEN", "job_grade": "A"})
    body = r.json()
    assert body["has_unseen"] is True
    city = next(f for f in body["features"] if f["name"] == "city")
    assert city["status"] == "unseen"
    assert city["bin_label"] is None
    assert "未见" in city["detail"]


def test_missing_lands_on_missing_bin(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    r = _score(client, CARD, {"age": None, "income": 9000,
                              "city": "BJ", "job_grade": "A"})
    body = r.json()
    age = next(f for f in body["features"] if f["name"] == "age")
    assert age["status"] == "missing"
    assert age["bin_label"] == "缺失"
    assert body["has_missing"] is True


def test_score_default_latest_version(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes,
                   base_score=600, base_odds=20, pdo=50)
    build_and_wait(client, CARD, dev_csv_bytes,
                   base_score=650, base_odds=10, pdo=25)
    r = _score(client, CARD, {"age": 40, "income": 9000,
                              "city": "BJ", "job_grade": "A"})
    assert r.json()["version"] == 2
    r1 = _score(client, CARD, {"age": 40, "income": 9000,
                               "city": "BJ", "job_grade": "A"}, version=1)
    assert r1.json()["version"] == 1


def test_batch_isolation(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    req = {
        "card_name": CARD,
        "applicants": [
            {"applicant_id": "ok1",
             "features": {"age": 40, "income": 9000, "city": "BJ",
                          "job_grade": "A"}},
            {"applicant_id": "bad_type",
             "features": {"age": "not_a_number", "income": 9000,
                          "city": "BJ", "job_grade": "A"}},
            {"applicant_id": "ok2",
             "features": {"age": 25, "income": 4000, "city": "CD",
                          "job_grade": "E"}},
        ],
    }
    r = client.post(f"/cards/{CARD}/score/batch", json=req)
    body = r.json()
    assert body["succeeded"] == 2 and body["failed"] == 1
    by_id = {x["applicant_id"]: x for x in body["results"]}
    assert by_id["ok1"]["ok"] is True and by_id["ok1"]["pd"] is not None
    assert by_id["ok2"]["ok"] is True
    assert by_id["bad_type"]["ok"] is False
    assert by_id["bad_type"]["error"]  # 错误原因保留，且不影响其他条目


# ------------------------------------------------------------ 作业前校验

def test_reject_fewer_than_500(client):
    header = "label,x\n"
    body = (header + "".join(f"{i % 2},{i}\n" for i in range(499))).encode()
    r = client.post(f"/cards/{CARD}/build",
                    files={"file": ("s.csv", body, "text/csv")})
    assert r.status_code == 400
    assert "500" in r.json()["detail"]
    assert client.get("/jobs").json() == []  # 不产生作业


def test_reject_non_binary_label(client):
    body = ("label,x\n" + "".join(f"{i % 3},{i}\n" for i in range(600))).encode()
    r = client.post(f"/cards/{CARD}/build",
                    files={"file": ("s.csv", body, "text/csv")})
    assert r.status_code == 400 and "0/1" in r.json()["detail"]


def test_reject_single_class_label(client):
    body = ("label,x\n" + "".join(f"0,{i}\n" for i in range(600))).encode()
    r = client.post(f"/cards/{CARD}/build",
                    files={"file": ("s.csv", body, "text/csv")})
    assert r.status_code == 400 and "同一类" in r.json()["detail"]


def test_reject_nonpositive_pdo(client, dev_csv_bytes):
    r = client.post(f"/cards/{CARD}/build",
                    files={"file": ("s.csv", dev_csv_bytes, "text/csv")},
                    data={"pdo": "0"})
    assert r.status_code == 400 and "PDO" in r.json()["detail"]


def test_reject_missing_model_feature(client, dev_csv_bytes):
    r = client.post(f"/cards/{CARD}/build",
                    files={"file": ("s.csv", dev_csv_bytes, "text/csv")},
                    data={"features": "age,no_such_column"})
    assert r.status_code == 400 and "不存在" in r.json()["detail"]


def test_score_unknown_card_404(client):
    r = _score(client, "nope", {"x": 1})
    assert r.status_code == 404


# ------------------------------------------------------------ 并发隔离

def test_concurrent_builds_do_not_mix(client, dev_csv_bytes):
    # 同名卡 4 个并发建卡：版本号必须恰好 1..4，且全部成功
    files = {"file": ("s.csv", dev_csv_bytes, "text/csv")}
    ids = []
    for _ in range(4):
        r = client.post(f"/cards/{CARD}/build", files=files)
        ids.append(r.json()["job_id"])
    finals = []
    for _ in range(500):
        finals = [client.get(f"/jobs/{j}").json() for j in ids]
        if all(j["status"] in ("succeeded", "failed") for j in finals):
            break
        time.sleep(0.02)
    assert all(j["status"] == "succeeded" for j in finals), finals
    assert sorted(j["version"] for j in finals) == [1, 2, 3, 4]

    # 各版本产物都能独立加载，且参数与训练指标自洽
    for v in (1, 2, 3, 4):
        art = client.get(f"/cards/{CARD}/versions/{v}/artifact").json()
        assert art["validation"]["passed"]
        assert abs(art["validation"]["avg_predicted_pd"]
                   - art["data_summary"]["bad_rate"]) < 1e-6


def test_concurrent_different_cards_independent(client, dev_csv_bytes):
    files = {"file": ("s.csv", dev_csv_bytes, "text/csv")}
    resp = [client.post(f"/cards/card_{i}/build", files=files)
            for i in range(3)]
    ids = [(r.json()["card_name"], r.json()["job_id"]) for r in resp]
    for _ in range(500):
        st = [client.get(f"/jobs/{j}").json()["status"] for _, j in ids]
        if all(s in ("succeeded", "failed") for s in st):
            break
        time.sleep(0.02)
    for name, j in ids:
        job = client.get(f"/jobs/{j}").json()
        assert job["status"] == "succeeded", job
        assert job["version"] == 1
        assert client.get(f"/cards/{name}/versions/1").status_code == 200
