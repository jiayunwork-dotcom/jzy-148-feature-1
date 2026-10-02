"""打分留痕与请求标识幂等。

覆盖：
- 每次成功打分（含批量成功条目）都留痕，失败条目不留痕；
- 同标识同内容重放返回完全相同结果且只算一次；
- 同标识不同内容明确拒绝、不覆盖；不带标识照常打分留痕；
- 打分失败的标识可修正后重试；批量逐条幂等；
- 打分响应原有字段与数值不变。
"""
from __future__ import annotations

import threading

from tests.conftest import build_and_wait
from tests.helpers import row_to_features, load_dev_rows

CARD = "card_audit"
FEAT = {"age": 40, "income": 9000, "city": "BJ", "job_grade": "A"}


def _logs(client, card=CARD, version=1, limit=10_000):
    return client.get(
        f"/cards/{card}/versions/{version}/score-logs",
        params={"limit": limit}).json()["logs"]


# ------------------------------------------------------------ 基本留痕

def test_every_successful_score_is_logged(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    r = client.post(f"/cards/{CARD}/score", json={"features": FEAT})
    assert r.status_code == 200
    body = r.json()
    # 原有字段一个不少
    assert set(body) == {
        "card_name", "version", "total_score", "pd", "features",
        "has_unseen", "has_missing"}
    logs = _logs(client)
    assert len(logs) == 1
    log = logs[0]
    assert log["version"] == 1 and log["request_id"] is None
    assert log["total_score"] == body["total_score"]
    assert log["pd"] == body["pd"]
    assert log["scored_at"]
    by_name = {f["name"]: f for f in log["feature_bins"]}
    for f in body["features"]:
        entry = by_name[f["name"]]
        assert entry["bin_label"] == f["bin_label"]
        assert entry["status"] == f["status"]


def test_batch_failures_are_not_logged(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    req = {"applicants": [
        {"applicant_id": "ok1", "features": FEAT},
        {"applicant_id": "bad",
         "features": {"age": "x", "income": 1, "city": "BJ", "job_grade": "A"}},
        {"applicant_id": "ok2",
         "features": {"age": 25, "income": 4000, "city": "CD", "job_grade": "E"}},
    ]}
    r = client.post(f"/cards/{CARD}/score/batch", json=req)
    body = r.json()
    assert body["succeeded"] == 2 and body["failed"] == 1
    assert body["replayed"] == 0 and body["conflicts"] == 0
    assert len(_logs(client)) == 2  # 失败那一条不进记录


# ------------------------------------------------------------ 幂等重放

def test_replay_same_id_same_content_counts_once(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    first = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "R1"}).json()
    for _ in range(100):
        rr = client.post(f"/cards/{CARD}/score",
                         json={"features": FEAT, "request_id": "R1"})
        assert rr.status_code == 200
        assert rr.json() == first  # 逐字段完全相同
    logs = _logs(client)
    assert len(logs) == 1
    assert logs[0]["request_id"] == "R1"


def test_replay_key_order_independent(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    first = client.post(f"/cards/{CARD}/score", json={
        "request_id": "R2",
        "features": {"age": 40, "city": "BJ", "income": 9000, "job_grade": "A"},
    }).json()
    again = client.post(f"/cards/{CARD}/score", json={
        "request_id": "R2",
        "features": {"job_grade": "A", "income": 9000, "age": 40, "city": "BJ"},
    }).json()
    assert again == first
    assert len(_logs(client)) == 1


def test_same_id_different_content_rejected_and_not_overwritten(
        client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    first = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "R3"}).json()
    conflict = client.post(f"/cards/{CARD}/score", json={
        "features": {**FEAT, "age": 41}, "request_id": "R3"})
    assert conflict.status_code == 409
    assert "不同内容" in conflict.json()["detail"]
    # 冲突不留痕，原记录仍在、内容不变
    logs = _logs(client)
    assert len(logs) == 1
    again = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "R3"}).json()
    assert again == first


def test_no_id_scores_and_logs_normally(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    for _ in range(3):
        r = client.post(f"/cards/{CARD}/score", json={"features": FEAT})
        assert r.status_code == 200
    assert len(_logs(client)) == 3  # 无标识：每次都留痕


def test_failed_score_releases_id_for_retry(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    bad = client.post(f"/cards/{CARD}/score", json={
        "features": {"age": "NaN-as-text", "income": 9000, "city": "BJ",
                     "job_grade": "A"},
        "request_id": "R4"})
    assert bad.status_code == 400
    assert len(_logs(client)) == 0
    # 修正内容后同标识可重试，且重试内容成为唯一记录
    ok = client.post(f"/cards/{CARD}/score",
                     json={"features": FEAT, "request_id": "R4"})
    assert ok.status_code == 200
    assert len(_logs(client)) == 1
    # 再拿错误内容来仍是冲突（已提交内容受保护）
    conflict = client.post(f"/cards/{CARD}/score", json={
        "features": {"age": 1, "income": 9000, "city": "BJ",
                     "job_grade": "A"},
        "request_id": "R4"})
    assert conflict.status_code == 409


def test_concurrent_replays_count_once(client, dev_csv_bytes):
    from app.deps import audit, repo, runtimes
    build_and_wait(client, CARD, dev_csv_bytes)
    ver, rt = runtimes.get(CARD, 1)
    errors: list[Exception] = []
    results: list = []

    def worker():
        try:
            res, replayed = audit.score_single(
                CARD, ver, rt, FEAT, "RR")
            results.append((res["total_score"], replayed))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    assert len({r[0] for r in results}) == 1
    assert sum(1 for _, rp in results if rp) == 99
    assert len(_logs(client)) == 1


# ------------------------------------------------------------ 批量幂等

def test_concurrent_different_content_conflicts(client, dev_csv_bytes):
    # 首个请求打分很慢时，同标识不同内容仍必须立即明确拒绝
    from app.audit.service import IdempotentConflict
    from app.deps import audit, repo, runtimes
    import time as time_mod
    build_and_wait(client, CARD, dev_csv_bytes)
    ver, rt = runtimes.get(CARD, 1)

    class SlowRuntime:
        def score(self, feats):
            time_mod.sleep(0.3)
            return rt.score(feats)

    outcome = []

    def first():
        try:
            audit.score_single(CARD, ver, SlowRuntime(), FEAT, "SLOW")
            outcome.append("first-ok")
        except Exception as exc:
            outcome.append(("first-err", exc))

    def second():
        time_mod.sleep(0.05)  # 确保第一个已登记 pending
        try:
            audit.score_single(CARD, ver, rt, {**FEAT, "age": 77}, "SLOW")
            outcome.append("second-ok")  # 不该走到这里
        except IdempotentConflict:
            outcome.append("second-conflict")
        except Exception as exc:
            outcome.append(("second-err", exc))

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert "first-ok" in outcome and "second-conflict" in outcome
    # 第一个的记录保留；再用第一个内容重放正常
    again = client.post(f"/cards/{CARD}/score",
                        json={"features": FEAT, "request_id": "SLOW"})
    assert again.status_code == 200


def test_batch_item_idempotency(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    body = {"applicants": [
        {"applicant_id": "a1", "features": FEAT, "request_id": "B1"},
        {"applicant_id": "a2",
         "features": {"age": 25, "income": 4000, "city": "CD",
                      "job_grade": "E"},
         "request_id": "B2"},
    ]}
    r1 = client.post(f"/cards/{CARD}/score/batch", json=body).json()
    assert len(_logs(client)) == 2
    r2 = client.post(f"/cards/{CARD}/score/batch", json=body).json()
    # 第二次全部重放，记录不增加；逐字段结果一致
    assert len(_logs(client)) == 2
    assert r2["succeeded"] == 2 and r2["replayed"] == 2
    by1 = {x["applicant_id"]: x for x in r1["results"]}
    by2 = {x["applicant_id"]: x for x in r2["results"]}
    for aid in ("a1", "a2"):
        for k in ("total_score", "pd", "features",
                  "has_unseen", "has_missing"):
            assert by1[aid][k] == by2[aid][k]


def test_batch_conflict_isolated_per_item(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    client.post(f"/cards/{CARD}/score", json={"features": FEAT,
                                              "request_id": "B9"})
    body = {"applicants": [
        {"applicant_id": "a1",
         "features": {**FEAT, "age": 99}, "request_id": "B9"},   # 冲突
        {"applicant_id": "a2",
         "features": {"age": 25, "income": 4000, "city": "CD",
                      "job_grade": "E"}, "request_id": "B10"},  # 正常
    ]}
    out = client.post(f"/cards/{CARD}/score/batch", json=body).json()
    by = {x["applicant_id"]: x for x in out["results"]}
    assert by["a1"]["ok"] is False and by["a1"]["conflict"] is True
    assert by["a2"]["ok"] is True
    assert out["conflicts"] == 1 and out["failed"] == 1
    assert len(_logs(client)) == 2  # 首次 B9 + 新的 B10


def test_single_and_batch_share_request_id_namespace(client, dev_csv_bytes):
    build_and_wait(client, CARD, dev_csv_bytes)
    s = client.post(f"/cards/{CARD}/score",
                    json={"features": FEAT, "request_id": "S1"}).json()
    body = {"applicants": [
        {"applicant_id": "a1", "features": FEAT, "request_id": "S1"}]}
    out = client.post(f"/cards/{CARD}/score/batch", json=body).json()
    item = out["results"][0]
    assert item["replayed"] is True
    assert item["total_score"] == s["total_score"]
    assert len(_logs(client)) == 1


def test_replay_returns_exact_training_row_scores(client, dev_csv_bytes):
    # 建卡样本逐条打分后重放，响应与第一次逐位一致
    build_and_wait(client, CARD, dev_csv_bytes)
    rows = load_dev_rows()
    first_bodies = []
    for i, row in enumerate(rows[:20]):
        r = client.post(f"/cards/{CARD}/score", json={
            "features": row_to_features(row), "request_id": f"X{i}"})
        assert r.status_code == 200, r.text
        first_bodies.append(r.json())
    for i, row in enumerate(rows[:20]):
        r = client.post(f"/cards/{CARD}/score", json={
            "features": row_to_features(row), "request_id": f"X{i}"})
        assert r.json() == first_bodies[i]
    assert len(_logs(client)) == 20
