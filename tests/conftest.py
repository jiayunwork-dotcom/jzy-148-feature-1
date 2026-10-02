"""pytest 公共夹具：强制内存后端、轮询后台作业。"""
from __future__ import annotations

import os
import time

os.environ["STORAGE_BACKEND"] = "memory"

import pytest
from fastapi.testclient import TestClient

from app.deps import repo, runtimes, scheduler
from app.main import app

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="session", autouse=True)
def _init():
    repo.init_schema()
    yield


@pytest.fixture()
def client():
    # 每个用例拿到干净的内存仓储与空运行时缓存（不用 with 触发 lifespan，
    # 调度线程池在整个测试会话复用）
    repo._cards.clear()
    repo._versions.clear()
    repo._jobs.clear()
    repo._score_records.clear()
    repo._score_index.clear()
    repo._backfill_jobs.clear()
    with repo._lock:
        repo._job_seq = 0
        repo._backfill_seq = 0
    runtimes._cache.clear()
    yield TestClient(app)


@pytest.fixture()
def dev_csv_bytes():
    with open(os.path.join(DATA_DIR, "dev_sample.csv"), "rb") as f:
        return f.read()


@pytest.fixture()
def handcalc_csv_bytes():
    with open(os.path.join(DATA_DIR, "handcalc_sample.csv"), "rb") as f:
        return f.read()


def build_and_wait(client, card: str, csv_bytes: bytes, **form) -> dict:
    """提交建卡作业并轮询到终态，返回 (job_json)。"""
    files = {"file": ("sample.csv", csv_bytes, "text/csv")}
    resp = client.post(f"/cards/{card}/build", files=files, data=form)
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    for _ in range(500):
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.01)
    raise AssertionError("作业长时间未结束")


def latest_artifact(client, card: str, version=None) -> dict:
    if version is None:
        versions = client.get(f"/cards/{card}/versions").json()
        version = max(v["version"] for v in versions)
    return client.get(f"/cards/{card}/versions/{version}/artifact").json()
