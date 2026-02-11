from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from sdcpy_studio.main import create_app
from sdcpy_studio.schemas import SDCJobRequest
from sdcpy_studio.service import run_sdc_job


class InlineJobManager:
    """Test helper that executes jobs synchronously in-process."""

    def __init__(self):
        self.jobs = {}

    def submit(self, request: SDCJobRequest):
        job_id = uuid4().hex
        now = datetime.now(timezone.utc)
        job = type("Job", (), {})()
        job.job_id = job_id
        job.status = "running"
        job.created_at = now
        job.started_at = now
        job.completed_at = None
        job.error = None
        job.result = None

        try:
            job.result = run_sdc_job(request.model_dump(mode="python"))
            job.status = "succeeded"
        except Exception as exc:  # pragma: no cover
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.completed_at = datetime.now(timezone.utc)

        self.jobs[job_id] = job
        return job

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def shutdown(self):
        return None


def _series_payload(n: int = 80):
    x = np.linspace(0, 8 * np.pi, n)
    ts1 = np.sin(x)
    ts2 = np.sin(x + 0.7)
    return ts1.tolist(), ts2.tolist()


def test_root_page_renders():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "sdcpy Studio" in response.text


def test_submit_json_job_and_fetch_result():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    ts1, ts2 = _series_payload()
    response = client.post(
        "/api/v1/jobs/sdc",
        json={
            "ts1": ts1,
            "ts2": ts2,
            "fragment_size": 12,
            "n_permutations": 9,
            "method": "pearson",
            "alpha": 0.05,
            "min_lag": -10,
            "max_lag": 10,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"

    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    payload = result.json()["result"]
    assert payload["summary"]["series_length"] == len(ts1)
    assert payload["series"]["index"][0] == 0
    assert len(payload["series"]["ts1"]) == len(ts1)
    assert len(payload["series"]["ts2"]) == len(ts2)
    assert "heatmap_all" in payload
    assert "runtime_seconds" in payload


def test_submit_csv_job():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    ts1, ts2 = _series_payload(72)
    csv_1 = "value\n" + "\n".join(f"{v:.6f}" for v in ts1)
    csv_2 = "value\n" + "\n".join(f"{v:.6f}" for v in ts2)

    response = client.post(
        "/api/v1/jobs/sdc/csv",
        files={
            "ts1_file": ("ts1.csv", BytesIO(csv_1.encode("utf-8")), "text/csv"),
            "ts2_file": ("ts2.csv", BytesIO(csv_2.encode("utf-8")), "text/csv"),
        },
        data={
            "fragment_size": 10,
            "n_permutations": 9,
            "method": "pearson",
            "alpha": 0.05,
            "min_lag": -8,
            "max_lag": 8,
            "two_tailed": "true",
            "permutations": "true",
            "max_memory_gb": 1.0,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["status"] == "succeeded"
