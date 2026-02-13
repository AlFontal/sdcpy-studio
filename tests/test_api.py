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
        self.datasets = {}

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
        job.progress_current = 0
        job.progress_total = 1
        job.progress_description = "Running"

        try:
            job.result = run_sdc_job(request.model_dump(mode="python"))
            job.status = "succeeded"
            job.progress_current = 1
            job.progress_total = 1
            job.progress_description = "Completed"
        except Exception as exc:  # pragma: no cover
            job.status = "failed"
            job.error = str(exc)
            job.progress_current = 1
            job.progress_total = 1
            job.progress_description = "Failed"
        finally:
            job.completed_at = datetime.now(timezone.utc)

        self.jobs[job_id] = job
        return job

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def register_dataset(self, dataframe, filename: str):
        dataset_id = uuid4().hex
        record = type("Dataset", (), {})()
        record.dataset_id = dataset_id
        record.filename = filename
        record.dataframe = dataframe
        self.datasets[dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str):
        return self.datasets.get(dataset_id)

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


def test_load_oni_example_dataset():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    response = client.get("/api/v1/examples/oni-dataset")
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "oni_temp_sa.csv"
    assert payload["n_rows"] == 300
    assert "date" in payload["datetime_columns"]
    assert "oni_anomaly" in payload["numeric_columns"]
    assert "temp_anomaly_sa" in payload["numeric_columns"]


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
    assert "matrix_r" in payload
    assert "matrix_p" in payload
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


def test_dataset_inspection_and_submission_flow():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    months = np.arange(0, 30)
    nino34 = np.sin(months / 4.0)
    oni = np.sin(months / 4.0 + 0.3)
    rows = ["date,nino34,oni"]
    for idx in range(len(months)):
        year = 2000 + (idx // 12)
        month = (idx % 12) + 1
        date = f"{year}-{month:02d}-01"
        rows.append(f"{date},{nino34[idx]:.6f},{oni[idx]:.6f}")
    csv = "\n".join(rows)

    inspect = client.post(
        "/api/v1/datasets/inspect",
        files={"dataset_file": ("enso.csv", BytesIO(csv.encode("utf-8")), "text/csv")},
    )
    assert inspect.status_code == 200
    payload = inspect.json()
    assert payload["filename"] == "enso.csv"
    assert "date" in payload["datetime_columns"]
    assert "nino34" in payload["numeric_columns"]

    submit = client.post(
        "/api/v1/jobs/sdc/dataset",
        json={
            "dataset_id": payload["dataset_id"],
            "date_column": "date",
            "ts1_column": "nino34",
            "ts2_column": "oni",
            "fragment_size": 4,
            "n_permutations": 9,
            "method": "pearson",
            "alpha": 0.05,
            "min_lag": -4,
            "max_lag": 4,
        },
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["status"] == "succeeded"
    assert status_payload["progress"]["description"] == "Completed"

    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    result_payload = result.json()["result"]
    assert result_payload["summary"]["series_length"] == 30
    assert result_payload["series"]["index"][0].startswith("2000-01-01")


def test_download_endpoints():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    ts1, ts2 = _series_payload()
    submit = client.post(
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
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    xlsx = client.get(f"/api/v1/jobs/{job_id}/download/xlsx")
    assert xlsx.status_code == 200
    assert (
        xlsx.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert len(xlsx.content) > 0

    png = client.get(f"/api/v1/jobs/{job_id}/download/png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

    bad = client.get(f"/api/v1/jobs/{job_id}/download/nope")
    assert bad.status_code == 400
