from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sdcpy_studio.main import create_app
from sdcpy_studio.schemas import SDCJobRequest
from sdcpy_studio.service import run_sdc_job


class InlineJobManager:
    """Test helper that executes jobs synchronously in-process."""

    def __init__(self):
        self.jobs = {}
        self.datasets = {}
        self.map_uploads = {}
        self._tmpdir = TemporaryDirectory(prefix="sdcpy-studio-tests-")
        self._tmpdir_path = Path(self._tmpdir.name)

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

    def submit_map(self, request):
        job_id = uuid4().hex
        now = datetime.now(timezone.utc)
        job = type("Job", (), {})()
        job.job_id = job_id
        job.status = "succeeded"
        job.created_at = now
        job.started_at = now
        job.completed_at = now
        job.error = None
        job.progress_current = 1
        job.progress_total = 1
        job.progress_description = "Completed"
        job.result = {
            "summary": {
                "driver_dataset": request.driver_dataset,
                "field_dataset": request.field_dataset,
                "correlation_width": request.correlation_width,
                "valid_cells": 12,
            },
            "event_catalog": {
                "selected_positive": [{"date": "2010-01-01", "value": 1.2, "sign": "positive"}],
                "selected_negative": [{"date": "2010-07-01", "value": -1.1, "sign": "negative"}],
                "ignored_positive": [],
                "ignored_negative": [],
                "base_state_threshold": 0.55,
                "base_state_count": 18,
                "warnings": [],
            },
            "class_results": {
                "positive": {
                    "summary": {"selected_event_count": 1, "valid_cells": 12, "lag_valid_cells": [12]},
                    "layer_maps": {
                        "lat": [0.0],
                        "lon": [0.0],
                        "coastline": {"lat": [None], "lon": [None]},
                        "layers": [],
                    },
                    "lag_maps": {
                        "lat": [0.0],
                        "lon": [0.0],
                        "lags": [0],
                        "coastline": {"lat": [None], "lon": [None]},
                        "corr_by_lag": [[[0.6]]],
                        "event_count_by_lag": [[[1]]],
                    },
                    "empty_reason": None,
                },
                "negative": {
                    "summary": {"selected_event_count": 1, "valid_cells": 9, "lag_valid_cells": [9]},
                    "layer_maps": {
                        "lat": [0.0],
                        "lon": [0.0],
                        "coastline": {"lat": [None], "lon": [None]},
                        "layers": [],
                    },
                    "lag_maps": {
                        "lat": [0.0],
                        "lon": [0.0],
                        "lags": [0],
                        "coastline": {"lat": [None], "lon": [None]},
                        "corr_by_lag": [[[-0.5]]],
                        "event_count_by_lag": [[[1]]],
                    },
                    "empty_reason": None,
                },
            },
            "notes": [],
            "runtime_seconds": 0.12,
            "figure_png_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR4nGMwMDD4DwAD1QG6hQm8WQAAAABJRU5ErkJggg==",
            "download_formats": ["png", "nc"],
            "_artifacts_map": {
                "png": b"\x89PNG\r\n\x1a\n",
                "png_positive": b"\x89PNG\r\n\x1a\nPOS",
                "png_negative": b"\x89PNG\r\n\x1a\nNEG",
                "nc": b"CDF\x01",
                "driver_dataset": request.driver_dataset,
                "field_dataset": request.field_dataset,
                "correlation_width": request.correlation_width,
            },
        }
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

    def register_map_upload(self, *, kind: str, filename: str, content: bytes, metadata=None):
        upload_id = uuid4().hex
        suffix = Path(filename).suffix or (".csv" if kind == "driver" else ".nc")
        path = self._tmpdir_path / f"{kind}_{upload_id}{suffix}"
        path.write_bytes(content)
        record = type("MapUpload", (), {})()
        record.upload_id = upload_id
        record.kind = kind
        record.filename = filename
        record.path = path
        record.metadata = metadata or {}
        self.map_uploads[upload_id] = record
        return record

    def get_map_upload(self, upload_id: str):
        return self.map_uploads.get(upload_id)

    def shutdown(self):
        self._tmpdir.cleanup()
        return None


class StaticMapCacheManager:
    def __init__(self, snapshot_payload: dict):
        self._snapshot_payload = snapshot_payload

    def start(self):
        return None

    def stop(self):
        return None

    def snapshot(self):
        return dict(self._snapshot_payload)


def _series_payload(n: int = 80):
    x = np.linspace(0, 8 * np.pi, n)
    ts1 = np.sin(x)
    ts2 = np.sin(x + 0.7)
    return ts1.tolist(), ts2.tolist()


def _custom_map_driver_csv(n: int = 72) -> str:
    rows = ["date,my_index,other"]
    for i in range(n):
        year = 2000 + (i // 12)
        month = (i % 12) + 1
        rows.append(
            f"{year}-{month:02d}-01,{np.sin(i / 5):.6f},{np.cos(i / 7):.6f}"
        )
    return "\n".join(rows)


def _custom_map_field_netcdf_bytes() -> bytes:
    xr = pytest.importorskip("xarray")
    times = np.array(
        [f"2000-{m:02d}-01" for m in range(1, 7)] + [f"2001-{m:02d}-01" for m in range(1, 7)],
        dtype="datetime64[ns]",
    )
    lats = np.array([-10.0, 0.0, 10.0], dtype=float)
    lons = np.array([120.0, 140.0, 160.0, 180.0], dtype=float)
    tt, yy, xx = np.meshgrid(np.arange(times.size), lats, lons, indexing="ij")
    values = np.sin(tt / 2.0) + 0.1 * yy + 0.01 * xx
    ds = xr.Dataset(
        data_vars={
            "sst_anom": (("time", "lat", "lon"), values.astype("float32")),
            "bad2d": (("lat", "lon"), values[0].astype("float32")),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    payload = ds.to_netcdf()
    return payload if isinstance(payload, bytes) else bytes(payload)


def _custom_map_field_singleton_level_netcdf_bytes() -> bytes:
    xr = pytest.importorskip("xarray")
    times = np.array([f"2000-{m:02d}-01" for m in range(1, 7)], dtype="datetime64[ns]")
    levels = np.array([2.0], dtype=float)
    lats = np.array([-10.0, 0.0, 10.0], dtype=float)
    lons = np.array([120.0, 140.0, 160.0, 180.0], dtype=float)
    tt, ll, yy, xx = np.meshgrid(
        np.arange(times.size),
        np.arange(levels.size),
        lats,
        lons,
        indexing="ij",
    )
    values = np.sin(tt / 2.0) + 0.05 * ll + 0.1 * yy + 0.01 * xx
    ds = xr.Dataset(
        data_vars={"air": (("time", "level", "lat", "lon"), values.astype("float32"))},
        coords={"time": times, "level": levels, "lat": lats, "lon": lons},
    )
    payload = ds.to_netcdf()
    return payload if isinstance(payload, bytes) else bytes(payload)


def _custom_map_field_multilevel_netcdf_bytes() -> bytes:
    xr = pytest.importorskip("xarray")
    times = np.array([f"2000-{m:02d}-01" for m in range(1, 5)], dtype="datetime64[ns]")
    levels = np.array([850.0, 500.0], dtype=float)
    lats = np.array([-10.0, 10.0], dtype=float)
    lons = np.array([120.0, 150.0], dtype=float)
    tt, ll, yy, xx = np.meshgrid(
        np.arange(times.size),
        np.arange(levels.size),
        lats,
        lons,
        indexing="ij",
    )
    values = np.sin(tt / 2.0) + 0.2 * ll + 0.1 * yy + 0.01 * xx
    ds = xr.Dataset(
        data_vars={"air": (("time", "level", "lat", "lon"), values.astype("float32"))},
        coords={"time": times, "level": levels, "lat": lats, "lon": lons},
    )
    payload = ds.to_netcdf()
    return payload if isinstance(payload, bytes) else bytes(payload)


def _custom_map_field_multilevel_without_coord_netcdf_bytes() -> bytes:
    xr = pytest.importorskip("xarray")
    times = np.array([f"2000-{m:02d}-01" for m in range(1, 5)], dtype="datetime64[ns]")
    lats = np.array([-10.0, 10.0], dtype=float)
    lons = np.array([120.0, 150.0], dtype=float)
    values = np.arange(times.size * 3 * lats.size * lons.size, dtype="float32").reshape(
        times.size, 3, lats.size, lons.size
    )
    ds = xr.Dataset(
        data_vars={"air": (("time", "member", "lat", "lon"), values)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    payload = ds.to_netcdf()
    return payload if isinstance(payload, bytes) else bytes(payload)


def test_root_page_renders():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "scale-dependent correlation" in response.text.lower()


def test_health_exposes_map_cache_status():
    app = create_app(
        job_manager=InlineJobManager(),
        map_cache_manager=StaticMapCacheManager(
            {
                "status": "warming",
                "mode": "auto",
                "cache_dir": "/tmp/map-cache",
                "manifest_present": False,
                "catalog_ready": False,
                "coastline_ready": True,
                "completed_at": None,
                "detail": "warming",
                "error": None,
            }
        ),
    )
    client = TestClient(app)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["map_cache"]["status"] == "warming"
    assert payload["map_cache"]["coastline_ready"] is True


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


def test_load_oni_example_dataset_csv():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    response = client.get("/api/v1/examples/oni-dataset.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="oni_temp_sa.csv"' in response.headers["content-disposition"]
    assert response.text.startswith("date,oni_anomaly,temp_anomaly_sa")


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
    assert "lag_matrix_r" in payload
    assert "lag_matrix_p" in payload
    assert payload["lag_default"] == 0
    assert payload["lag_matrix_r"]["y"]
    assert payload["lag_matrix_p"]["y"] == payload["lag_matrix_r"]["y"]
    matrix_x = payload["matrix_r"]["x"]
    matrix_y = payload["matrix_r"]["y"]
    matrix_z = payload["matrix_r"]["z"]
    checked = 0
    for row_idx, start2 in enumerate(matrix_y):
        for col_idx, start1 in enumerate(matrix_x):
            lag = int(start1) - int(start2)
            if lag < -10 or lag > 10:
                assert matrix_z[row_idx][col_idx] is None
                checked += 1
                if checked >= 40:
                    break
        if checked >= 40:
            break

    lag_x = payload["lag_matrix_r"]["x"]
    lag_y = payload["lag_matrix_r"]["y"]
    lag_z = payload["lag_matrix_r"]["z"]
    min_start2 = min(matrix_y)
    max_start2 = max(matrix_y)
    checked_lag = 0
    for row_idx, lag in enumerate(lag_y):
        for col_idx, start1 in enumerate(lag_x):
            start2 = int(start1) - int(lag)
            if start2 < min_start2 or start2 > max_start2:
                assert lag_z[row_idx][col_idx] is None
                checked_lag += 1
                if checked_lag >= 40:
                    break
        if checked_lag >= 40:
            break

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


def test_lag_default_uses_nearest_available_when_zero_not_in_range():
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
            "min_lag": 5,
            "max_lag": 10,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    payload = result.json()["result"]
    lag_values = payload["lag_matrix_r"]["y"]
    assert lag_values
    nearest = min(lag_values, key=lambda value: abs(value))
    assert payload["lag_default"] == nearest


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
    assert 'filename="sdc_ts1_ts2_12.xlsx"' in xlsx.headers["content-disposition"]
    assert len(xlsx.content) > 0

    png = client.get(f"/api/v1/jobs/{job_id}/download/png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert 'filename="sdc_ts1_ts2_12.png"' in png.headers["content-disposition"]
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

    bad = client.get(f"/api/v1/jobs/{job_id}/download/nope")
    assert bad.status_code == 400


def test_map_job_endpoints():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    submit = client.post(
        "/api/v1/jobs/sdc-map",
        json={
            "driver_dataset": "pdo",
            "field_dataset": "ncep_air",
            "correlation_width": 12,
            "n_positive_peaks": 3,
            "n_negative_peaks": 3,
            "base_state_beta": 0.5,
            "alpha": 0.05,
            "n_permutations": 49,
            "min_lag": -6,
            "max_lag": 6,
            "time_start": "2010-01-01",
            "time_end": "2012-12-01",
        },
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    status = client.get(f"/api/v1/jobs/sdc-map/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"

    result = client.get(f"/api/v1/jobs/sdc-map/{job_id}/result")
    assert result.status_code == 200
    payload = result.json()["result"]
    assert payload["summary"]["driver_dataset"] == "pdo"
    assert payload["download_formats"] == ["png", "nc"]
    assert payload["summary"]["correlation_width"] == 12

    png = client.get(f"/api/v1/jobs/sdc-map/{job_id}/download/png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert 'positive_negative.png"' in png.headers["content-disposition"]

    png_positive = client.get(f"/api/v1/jobs/sdc-map/{job_id}/download/png?sign=positive")
    assert png_positive.status_code == 200
    assert png_positive.content.endswith(b"POS")
    assert 'positive.png"' in png_positive.headers["content-disposition"]

    png_negative = client.get(f"/api/v1/jobs/sdc-map/{job_id}/download/png?sign=negative")
    assert png_negative.status_code == 200
    assert png_negative.content.endswith(b"NEG")
    assert 'negative.png"' in png_negative.headers["content-disposition"]

    nc = client.get(f"/api/v1/jobs/sdc-map/{job_id}/download/nc")
    assert nc.status_code == 200
    assert nc.headers["content-type"] == "application/x-netcdf"
    assert nc.content[:3] == b"CDF"


def test_map_custom_upload_inspection_endpoints():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    driver_csv = _custom_map_driver_csv()
    driver_resp = client.post(
        "/api/v1/sdc-map/driver/inspect",
        files={"driver_file": ("driver.csv", BytesIO(driver_csv.encode("utf-8")), "text/csv")},
    )
    assert driver_resp.status_code == 200
    driver = driver_resp.json()
    assert driver["filename"] == "driver.csv"
    assert driver["upload_id"]
    assert "date" in driver["datetime_columns"]
    assert "my_index" in driver["numeric_columns"]
    assert driver["suggested_date_column"] == "date"
    assert driver["defaults"]["time_start"]
    assert "event_catalog" in driver

    field_resp = client.post(
        "/api/v1/sdc-map/field/inspect",
        files={
            "field_file": (
                "field.nc",
                BytesIO(_custom_map_field_netcdf_bytes()),
                "application/x-netcdf",
            )
        },
    )
    assert field_resp.status_code == 200
    field = field_resp.json()
    assert field["filename"] == "field.nc"
    assert field["upload_id"]
    assert "sst_anom" in field["variables"]
    assert "sst_anom" in field["compatible_variables"]
    assert field["suggested_variable"] == "sst_anom"
    assert field["dims"]["time"] == 12
    assert field["dims"]["lat"] == 3
    assert field["dims"]["lon"] == 4


def test_map_driver_inspect_accepts_semicolon_delimited_csv():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    rows = ["date;my_index;alt_series"]
    for idx in range(12):
        rows.append(f"2000-{idx + 1:02d}-01;{idx / 10:.3f};{1 - idx / 20:.3f}")
    payload = "\n".join(rows)

    response = client.post(
        "/api/v1/sdc-map/driver/inspect",
        files={"driver_file": ("driver.csv", BytesIO(payload.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detected_delimiter"] == ";"
    assert body["delimiter_name"] == "semicolon"
    assert body["suggested_date_column"] == "date"
    assert body["suggested_value_column"] == "my_index"


def test_map_driver_inspect_reports_auto_selected_numeric_column_warning():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    rows = ["date,driver_index,alt_series"]
    alt_values = [1.0, 0.95, 0.87, 0.75, 0.61, 0.44, 0.26, 0.07, -0.12, -0.30, -0.48, -0.62]
    for idx, alt_value in enumerate(alt_values, start=1):
        driver_value = "oops" if idx in {2, 5, 8, 11} else f"{idx / 10:.3f}"
        rows.append(f"1998-{idx:02d}-01,{driver_value},{alt_value:.3f}")
    payload = "\n".join(rows)

    response = client.post(
        "/api/v1/sdc-map/driver/inspect",
        files={"driver_file": ("driver.csv", BytesIO(payload.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_value_column"] == "alt_series"
    assert "driver_index" in body["rejected_numeric_columns"]
    assert any(warning["code"] == "series_auto_selected" for warning in body["warnings"])


def test_map_field_inspect_accepts_singleton_level_dimension():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    response = client.post(
        "/api/v1/sdc-map/field/inspect",
        files={
            "field_file": (
                "air.2m.mon.mean.nc",
                BytesIO(_custom_map_field_singleton_level_netcdf_bytes()),
                "application/x-netcdf",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_variable"] == "air"
    assert body["dims"] == {"time": 6, "lat": 3, "lon": 4}
    assert body["normalization"]["original_dims"] == ["time", "level", "lat", "lon"]
    assert body["normalization"]["squeezed_dims"] == ["level"]
    assert any(warning["code"] == "singleton_dimensions_squeezed" for warning in body["warnings"])


def test_map_field_inspect_exposes_selector_for_multilevel_dimension():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    response = client.post(
        "/api/v1/sdc-map/field/inspect",
        files={
            "field_file": (
                "air.multilevel.nc",
                BytesIO(_custom_map_field_multilevel_netcdf_bytes()),
                "application/x-netcdf",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_variable"] == "air"
    assert body["compatible_variables"] == ["air"]
    assert body["dims"] == {"time": 4, "lat": 2, "lon": 2}
    assert body["normalization"]["selected_dimensions"] == {"level": "850"}
    variable_option = body["variable_options"][0]
    assert variable_option["name"] == "air"
    assert variable_option["selectors"][0]["dimension"] == "level"
    assert variable_option["selectors"][0]["suggested_value"] == "850"
    assert [option["value"] for option in variable_option["selectors"][0]["options"]] == ["850", "500"]
    assert any(warning["code"] == "dimension_selection_available" for warning in body["warnings"])


def test_map_field_inspect_rejects_non_singleton_dimension_without_coordinate_values():
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    response = client.post(
        "/api/v1/sdc-map/field/inspect",
        files={
            "field_file": (
                "air.members.nc",
                BytesIO(_custom_map_field_multilevel_without_coord_netcdf_bytes()),
                "application/x-netcdf",
            )
        },
    )
    assert response.status_code == 422
    assert "No compatible variable found in NetCDF" in response.json()["detail"]
    assert "selectable coordinate axis" in response.json()["detail"]


def test_map_field_inspect_falls_back_to_manual_time_decode(monkeypatch):
    xr = pytest.importorskip("xarray")
    original_open_dataset = xr.open_dataset

    def fake_open_dataset(*args, **kwargs):
        if kwargs.get("decode_times", True):
            raise ValueError("raw xarray decode failure")
        return original_open_dataset(*args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)
    response = client.post(
        "/api/v1/sdc-map/field/inspect",
        files={
            "field_file": (
                "field.nc",
                BytesIO(_custom_map_field_netcdf_bytes()),
                "application/x-netcdf",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_variable"] == "sst_anom"
    assert any(warning["code"] == "time_axis_manually_decoded" for warning in body["warnings"])


def test_map_custom_upload_submit_and_explore_with_stubbed_map_service(monkeypatch):
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    driver_resp = client.post(
        "/api/v1/sdc-map/driver/inspect",
        files={
            "driver_file": (
                "driver.csv",
                BytesIO(_custom_map_driver_csv().encode("utf-8")),
                "text/csv",
            )
        },
    )
    field_resp = client.post(
        "/api/v1/sdc-map/field/inspect",
        files={
            "field_file": (
                "field.nc",
                BytesIO(_custom_map_field_netcdf_bytes()),
                "application/x-netcdf",
            )
        },
    )
    assert driver_resp.status_code == 200
    assert field_resp.status_code == 200
    driver = driver_resp.json()
    field = field_resp.json()

    captured = {}

    def fake_build(payload):
        captured["explore"] = payload
        return {
            "summary": {
                "driver_dataset": "driver.csv",
                "field_dataset": "field.nc",
                "time_start": "2000-01-01",
                "time_end": "2001-12-01",
                "correlation_width": 12,
                "n_positive_peaks": 3,
                "n_negative_peaks": 3,
                "base_state_beta": 0.5,
                "n_time": 12,
                "n_lat": 3,
                "n_lon": 4,
                "valid_values": 144,
                "valid_rate": 1.0,
                "first_valid_index": [0, 0, 0],
                "field_lat_min": -10.0,
                "field_lat_max": 10.0,
                "field_lon_min": 120.0,
                "field_lon_max": 180.0,
                "field_value_min": -1.0,
                "field_value_max": 2.0,
                "used_lat_min": -10.0,
                "used_lat_max": 10.0,
                "used_lon_min": 120.0,
                "used_lon_max": 180.0,
                "full_bounds_selected": True,
                "driver_source_type": "upload",
                "field_source_type": "upload",
                "field_variable": "sst_anom",
            },
            "event_catalog": {
                "selected_positive": [{"date": "2000-06-01", "value": 1.2, "sign": "positive"}],
                "selected_negative": [{"date": "2001-01-01", "value": -1.1, "sign": "negative"}],
                "ignored_positive": [],
                "ignored_negative": [],
                "base_state_threshold": 0.55,
                "base_state_count": 8,
                "warnings": [],
            },
            "time_index": ["2000-01-01"],
            "driver_values": [0.1],
            "lat": [-10.0],
            "lon": [120.0],
            "field_frames": [[[0.2]]],
            "coastline": {"lat": [None], "lon": [None]},
        }

    monkeypatch.setattr("sdcpy_studio.main.build_sdc_map_exploration", fake_build)

    explore_payload = {
        "driver_dataset": "custom_driver",
        "field_dataset": "custom_field",
        "driver_source_type": "upload",
        "field_source_type": "upload",
        "driver_upload_id": driver["upload_id"],
        "driver_date_column": driver["suggested_date_column"],
        "driver_value_column": driver["suggested_value_column"],
        "field_upload_id": field["upload_id"],
        "field_variable": field["suggested_variable"],
        "correlation_width": 12,
        "n_positive_peaks": 3,
        "n_negative_peaks": 3,
        "base_state_beta": 0.5,
        "n_permutations": 9,
        "alpha": 0.05,
        "min_lag": -4,
        "max_lag": 4,
    }

    explore = client.post("/api/v1/sdc-map/explore", json=explore_payload)
    assert explore.status_code == 200
    assert captured["explore"]["driver_upload_path"]
    assert captured["explore"]["field_upload_path"]
    assert captured["explore"]["driver_upload_filename"] == "driver.csv"
    assert captured["explore"]["field_upload_filename"] == "field.nc"
    assert explore.json()["result"]["summary"]["driver_source_type"] == "upload"

    submit = client.post("/api/v1/jobs/sdc-map", json=explore_payload)
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]
    result = client.get(f"/api/v1/jobs/sdc-map/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["result"]["summary"]["correlation_width"] == 12


def test_map_custom_upload_submit_and_explore_pass_field_dimension_selections(monkeypatch):
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    driver_resp = client.post(
        "/api/v1/sdc-map/driver/inspect",
        files={
            "driver_file": (
                "driver.csv",
                BytesIO(_custom_map_driver_csv().encode("utf-8")),
                "text/csv",
            )
        },
    )
    field_resp = client.post(
        "/api/v1/sdc-map/field/inspect",
        files={
            "field_file": (
                "field-multilevel.nc",
                BytesIO(_custom_map_field_multilevel_netcdf_bytes()),
                "application/x-netcdf",
            )
        },
    )
    assert driver_resp.status_code == 200
    assert field_resp.status_code == 200
    driver = driver_resp.json()
    field = field_resp.json()

    captured = {}

    def fake_build(payload):
        captured["explore"] = payload
        return {
            "summary": {
                "driver_dataset": "driver.csv",
                "field_dataset": "field-multilevel.nc",
                "time_start": "2000-01-01",
                "time_end": "2000-04-01",
                "correlation_width": 12,
                "n_positive_peaks": 3,
                "n_negative_peaks": 3,
                "base_state_beta": 0.5,
                "n_time": 4,
                "n_lat": 2,
                "n_lon": 2,
                "valid_values": 16,
                "valid_rate": 1.0,
                "field_lat_min": -10.0,
                "field_lat_max": 10.0,
                "field_lon_min": 120.0,
                "field_lon_max": 150.0,
                "full_bounds_selected": True,
                "driver_source_type": "upload",
                "field_source_type": "upload",
                "field_variable": "air",
                "field_dimension_selections": {"level": "850"},
            },
            "event_catalog": {
                "selected_positive": [{"date": "2000-02-01", "value": 0.8, "sign": "positive"}],
                "selected_negative": [],
                "ignored_positive": [],
                "ignored_negative": [],
                "base_state_threshold": 0.4,
                "base_state_count": 2,
                "warnings": [],
            },
            "time_index": ["2000-01-01"],
            "driver_values": [0.1],
            "lat": [-10.0],
            "lon": [120.0],
            "field_frames": [[[0.2]]],
            "coastline": {"lat": [None], "lon": [None]},
        }

    monkeypatch.setattr("sdcpy_studio.main.build_sdc_map_exploration", fake_build)

    explore_payload = {
        "driver_dataset": "custom_driver",
        "field_dataset": "custom_field",
        "driver_source_type": "upload",
        "field_source_type": "upload",
        "driver_upload_id": driver["upload_id"],
        "driver_date_column": driver["suggested_date_column"],
        "driver_value_column": driver["suggested_value_column"],
        "field_upload_id": field["upload_id"],
        "field_variable": field["suggested_variable"],
        "field_dimension_selections": {"level": "850"},
        "correlation_width": 12,
        "n_positive_peaks": 3,
        "n_negative_peaks": 3,
        "base_state_beta": 0.5,
        "n_permutations": 9,
        "alpha": 0.05,
        "min_lag": -4,
        "max_lag": 4,
    }

    explore = client.post("/api/v1/sdc-map/explore", json=explore_payload)
    assert explore.status_code == 200
    assert captured["explore"]["field_dimension_selections"] == {"level": "850"}


def test_map_explore_endpoint(monkeypatch):
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    def _fake_explore(_payload):
        return {
            "summary": {
                "driver_dataset": "pdo",
                "field_dataset": "ncep_air",
                "n_time": 3,
                "correlation_width": 12,
                "n_positive_peaks": 3,
                "n_negative_peaks": 3,
                "base_state_beta": 0.5,
            },
            "event_catalog": {
                "selected_positive": [{"date": "2010-02-01", "value": 0.9, "sign": "positive"}],
                "selected_negative": [{"date": "2010-03-01", "value": -0.8, "sign": "negative"}],
                "ignored_positive": [],
                "ignored_negative": [],
                "base_state_threshold": 0.45,
                "base_state_count": 1,
                "warnings": [],
            },
            "time_index": ["2010-01-01", "2010-02-01", "2010-03-01"],
            "driver_values": [0.1, 0.2, 0.3],
            "lat": [10.0, 20.0],
            "lon": [30.0, 40.0],
            "field_frames": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.1, 2.1], [3.1, 4.1]],
                [[1.2, 2.2], [3.2, 4.2]],
            ],
            "coastline": {"lon": [30.0, 40.0, None], "lat": [10.0, 20.0, None]},
        }

    monkeypatch.setattr("sdcpy_studio.main.build_sdc_map_exploration", _fake_explore)
    response = client.post(
        "/api/v1/sdc-map/explore",
        json={
            "driver_dataset": "pdo",
            "field_dataset": "ncep_air",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["result"]["summary"]["driver_dataset"] == "pdo"
    assert len(payload["result"]["time_index"]) == 3
    assert len(payload["result"]["field_frames"]) == 3


def test_map_catalog_endpoint(monkeypatch):
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    monkeypatch.setattr(
        "sdcpy_studio.main.get_sdc_map_catalog",
        lambda: {
            "drivers": [{"key": "pdo", "description": "Pacific Decadal Oscillation"}],
            "fields": [{"key": "ncep_air", "description": "NCEP air temperature", "variable": "air"}],
        },
    )
    response = client.get("/api/v1/sdc-map/catalog")
    assert response.status_code == 200
    payload = response.json()
    assert payload["drivers"][0]["key"] == "pdo"
    assert payload["fields"][0]["key"] == "ncep_air"


def test_map_defaults_endpoint(monkeypatch):
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    monkeypatch.setattr(
        "sdcpy_studio.main.get_sdc_map_driver_defaults",
        lambda key: {
            "driver_dataset": key,
            "time_start": "2012-01-01",
            "time_end": "2018-01-01",
            "driver_min_date": "2012-01-01",
            "driver_max_date": "2018-01-01",
            "n_points": 84,
            "event_catalog": {
                "selected_positive": [{"date": "2015-01-01", "value": 1.1, "sign": "positive"}],
                "selected_negative": [{"date": "2016-01-01", "value": -1.0, "sign": "negative"}],
                "ignored_positive": [],
                "ignored_negative": [],
                "base_state_threshold": 0.5,
                "base_state_count": 12,
                "warnings": [],
            },
        },
    )
    response = client.get("/api/v1/sdc-map/defaults?driver_dataset=pdo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["driver_dataset"] == "pdo"
    assert payload["event_catalog"]["selected_positive"][0]["date"] == "2015-01-01"


def test_map_driver_preview_endpoint(monkeypatch):
    app = create_app(job_manager=InlineJobManager())
    client = TestClient(app)

    monkeypatch.setattr(
        "sdcpy_studio.main.build_sdc_map_driver_preview",
        lambda payload: {
            "summary": {
                "driver_dataset": payload["driver_dataset"],
                "time_start": "2000-01-01",
                "time_end": "2001-12-01",
                "correlation_width": 12,
                "n_positive_peaks": 2,
                "n_negative_peaks": 1,
                "base_state_beta": 0.5,
                "n_points": 24,
            },
            "event_catalog": {
                "selected_positive": [{"date": "2000-08-01", "value": 1.1, "sign": "positive"}],
                "selected_negative": [{"date": "2001-01-01", "value": -0.9, "sign": "negative"}],
                "ignored_positive": [],
                "ignored_negative": [],
                "base_state_threshold": 0.45,
                "base_state_count": 8,
                "warnings": [],
            },
            "time_index": ["2000-01-01", "2000-02-01"],
            "driver_values": [0.1, 0.3],
        },
    )

    response = client.post(
        "/api/v1/sdc-map/driver/preview",
        json={
            "driver_dataset": "pdo",
            "field_dataset": "ncep_air",
            "correlation_width": 12,
            "n_positive_peaks": 2,
            "n_negative_peaks": 1,
            "base_state_beta": 0.5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["result"]["summary"]["n_positive_peaks"] == 2
    assert payload["result"]["event_catalog"]["selected_positive"][0]["date"] == "2000-08-01"
