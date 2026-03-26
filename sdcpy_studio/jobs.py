"""Asynchronous job and dataset management for sdcpy-studio."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
from uuid import uuid4

import pandas as pd

from sdcpy_studio.schemas import SDCJobRequest, SDCMapJobRequest
from sdcpy_studio.service import SDCMapJobCancelledError, run_sdc_job, run_sdc_map_job


@dataclass
class JobRecord:
    """In-memory record for one submitted SDC job."""

    job_id: str
    status: str
    request: SDCJobRequest | SDCMapJobRequest
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict | None = None
    error: str | None = None
    progress_current: int = 0
    progress_total: int = 1
    progress_description: str = "Queued"


@dataclass
class DatasetRecord:
    """In-memory record for one uploaded dataset."""

    dataset_id: str
    filename: str
    created_at: datetime
    dataframe: pd.DataFrame


@dataclass
class MapUploadRecord:
    """In-memory record for one uploaded SDC map asset stored on disk."""

    upload_id: str
    kind: str
    filename: str
    created_at: datetime
    path: Path
    metadata: dict | None = None


class JobManager:
    """Asynchronous job manager using a thread pool plus in-memory dataset storage."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}
        self._cancel_events: dict[str, Event] = {}
        self._datasets: dict[str, DatasetRecord] = {}
        self._map_uploads: dict[str, MapUploadRecord] = {}
        self._tmpdir = TemporaryDirectory(prefix="sdcpy-studio-")
        self._tmpdir_path = Path(self._tmpdir.name)

    def submit(self, request: SDCJobRequest) -> JobRecord:
        """Submit a new background computation."""
        return self._submit_with_runner(request, run_sdc_job)

    def submit_map(self, request: SDCMapJobRequest) -> JobRecord:
        """Submit a new background computation for SDC map mode."""
        return self._submit_with_runner(request, run_sdc_map_job, supports_cancel=True)

    def _submit_with_runner(
        self,
        request: SDCJobRequest | SDCMapJobRequest,
        runner,
        *,
        supports_cancel: bool = False,
    ) -> JobRecord:
        """Submit a new background computation."""
        job_id = uuid4().hex
        created_at = datetime.now(timezone.utc)
        record = JobRecord(
            job_id=job_id,
            status="queued",
            request=request,
            created_at=created_at,
            progress_current=0,
            progress_total=1,
            progress_description="Queued",
        )

        cancel_event = Event() if supports_cancel else None

        with self._lock:
            self._jobs[job_id] = record
            if cancel_event is not None:
                self._cancel_events[job_id] = cancel_event

        def _progress_update(current: int, total: int, description: str) -> None:
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                if rec.status == "cancelled":
                    return
                rec.progress_current = int(max(0, current))
                rec.progress_total = int(max(1, total))
                rec.progress_description = description

        def _run_job():
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    raise RuntimeError("Submitted job record disappeared before execution.")
                if cancel_event is not None and cancel_event.is_set():
                    rec.status = "cancelled"
                    rec.completed_at = datetime.now(timezone.utc)
                    rec.progress_description = "Cancelled"
                    raise SDCMapJobCancelledError("Map job cancelled before execution started.")
                rec.status = "running"
                rec.started_at = datetime.now(timezone.utc)
                if rec.progress_description.lower() == "queued":
                    rec.progress_description = "Starting job"
            if cancel_event is not None:
                return runner(request.model_dump(mode="python"), _progress_update, cancel_event)
            return runner(request.model_dump(mode="python"), _progress_update)

        future = self._executor.submit(_run_job)

        with self._lock:
            self._futures[job_id] = future

        future.add_done_callback(lambda fut, jid=job_id: self._finalize(jid, fut))
        return record

    def cancel_map(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.status in {"succeeded", "failed", "cancelled"}:
                return record
            cancel_event = self._cancel_events.get(job_id)
            future = self._futures.get(job_id)
            if cancel_event is None or future is None:
                return record

            cancel_event.set()
            if future.cancel():
                record.status = "cancelled"
                record.completed_at = datetime.now(timezone.utc)
                record.progress_description = "Cancelled"
                return record

            if record.status in {"queued", "running"}:
                record.status = "cancelling"
                record.progress_description = "Cancelling"
            return record

    def _finalize(self, job_id: str, future: Future) -> None:
        with self._lock:
            record = self._jobs[job_id]
            try:
                if future.cancelled():
                    record.status = "cancelled"
                    record.error = None
                    record.progress_description = "Cancelled"
                    return
                record.result = future.result()
                record.status = "succeeded"
                record.error = None
                record.progress_current = record.progress_total
                record.progress_description = "Completed"
            except SDCMapJobCancelledError:
                record.status = "cancelled"
                record.error = None
                record.progress_description = "Cancelled"
            except Exception as exc:  # pragma: no cover - exercised by API tests
                record.status = "failed"
                record.error = str(exc)
                if not record.progress_description or record.progress_description.lower() in {
                    "queued",
                    "running",
                    "cancelling",
                }:
                    record.progress_description = "Failed"
            finally:
                record.completed_at = datetime.now(timezone.utc)
                self._futures.pop(job_id, None)
                self._cancel_events.pop(job_id, None)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def register_dataset(self, dataframe: pd.DataFrame, filename: str) -> DatasetRecord:
        """Register an uploaded dataset for later job submission by selected columns."""
        dataset_id = uuid4().hex
        record = DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            created_at=datetime.now(timezone.utc),
            dataframe=dataframe,
        )
        with self._lock:
            self._datasets[dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        with self._lock:
            return self._datasets.get(dataset_id)

    def register_map_upload(
        self,
        *,
        kind: str,
        filename: str,
        content: bytes,
        metadata: dict | None = None,
    ) -> MapUploadRecord:
        """Persist an uploaded map asset (CSV/NetCDF) to a temporary file."""
        upload_id = uuid4().hex
        safe_kind = str(kind or "asset").strip().lower()
        suffix = Path(filename or "").suffix or (".csv" if safe_kind == "driver" else ".nc")
        path = self._tmpdir_path / f"{safe_kind}_{upload_id}{suffix}"
        path.write_bytes(content)
        record = MapUploadRecord(
            upload_id=upload_id,
            kind=safe_kind,
            filename=filename or path.name,
            created_at=datetime.now(timezone.utc),
            path=path,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._map_uploads[upload_id] = record
        return record

    def get_map_upload(self, upload_id: str) -> MapUploadRecord | None:
        with self._lock:
            return self._map_uploads.get(upload_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._tmpdir.cleanup()
