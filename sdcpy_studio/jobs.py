"""Asynchronous job and dataset management for sdcpy-studio."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

import pandas as pd

from sdcpy_studio.schemas import SDCJobRequest, SDCMapJobRequest
from sdcpy_studio.service import run_sdc_job, run_sdc_map_job


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


class JobManager:
    """Asynchronous job manager using a thread pool plus in-memory dataset storage."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}
        self._datasets: dict[str, DatasetRecord] = {}

    def submit(self, request: SDCJobRequest) -> JobRecord:
        """Submit a new background computation."""
        return self._submit_with_runner(request, run_sdc_job)

    def submit_map(self, request: SDCMapJobRequest) -> JobRecord:
        """Submit a new background computation for SDC map mode."""
        return self._submit_with_runner(request, run_sdc_map_job)

    def _submit_with_runner(
        self,
        request: SDCJobRequest | SDCMapJobRequest,
        runner,
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

        with self._lock:
            self._jobs[job_id] = record
            record.status = "running"
            record.started_at = datetime.now(timezone.utc)
            if record.progress_description.lower() == "queued":
                record.progress_description = "Starting job"

        def _progress_update(current: int, total: int, description: str) -> None:
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec.progress_current = int(max(0, current))
                rec.progress_total = int(max(1, total))
                rec.progress_description = description

        future = self._executor.submit(runner, request.model_dump(mode="python"), _progress_update)

        with self._lock:
            self._futures[job_id] = future

        future.add_done_callback(lambda fut, jid=job_id: self._finalize(jid, fut))
        return record

    def _finalize(self, job_id: str, future: Future) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.completed_at = datetime.now(timezone.utc)
            try:
                record.result = future.result()
                record.status = "succeeded"
                record.progress_current = record.progress_total
                record.progress_description = "Completed"
            except Exception as exc:  # pragma: no cover - exercised by API tests
                record.status = "failed"
                record.error = str(exc)
                if not record.progress_description or record.progress_description.lower() in {
                    "queued",
                    "running",
                }:
                    record.progress_description = "Failed"

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

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
