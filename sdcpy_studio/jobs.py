"""Asynchronous job management for the SDC web API."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from sdcpy_studio.schemas import SDCJobRequest
from sdcpy_studio.service import run_sdc_job


@dataclass
class JobRecord:
    """In-memory record for one submitted SDC job."""

    job_id: str
    status: str
    request: SDCJobRequest
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict | None = None
    error: str | None = None


class JobManager:
    """Simple asynchronous job manager using a process pool."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._lock = RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}

    def submit(self, request: SDCJobRequest) -> JobRecord:
        """Submit a new background computation."""
        job_id = uuid4().hex
        created_at = datetime.now(timezone.utc)
        record = JobRecord(
            job_id=job_id,
            status="queued",
            request=request,
            created_at=created_at,
        )

        with self._lock:
            self._jobs[job_id] = record

        future = self._executor.submit(run_sdc_job, request.model_dump(mode="python"))

        with self._lock:
            record.status = "running"
            record.started_at = datetime.now(timezone.utc)
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
            except Exception as exc:  # pragma: no cover - exercised by API tests
                record.status = "failed"
                record.error = str(exc)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
