"""FastAPI application exposing interactive SDC analysis."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sdcpy_studio.jobs import JobManager
from sdcpy_studio.schemas import (
    JobResultResponse,
    JobStatusResponse,
    JobSubmissionResponse,
    SDCJobRequest,
)
from sdcpy_studio.service import build_synthetic_example, parse_series_csv

BASE_DIR = Path(__file__).resolve().parent


def _job_status_payload(job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    """Create the FastAPI app instance."""

    manager = job_manager or JobManager(max_workers=int(os.getenv("SDCPY_STUDIO_MAX_WORKERS", "2")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.job_manager = manager
        yield
        manager.shutdown()

    app = FastAPI(
        title="sdcpy Studio",
        description="Interactive Scale-Dependent Correlation analysis for non-technical users.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.job_manager = manager

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {"title": "sdcpy Studio"})

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/examples/synthetic")
    async def synthetic_example() -> dict[str, list[float]]:
        return build_synthetic_example()

    @app.post("/api/v1/jobs/sdc", response_model=JobSubmissionResponse)
    async def submit_job(payload: SDCJobRequest) -> JobSubmissionResponse:
        job = app.state.job_manager.submit(payload)
        return JobSubmissionResponse(
            job_id=job.job_id,
            status=job.status,
            message="Job submitted. Poll /api/v1/jobs/{job_id} for completion.",
        )

    @app.post("/api/v1/jobs/sdc/csv", response_model=JobSubmissionResponse)
    async def submit_job_csv(
        ts1_file: Annotated[UploadFile, File(...)],
        ts2_file: Annotated[UploadFile, File(...)],
        fragment_size: Annotated[int, Form()] = 24,
        n_permutations: Annotated[int, Form()] = 99,
        method: Annotated[str, Form()] = "pearson",
        two_tailed: Annotated[bool, Form()] = True,
        permutations: Annotated[bool, Form()] = True,
        min_lag: Annotated[int, Form()] = -120,
        max_lag: Annotated[int, Form()] = 120,
        alpha: Annotated[float, Form()] = 0.05,
        max_memory_gb: Annotated[float, Form()] = 1.0,
    ) -> JobSubmissionResponse:
        ts1 = parse_series_csv(await ts1_file.read())
        ts2 = parse_series_csv(await ts2_file.read())

        payload = SDCJobRequest(
            ts1=ts1,
            ts2=ts2,
            fragment_size=fragment_size,
            n_permutations=n_permutations,
            method=method,
            two_tailed=two_tailed,
            permutations=permutations,
            min_lag=min_lag,
            max_lag=max_lag,
            alpha=alpha,
            max_memory_gb=max_memory_gb,
        )
        job = app.state.job_manager.submit(payload)

        return JobSubmissionResponse(
            job_id=job.job_id,
            status=job.status,
            message="CSV job submitted. Poll /api/v1/jobs/{job_id} for completion.",
        )

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str) -> JobStatusResponse:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_status_payload(job)

    @app.get("/api/v1/jobs/{job_id}/result", response_model=JobResultResponse)
    async def get_job_result(job_id: str) -> JobResultResponse:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.status == "failed":
            raise HTTPException(status_code=422, detail=job.error or "Job failed.")
        if job.status != "succeeded" or job.result is None:
            raise HTTPException(status_code=409, detail="Job still running.")

        return JobResultResponse(job_id=job_id, status="succeeded", result=job.result)

    return app


def run() -> None:
    """Entrypoint for the `sdcpy-studio` console script."""
    import uvicorn

    uvicorn.run(
        "sdcpy_studio.main:create_app",
        factory=True,
        host=os.getenv("SDCPY_STUDIO_HOST", "0.0.0.0"),
        port=int(os.getenv("SDCPY_STUDIO_PORT", "8000")),
        reload=bool(int(os.getenv("SDCPY_STUDIO_RELOAD", "0"))),
    )
