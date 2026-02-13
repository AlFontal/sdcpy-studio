"""FastAPI application exposing interactive SDC analysis."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sdcpy_studio.jobs import JobManager
from sdcpy_studio.schemas import (
    DatasetInspectResponse,
    JobProgressResponse,
    JobResultResponse,
    JobStatusResponse,
    JobSubmissionResponse,
    SDCJobFromDatasetRequest,
    SDCJobRequest,
)
from sdcpy_studio.service import (
    build_job_request_from_dataset,
    build_synthetic_example,
    export_job_artifact,
    inspect_dataset_csv,
    parse_series_csv,
)

BASE_DIR = Path(__file__).resolve().parent
ONI_EXAMPLE_DATASET = BASE_DIR / "data" / "oni_temp_sa.csv"


def _job_status_payload(job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=getattr(job, "error", None),
        progress=JobProgressResponse(
            current=int(getattr(job, "progress_current", 0)),
            total=int(getattr(job, "progress_total", 1)),
            description=str(getattr(job, "progress_description", "Unknown")),
        ),
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
        title="SDCpy Studio",
        description="A GUI for Interactive Scale-Dependent Correlation analysis.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.job_manager = manager

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {"title": "SDCpy Studio"})

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health")
    async def health_root() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/examples/synthetic")
    async def synthetic_example() -> dict[str, list[float]]:
        return build_synthetic_example()

    @app.get("/api/v1/examples/oni-dataset", response_model=DatasetInspectResponse)
    async def oni_dataset_example() -> DatasetInspectResponse:
        if not ONI_EXAMPLE_DATASET.exists():
            raise HTTPException(status_code=500, detail="Bundled ONI example dataset not found.")
        try:
            frame, metadata = inspect_dataset_csv(
                ONI_EXAMPLE_DATASET.read_bytes(),
                filename=ONI_EXAMPLE_DATASET.name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        dataset = app.state.job_manager.register_dataset(frame, filename=ONI_EXAMPLE_DATASET.name)
        return DatasetInspectResponse(dataset_id=dataset.dataset_id, **metadata)

    @app.post("/api/v1/datasets/inspect", response_model=DatasetInspectResponse)
    async def inspect_dataset(dataset_file: Annotated[UploadFile, File(...)]) -> DatasetInspectResponse:
        filename = dataset_file.filename or "uploaded.csv"
        content = await dataset_file.read()
        try:
            frame, metadata = inspect_dataset_csv(content, filename=filename)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        dataset = app.state.job_manager.register_dataset(frame, filename=filename)
        return DatasetInspectResponse(dataset_id=dataset.dataset_id, **metadata)

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
        heatmap_step: Annotated[int, Form()] = 1,
        n_permutations: Annotated[int, Form()] = 99,
        method: Annotated[str, Form()] = "pearson",
        two_tailed: Annotated[bool, Form()] = True,
        permutations: Annotated[bool, Form()] = True,
        min_lag: Annotated[int, Form()] = -120,
        max_lag: Annotated[int, Form()] = 120,
        alpha: Annotated[float, Form()] = 0.05,
        max_memory_gb: Annotated[float, Form()] = 1.0,
    ) -> JobSubmissionResponse:
        try:
            ts1 = parse_series_csv(await ts1_file.read())
            ts2 = parse_series_csv(await ts2_file.read())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        payload = SDCJobRequest(
            ts1=ts1,
            ts2=ts2,
            fragment_size=fragment_size,
            heatmap_step=heatmap_step,
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

    @app.post("/api/v1/jobs/sdc/dataset", response_model=JobSubmissionResponse)
    async def submit_job_dataset(payload: SDCJobFromDatasetRequest) -> JobSubmissionResponse:
        dataset = app.state.job_manager.get_dataset(payload.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found or expired.")

        try:
            request = build_job_request_from_dataset(dataset.dataframe, payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        job = app.state.job_manager.submit(request)
        return JobSubmissionResponse(
            job_id=job.job_id,
            status=job.status,
            message="Dataset job submitted. Poll /api/v1/jobs/{job_id} for completion.",
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

    @app.get("/api/v1/jobs/{job_id}/download/{fmt}")
    async def download_result(job_id: str, fmt: str) -> Response:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.status == "failed":
            raise HTTPException(status_code=422, detail=job.error or "Job failed.")
        if job.status != "succeeded" or job.result is None:
            raise HTTPException(status_code=409, detail="Job still running.")

        try:
            payload, media_type, filename = export_job_artifact(job.result, fmt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(
            content=payload,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
        )

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
