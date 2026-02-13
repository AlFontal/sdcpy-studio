"""Pydantic schemas for the sdcpy-studio API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_SERIES_LENGTH = 2000
MIN_SERIES_LENGTH = 20


class SDCJobRequest(BaseModel):
    """Request body for launching an SDC computation job."""

    ts1: list[float] = Field(..., description="First time series values.")
    ts2: list[float] = Field(..., description="Second time series values.")

    fragment_size: int = Field(24, ge=2, le=1024)
    heatmap_step: int = Field(1, ge=1, le=64)
    n_permutations: int = Field(99, ge=1, le=999)
    method: Literal["pearson", "spearman", "kendall"] = "pearson"

    two_tailed: bool = True
    permutations: bool = True
    min_lag: int = -120
    max_lag: int = 120

    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    max_memory_gb: float = Field(1.0, ge=0.1, le=16.0)
    ts1_label: str = "TS1"
    ts2_label: str = "TS2"
    index_values: list[str] | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> SDCJobRequest:
        if len(self.ts1) != len(self.ts2):
            raise ValueError("`ts1` and `ts2` must have the same length.")
        if len(self.ts1) < MIN_SERIES_LENGTH:
            raise ValueError(f"Each series must have at least {MIN_SERIES_LENGTH} points.")
        if len(self.ts1) > MAX_SERIES_LENGTH:
            raise ValueError(
                f"Series are too long for interactive mode (max {MAX_SERIES_LENGTH} points)."
            )
        if self.fragment_size >= len(self.ts1):
            raise ValueError("`fragment_size` must be smaller than the series length.")
        if self.min_lag > self.max_lag:
            raise ValueError("`min_lag` must be <= `max_lag`.")
        if self.index_values is not None and len(self.index_values) != len(self.ts1):
            raise ValueError("`index_values` must match the series length when provided.")
        return self


class SDCJobFromDatasetRequest(BaseModel):
    """Request body for launching SDC from an inspected dataset."""

    dataset_id: str
    ts1_column: str
    ts2_column: str
    date_column: str | None = None

    fragment_size: int = Field(24, ge=2, le=1024)
    heatmap_step: int = Field(1, ge=1, le=64)
    n_permutations: int = Field(99, ge=1, le=999)
    method: Literal["pearson", "spearman", "kendall"] = "pearson"
    two_tailed: bool = True
    permutations: bool = True
    min_lag: int = -120
    max_lag: int = 120
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    max_memory_gb: float = Field(1.0, ge=0.1, le=16.0)


class JobSubmissionResponse(BaseModel):
    """Response returned immediately after creating a job."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    message: str


class JobProgressResponse(BaseModel):
    """Progress payload for running jobs."""

    current: int
    total: int
    description: str


class JobStatusResponse(BaseModel):
    """Current state of an asynchronous job."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    progress: JobProgressResponse | None = None


class DatasetInspectResponse(BaseModel):
    """Response payload for dataset inspection."""

    dataset_id: str
    filename: str
    n_rows: int
    n_columns: int
    columns: list[str]
    numeric_columns: list[str]
    datetime_columns: list[str]
    suggested_date_column: str | None = None
    preview_rows: list[dict]


class MatrixPayload(BaseModel):
    """Compact matrix payload for frontend heatmap rendering."""

    x: list[int]
    y: list[int]
    z: list[list[float | None]]


class SeriesPayload(BaseModel):
    """Series payload used by the interactive 2-way explorer."""

    index: list[int | str]
    ts1: list[float]
    ts2: list[float]


class JobResultPayload(BaseModel):
    """Result payload returned for completed jobs."""

    summary: dict
    series: SeriesPayload
    matrix_r: MatrixPayload
    matrix_p: MatrixPayload
    strongest_links: list[dict]
    notes: list[str]
    runtime_seconds: float


class JobResultResponse(BaseModel):
    """Top-level completed job response."""

    job_id: str
    status: Literal["succeeded"]
    result: JobResultPayload
