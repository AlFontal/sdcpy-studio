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
    n_permutations: int = Field(99, ge=1, le=999)
    method: Literal["pearson", "spearman", "kendall"] = "pearson"

    two_tailed: bool = True
    permutations: bool = True
    min_lag: int = -120
    max_lag: int = 120

    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    max_memory_gb: float = Field(1.0, ge=0.1, le=16.0)

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
        return self


class JobSubmissionResponse(BaseModel):
    """Response returned immediately after creating a job."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    message: str


class JobStatusResponse(BaseModel):
    """Current state of an asynchronous job."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class MatrixPayload(BaseModel):
    """Compact matrix payload for frontend heatmap rendering."""

    x: list[int]
    y: list[int]
    z: list[list[float | None]]


class SeriesPayload(BaseModel):
    """Series payload used by the interactive 2-way explorer."""

    index: list[int]
    ts1: list[float]
    ts2: list[float]


class RangesPanelPayload(BaseModel):
    """Compact payload for the side `get_ranges_df` diagnostic panel."""

    bin_center: list[float]
    positive_freq: list[float]
    negative_freq: list[float]
    ns_freq: list[float]


class JobResultPayload(BaseModel):
    """Result payload returned for completed jobs."""

    summary: dict
    series: SeriesPayload
    matrix_r: MatrixPayload
    matrix_p: MatrixPayload
    ranges_panel: RangesPanelPayload
    strongest_links: list[dict]
    notes: list[str]
    runtime_seconds: float


class JobResultResponse(BaseModel):
    """Top-level completed job response."""

    job_id: str
    status: Literal["succeeded"]
    result: JobResultPayload
