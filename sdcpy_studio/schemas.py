"""Pydantic schemas for the sdcpy-studio API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MIN_SERIES_LENGTH = 20
JobStatusLiteral = Literal["queued", "running", "cancelling", "cancelled", "succeeded", "failed"]


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


class ManualEventSelection(BaseModel):
    """Explicit user-selected driver events for SDC Map."""

    selected_positive_dates: list[str] = Field(default_factory=list)
    selected_negative_dates: list[str] = Field(default_factory=list)


class SDCMapJobRequest(BaseModel):
    """Request body for launching a beta SDC map computation job."""

    driver_dataset: str = "pdo"
    field_dataset: str = "ncep_air"
    driver_source_type: Literal["catalog", "upload"] = "catalog"
    field_source_type: Literal["catalog", "upload"] = "catalog"
    driver_upload_id: str | None = None
    driver_date_column: str | None = None
    driver_value_column: str | None = None
    field_upload_id: str | None = None
    field_variable: str | None = None
    field_dimension_selections: dict[str, str] = Field(default_factory=dict)
    manual_event_selection: ManualEventSelection | None = None
    # Server-populated resolved file paths for uploaded assets.
    driver_upload_path: str | None = None
    field_upload_path: str | None = None
    driver_upload_filename: str | None = None
    field_upload_filename: str | None = None

    correlation_width: int = Field(12, ge=2, le=256)
    n_positive_peaks: int = Field(3, ge=0, le=24)
    n_negative_peaks: int = Field(3, ge=0, le=24)
    base_state_beta: float = Field(0.5, gt=0.0, le=2.0)
    n_permutations: int = Field(49, ge=1, le=999)
    two_tailed: bool = False
    min_lag: int = -6
    max_lag: int = 6
    alpha: float = Field(0.05, gt=0.0, lt=1.0)

    time_start: str | None = None
    time_end: str | None = None

    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    lat_stride: int = Field(1, ge=1, le=8)
    lon_stride: int = Field(1, ge=1, le=8)

    @model_validator(mode="after")
    def _check_consistency(self) -> SDCMapJobRequest:
        if not str(self.driver_dataset).strip():
            raise ValueError("`driver_dataset` must not be empty.")
        if not str(self.field_dataset).strip():
            raise ValueError("`field_dataset` must not be empty.")
        if self.driver_source_type == "upload":
            if not self.driver_upload_id:
                raise ValueError("`driver_upload_id` is required when `driver_source_type` is 'upload'.")
            if not self.driver_date_column:
                raise ValueError(
                    "`driver_date_column` is required when `driver_source_type` is 'upload'."
                )
            if not self.driver_value_column:
                raise ValueError(
                    "`driver_value_column` is required when `driver_source_type` is 'upload'."
                )
        if self.field_source_type == "upload":
            if not self.field_upload_id:
                raise ValueError("`field_upload_id` is required when `field_source_type` is 'upload'.")
            if not self.field_variable:
                raise ValueError("`field_variable` is required when `field_source_type` is 'upload'.")
        if self.min_lag > self.max_lag:
            raise ValueError("`min_lag` must be <= `max_lag`.")
        bounds = (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        provided_bounds = [value is not None for value in bounds]
        if any(provided_bounds) and not all(provided_bounds):
            raise ValueError(
                "`lat_min`, `lat_max`, `lon_min`, and `lon_max` must be provided together, or all omitted."
            )
        if all(provided_bounds):
            if float(self.lat_min) >= float(self.lat_max):
                raise ValueError("`lat_min` must be < `lat_max`.")
            if float(self.lon_min) >= float(self.lon_max):
                raise ValueError("`lon_min` must be < `lon_max`.")
        if self.time_start and self.time_end and self.time_start > self.time_end:
            raise ValueError("`time_start` must be <= `time_end`.")
        return self


class JobSubmissionResponse(BaseModel):
    """Response returned immediately after creating a job."""

    job_id: str
    status: JobStatusLiteral
    message: str


class JobProgressResponse(BaseModel):
    """Progress payload for running jobs."""

    current: int
    total: int
    description: str


class JobStatusResponse(BaseModel):
    """Current state of an asynchronous job."""

    job_id: str
    status: JobStatusLiteral
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
    detected_delimiter: str | None = None
    delimiter_name: str | None = None
    warnings: list[InspectWarning] = Field(default_factory=list)
    preview_rows: list[dict[str, str]]


class InspectWarning(BaseModel):
    """Structured warning surfaced during upload inspection."""

    code: str
    message: str
    columns: list[str] = Field(default_factory=list)


class DriverDefaults(BaseModel):
    """Suggested default time window derived from the uploaded driver series."""

    time_start: str | None = None
    time_end: str | None = None
    driver_min_date: str | None = None
    driver_max_date: str | None = None
    n_points: int | None = None


class DriverEventPreviewItem(BaseModel):
    """Preview of one detected driver event."""

    index: int | None = None
    date: str
    value: float
    sign: Literal["positive", "negative"]
    source: Literal["auto", "manual"] = "auto"


class DriverEventCatalog(BaseModel):
    """Detected driver events and base-state summary used by SDC Map."""

    selected_positive: list[DriverEventPreviewItem] = Field(default_factory=list)
    selected_negative: list[DriverEventPreviewItem] = Field(default_factory=list)
    ignored_positive: list[DriverEventPreviewItem] = Field(default_factory=list)
    ignored_negative: list[DriverEventPreviewItem] = Field(default_factory=list)
    base_state_threshold: float | None = None
    base_state_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    selection_mode: Literal["auto", "manual"] = "auto"


class FieldDims(BaseModel):
    """Normalized field dimensions exposed to the UI."""

    time: int = 0
    lat: int = 0
    lon: int = 0


class FieldNormalizationInfo(BaseModel):
    """Normalization details applied to a compatible field variable."""

    original_dims: list[str] = Field(default_factory=list)
    squeezed_dims: list[str] = Field(default_factory=list)
    selected_dimensions: dict[str, str] = Field(default_factory=dict)


class IncompatibleVariable(BaseModel):
    """Reason a NetCDF variable could not be used as a map field."""

    name: str
    reason: str


class FieldSelectorOption(BaseModel):
    """Single selectable coordinate value for an extra NetCDF dimension."""

    value: str
    label: str


class FieldSelectorDefinition(BaseModel):
    """Selector required to use a variable with non-spatial extra dimensions."""

    dimension: str
    label: str
    options: list[FieldSelectorOption] = Field(default_factory=list)
    suggested_value: str | None = None


class FieldVariableOption(BaseModel):
    """Variable-level compatibility metadata for uploaded NetCDF files."""

    name: str
    selectors: list[FieldSelectorDefinition] = Field(default_factory=list)
    normalization: FieldNormalizationInfo = Field(default_factory=FieldNormalizationInfo)
    warnings: list[InspectWarning] = Field(default_factory=list)


class SDCMapDriverUploadInspectResponse(BaseModel):
    """Inspection payload for a custom SDC map driver CSV upload."""

    upload_id: str
    filename: str
    n_rows: int
    columns: list[str]
    numeric_columns: list[str]
    datetime_columns: list[str]
    suggested_date_column: str | None = None
    suggested_value_column: str | None = None
    detected_delimiter: str | None = None
    delimiter_name: str | None = None
    rejected_numeric_columns: list[str] = Field(default_factory=list)
    warnings: list[InspectWarning] = Field(default_factory=list)
    preview_rows: list[dict[str, str]]
    defaults: DriverDefaults
    event_catalog: DriverEventCatalog = Field(default_factory=DriverEventCatalog)


class SDCMapFieldUploadInspectResponse(BaseModel):
    """Inspection payload for a custom SDC map field NetCDF upload."""

    upload_id: str
    filename: str
    variables: list[str]
    compatible_variables: list[str]
    variable_options: list[FieldVariableOption] = Field(default_factory=list)
    incompatible_variables: list[IncompatibleVariable] = Field(default_factory=list)
    suggested_variable: str | None = None
    dims: FieldDims
    normalization: FieldNormalizationInfo = Field(default_factory=FieldNormalizationInfo)
    warnings: list[InspectWarning] = Field(default_factory=list)
    time_start: str | None = None
    time_end: str | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None


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
    lag_matrix_r: MatrixPayload | None = None
    lag_matrix_p: MatrixPayload | None = None
    lag_default: int | None = None
    strongest_links: list[dict]
    notes: list[str]
    runtime_seconds: float


class SDCMapJobResultPayload(BaseModel):
    """Result payload returned for completed SDC map jobs."""

    summary: dict
    event_catalog: DriverEventCatalog = Field(default_factory=DriverEventCatalog)
    class_results: dict[str, dict] = Field(default_factory=dict)
    notes: list[str]
    runtime_seconds: float
    figure_png_base64: str
    layer_maps: dict | None = None
    download_formats: list[str]


class SDCMapExploreResultPayload(BaseModel):
    """Payload returned by the pre-run map exploration endpoint."""

    summary: dict
    event_catalog: DriverEventCatalog = Field(default_factory=DriverEventCatalog)
    time_index: list[str]
    driver_values: list[float | None]
    lat: list[float]
    lon: list[float]
    field_frames: list[list[list[float | None]]]
    coastline: dict[str, list[float | None]]


class SDCMapExploreResponse(BaseModel):
    """Top-level exploration response."""

    status: Literal["ready"]
    result: SDCMapExploreResultPayload


class SDCMapDriverPreviewPayload(BaseModel):
    """Lightweight driver-only preview used by the map configuration UI."""

    summary: dict
    event_catalog: DriverEventCatalog = Field(default_factory=DriverEventCatalog)
    time_index: list[str]
    driver_values: list[float | None]


class SDCMapDriverPreviewResponse(BaseModel):
    """Top-level response for the driver preview endpoint."""

    status: Literal["ready"]
    result: SDCMapDriverPreviewPayload


class JobResultResponse(BaseModel):
    """Top-level completed job response."""

    job_id: str
    status: Literal["succeeded"]
    result: JobResultPayload


class SDCMapJobResultResponse(BaseModel):
    """Top-level completed map-job response."""

    job_id: str
    status: Literal["succeeded"]
    result: SDCMapJobResultPayload
