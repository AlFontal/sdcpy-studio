"""Service-layer helpers for sdcpy-studio computations and exports."""

from __future__ import annotations

import base64
import io
import os
import re
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import sdcpy.core as sdc_core
from sdcpy import compute_sdc
from sdcpy.scale_dependent_correlation import SDCAnalysis

from sdcpy_studio.schemas import SDCJobFromDatasetRequest, SDCJobRequest, SDCMapJobRequest

MAX_STRONGEST_LINKS = 100
_MAP_DATA_PATH_CACHE: dict[tuple[str, str, str], dict[str, Path]] = {}
_MAP_DATA_PATH_CACHE_LOCK = Lock()
_FIELD_BOUNDS_CACHE: dict[str, dict[str, float]] = {}
_FIELD_BOUNDS_CACHE_LOCK = Lock()
_DRIVER_COVERAGE_CACHE: dict[str, dict[str, object]] = {}
_DRIVER_COVERAGE_CACHE_LOCK = Lock()
_FIELD_COVERAGE_CACHE: dict[str, dict[str, object]] = {}
_FIELD_COVERAGE_CACHE_LOCK = Lock()
_RD_BU_WHITE_CENTER: tuple[tuple[float, str], ...] = (
    (0.0, "#053061"),
    (0.125, "#2166ac"),
    (0.25, "#4393c3"),
    (0.375, "#92c5de"),
    (0.5, "#ffffff"),
    (0.625, "#f4a582"),
    (0.75, "#d6604d"),
    (0.875, "#b2182b"),
    (1.0, "#67001f"),
)
_MAP_LAYER_DEFS: tuple[dict[str, object], ...] = (
    {
        "key": "corr_mean",
        "label": "Mean extreme correlation",
        "colorscale": [[stop, color] for stop, color in _RD_BU_WHITE_CENTER],
        "zmin": -1.0,
        "zmax": 1.0,
    },
    {
        "key": "lag_mean",
        "label": "Mean lag (months)",
        "colorscale": "RdYlBu",
        "zmin": None,
        "zmax": None,
    },
    {
        "key": "driver_rel_time_mean",
        "label": "Mean driver-relative time (months)",
        "colorscale": "BrBG",
        "zmin": None,
        "zmax": None,
    },
)


def _sanitize_filename_token(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    if not cleaned:
        return fallback
    return cleaned.lower()[:64]


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _resolve_map_cache_dir() -> Path:
    return Path(
        os.getenv(
            "SDCPY_STUDIO_SDCPY_MAP_DATA_DIR",
            str(Path.home() / ".cache" / "sdcpy-studio" / "sdcpy-map"),
        )
    )


def _map_download_policy() -> tuple[bool, bool, bool]:
    refresh = _read_bool_env("SDCPY_STUDIO_SDCPY_MAP_REFRESH", default=False)
    verify_remote = _read_bool_env("SDCPY_STUDIO_SDCPY_MAP_VERIFY_REMOTE", default=False)
    offline = _read_bool_env("SDCPY_STUDIO_SDCPY_MAP_OFFLINE", default=False)
    if offline and refresh:
        raise ValueError(
            "Invalid cache policy: SDCPY_STUDIO_SDCPY_MAP_OFFLINE=1 cannot be combined with "
            "SDCPY_STUDIO_SDCPY_MAP_REFRESH=1."
        )
    return refresh, verify_remote, offline


def _download_map_asset(download_if_missing, url: str, destination: Path) -> Path:
    refresh, verify_remote, offline = _map_download_policy()
    kwargs = {
        "refresh": refresh,
        "verify_remote": verify_remote,
        "offline": offline,
    }
    try:
        return download_if_missing(url, destination, **kwargs)
    except TypeError:
        # Backward compatibility with older sdcpy-map versions.
        if destination.exists() and destination.stat().st_size > 0 and not refresh and not verify_remote:
            return destination
        if refresh and destination.exists():
            destination.unlink()
        if offline and not destination.exists():
            raise ValueError(
                f"Offline mode is enabled and required dataset is missing: {destination.name}"
            ) from None
        return download_if_missing(url, destination)


def _full_driver_config():
    from sdcpy_map import SDCMapConfig

    return SDCMapConfig(
        # Keep timestamps within pandas ns-safe bounds.
        time_start="1900-01-01",
        time_end="2261-12-31",
        lat_min=-90.0,
        lat_max=90.0,
        lon_min=-180.0,
        lon_max=180.0,
        lat_stride=1,
        lon_stride=1,
    )


def _as_calendar_date(value: object) -> date | None:
    """Parse common timestamp-like objects into a plain calendar date."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return _as_calendar_date(np.datetime_as_string(value, unit="D"))
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        try:
            return date(int(value.year), int(value.month), int(value.day))
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    if len(text) == 7 and text.count("-") == 1:
        text = f"{text}-01"
    elif len(text) == 4 and text.isdigit():
        text = f"{text}-01-01"
    try:
        return date.fromisoformat(text)
    except ValueError:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
        if not match:
            return None
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None


def _shift_years(value: date, years: int) -> date:
    year = int(value.year) + int(years)
    month = int(value.month)
    day = int(value.day)
    if month == 2 and day == 29:
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        day = 29 if is_leap else 28
    while True:
        try:
            return date(year, month, day)
        except ValueError:
            day -= 1
            if day < 1:
                return date(year, month, 1)


def _nearest_time_index(time_values: np.ndarray, target_date: str) -> int:
    if time_values.size == 0:
        return 0
    target = _as_calendar_date(target_date)
    if target is None:
        return int(time_values.size // 2)

    best_idx = 0
    best_dist: int | None = None
    for idx, raw in enumerate(time_values):
        parsed = _as_calendar_date(raw)
        if parsed is None:
            continue
        dist = abs((parsed - target).days)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return int(best_idx)


def _serialize_time_labels(time_values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for raw in time_values:
        parsed = _as_calendar_date(raw)
        labels.append(parsed.isoformat() if parsed is not None else str(raw))
    return labels


def _derive_driver_peak_window(driver: pd.Series, years: int = 3) -> dict[str, object]:
    clean = driver.dropna().sort_index()
    if clean.empty:
        raise ValueError("Driver dataset does not contain valid values.")

    peak_date = _as_calendar_date(clean.idxmax())
    parsed_index_dates = [_as_calendar_date(value) for value in clean.index]
    valid_dates = [value for value in parsed_index_dates if value is not None]
    if peak_date is None or not valid_dates:
        raise ValueError("Driver dataset does not contain parseable datetime values.")

    min_date = min(valid_dates)
    max_date = max(valid_dates)
    window_start = max(min_date, _shift_years(peak_date, -years))
    window_end = min(max_date, _shift_years(peak_date, years))
    if window_start >= window_end:
        window_start = min_date
        window_end = max_date

    return {
        "peak_date": peak_date.isoformat(),
        "time_start": window_start.isoformat(),
        "time_end": window_end.isoformat(),
        "driver_min_date": min_date.isoformat(),
        "driver_max_date": max_date.isoformat(),
        "peak_value": float(clean.max()),
    }


def _emit_progress(
    hook: Callable[[int, int, str], None] | None,
    current: int,
    total: int,
    description: str,
) -> None:
    if hook is None:
        return
    hook(int(max(0, current)), int(max(1, total)), str(description))


def parse_series_csv(content: bytes) -> list[float]:
    """Parse a CSV payload into a numeric series.

    Accepted layouts:
    - one numeric column,
    - a `value` column,
    - a table where the first numeric-looking column is selected.
    """
    frame = pd.read_csv(io.BytesIO(content))
    if frame.empty:
        raise ValueError("Uploaded CSV is empty.")

    if "value" in frame.columns:
        series = pd.to_numeric(frame["value"], errors="coerce")
    else:
        numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
        if len(frame.columns) == 1:
            series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        elif numeric_cols:
            series = pd.to_numeric(frame[numeric_cols[0]], errors="coerce")
        else:
            series = pd.to_numeric(frame.iloc[:, -1], errors="coerce")

    cleaned = series.dropna().astype(float)
    if cleaned.empty:
        raise ValueError("CSV does not contain valid numeric values.")
    return cleaned.to_list()


def inspect_dataset_csv(content: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """Inspect uploaded dataset and infer useful column types for the UI workflow."""
    frame = pd.read_csv(io.BytesIO(content))
    if frame.empty:
        raise ValueError("Uploaded dataset is empty.")

    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]

    datetime_columns: list[str] = []
    numeric_columns: list[str] = []

    for col in frame.columns:
        series = frame[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(col)
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            non_null = int(series.notna().sum())
            parsed_count = int(parsed.notna().sum())
            ratio = parsed_count / non_null if non_null else 0
            if parsed_count >= 3 and ratio >= 0.8:
                frame[col] = parsed
                datetime_columns.append(col)

    for col in frame.columns:
        series = frame[col]

        if col in datetime_columns:
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col)
            continue

        candidate = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        parsed_count = int(candidate.notna().sum())
        ratio = parsed_count / non_null if non_null else 0
        if parsed_count >= 3 and ratio >= 0.8:
            frame[col] = candidate
            numeric_columns.append(col)

    preview = (
        frame.head(8)
        .astype(object)
        .where(frame.head(8).notna(), "")
        .astype(str)
        .to_dict(orient="records")
    )

    metadata = {
        "filename": filename,
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "columns": [str(c) for c in frame.columns],
        "numeric_columns": numeric_columns,
        "datetime_columns": datetime_columns,
        "suggested_date_column": datetime_columns[0] if datetime_columns else None,
        "preview_rows": preview,
    }
    return frame, metadata


def build_job_request_from_dataset(
    dataframe: pd.DataFrame,
    request: SDCJobFromDatasetRequest,
) -> SDCJobRequest:
    """Create an SDC job request from selected dataset columns."""
    for col in (request.ts1_column, request.ts2_column):
        if col not in dataframe.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    frame = dataframe.copy()
    index_values: list[str] | None = None

    if request.date_column:
        if request.date_column not in frame.columns:
            raise ValueError(f"Date column '{request.date_column}' not found in dataset.")
        parsed = pd.to_datetime(frame[request.date_column], errors="coerce")
        frame = frame.assign(_date=parsed).dropna(subset=["_date"]).sort_values("_date")
        index_values = frame["_date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_list()

    ts1 = pd.to_numeric(frame[request.ts1_column], errors="coerce")
    ts2 = pd.to_numeric(frame[request.ts2_column], errors="coerce")

    valid_mask = ts1.notna() & ts2.notna()
    ts1_clean = ts1.loc[valid_mask].astype(float).to_numpy()
    ts2_clean = ts2.loc[valid_mask].astype(float).to_numpy()

    if index_values is not None:
        index_values = [index_values[i] for i, ok in enumerate(valid_mask.to_numpy()) if ok]

    return SDCJobRequest(
        ts1=ts1_clean.tolist(),
        ts2=ts2_clean.tolist(),
        fragment_size=request.fragment_size,
        heatmap_step=request.heatmap_step,
        n_permutations=request.n_permutations,
        method=request.method,
        two_tailed=request.two_tailed,
        permutations=request.permutations,
        min_lag=request.min_lag,
        max_lag=request.max_lag,
        alpha=request.alpha,
        max_memory_gb=request.max_memory_gb,
        ts1_label=request.ts1_column,
        ts2_label=request.ts2_column,
        index_values=index_values,
    )


def build_synthetic_example(n: int = 360) -> dict[str, list[float]]:
    """Build a synthetic pair with a transient shared component."""
    rng = np.random.default_rng(42)
    t = np.arange(n)
    envelope = ((t >= 90) & (t <= 260)).astype(float)
    phase = np.sin(2 * np.pi * t / 37)

    ts1 = 0.9 * envelope * phase + 0.45 * rng.normal(size=n)
    ts2 = 0.7 * envelope * np.roll(phase, 3) + 0.55 * rng.normal(size=n)

    return {
        "ts1": ts1.round(6).tolist(),
        "ts2": ts2.round(6).tolist(),
    }


def _downsample_matrix(matrix: pd.DataFrame, step: int = 1) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    stride = max(1, int(step))
    return matrix.iloc[::stride, ::stride]


def _downsample_matrix_columns(matrix: pd.DataFrame, step: int = 1) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    stride = max(1, int(step))
    return matrix.iloc[:, ::stride]


def _mask_matrix_outside_lag_window(
    r_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame,
    min_lag: int,
    max_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if r_matrix.empty:
        return r_matrix, p_matrix.reindex_like(r_matrix)

    rows = r_matrix.index.to_numpy(dtype=float)
    cols = r_matrix.columns.to_numpy(dtype=float)
    lag_grid = cols[np.newaxis, :] - rows[:, np.newaxis]
    valid_mask = (lag_grid >= float(min_lag)) & (lag_grid <= float(max_lag))

    r_masked = r_matrix.where(valid_mask, np.nan)
    p_masked = p_matrix.reindex_like(r_matrix).where(valid_mask, np.nan)
    return r_masked, p_masked


def _mask_lag_matrix_outside_start2_range(
    lag_r_matrix: pd.DataFrame,
    lag_p_matrix: pd.DataFrame,
    min_start2: int,
    max_start2: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lag_r_matrix.empty:
        return lag_r_matrix, lag_p_matrix.reindex_like(lag_r_matrix)

    lag_values = lag_r_matrix.index.to_numpy(dtype=float)
    start1_values = lag_r_matrix.columns.to_numpy(dtype=float)
    start2_grid = start1_values[np.newaxis, :] - lag_values[:, np.newaxis]
    valid_mask = (start2_grid >= float(min_start2)) & (start2_grid <= float(max_start2))

    r_masked = lag_r_matrix.where(valid_mask, np.nan)
    p_masked = lag_p_matrix.reindex_like(lag_r_matrix).where(valid_mask, np.nan)
    return r_masked, p_masked


def _matrix_payload(matrix: pd.DataFrame) -> dict:
    if matrix.empty:
        return {"x": [], "y": [], "z": []}

    arr = matrix.to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    z = [[None if np.isnan(value) else float(value) for value in row] for row in arr]

    return {
        "x": [int(v) for v in matrix.columns.to_numpy()],
        "y": [int(v) for v in matrix.index.to_numpy()],
        "z": z,
    }


def _serialize_artifacts(request: SDCJobRequest, sdc_df: pd.DataFrame) -> dict:
    return {
        "sdc_df": sdc_df.to_json(orient="split"),
        "ts1": request.ts1,
        "ts2": request.ts2,
        "index_values": request.index_values,
        "fragment_size": request.fragment_size,
        "heatmap_step": request.heatmap_step,
        "n_permutations": request.n_permutations,
        "method": request.method,
        "min_lag": request.min_lag,
        "max_lag": request.max_lag,
        "alpha": request.alpha,
        "ts1_label": request.ts1_label,
        "ts2_label": request.ts2_label,
    }


def _restore_analysis(artifacts: dict) -> tuple[SDCAnalysis, str, str, float, int, int]:
    sdc_df = pd.read_json(io.StringIO(artifacts["sdc_df"]), orient="split")
    ts1 = np.asarray(artifacts["ts1"], dtype=float)
    ts2 = np.asarray(artifacts["ts2"], dtype=float)
    index_values = artifacts.get("index_values")

    if index_values:
        try:
            index = pd.to_datetime(index_values)
        except Exception:
            index = pd.Index(index_values)
    else:
        index = pd.RangeIndex(start=0, stop=len(ts1), step=1)

    ts1_series = pd.Series(ts1, index=index)
    ts2_series = pd.Series(ts2, index=index)
    index_values = ts1_series.index.to_numpy()

    if "date_1" not in sdc_df.columns:
        sdc_df["date_1"] = sdc_df["start_1"].astype(int).map(lambda i: index_values[i])
    if "date_2" not in sdc_df.columns:
        sdc_df["date_2"] = sdc_df["start_2"].astype(int).map(lambda i: index_values[i])

    analysis = SDCAnalysis(
        ts1=ts1_series,
        ts2=ts2_series,
        fragment_size=int(artifacts["fragment_size"]),
        n_permutations=int(artifacts["n_permutations"]),
        method=str(artifacts["method"]),
        sdc_df=sdc_df,
    )
    return (
        analysis,
        str(artifacts.get("ts1_label", "TS1")),
        str(artifacts.get("ts2_label", "TS2")),
        float(artifacts.get("alpha", 0.05)),
        int(artifacts.get("min_lag", -999999)),
        int(artifacts.get("max_lag", 999999)),
    )


def _build_excel_bytes(analysis: SDCAnalysis) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        (
            analysis.sdc_df.dropna()
            .pivot(index="start_1", columns="start_2", values="r")
            .to_excel(writer, sheet_name="rs")
        )
        (
            analysis.sdc_df.dropna()
            .pivot(index="start_1", columns="start_2", values="p_value")
            .to_excel(writer, sheet_name="p_values")
        )
        (
            pd.DataFrame(
                {
                    "index": analysis.ts1.index.astype(str),
                    "ts1": analysis.ts1.to_numpy(),
                    "ts2": analysis.ts2.to_numpy(),
                }
            ).to_excel(writer, sheet_name="time_series", index=False)
        )
        pd.DataFrame(
            {
                "fragment_size": analysis.fragment_size,
                "n_permutations": analysis.n_permutations,
                "method": analysis.method,
            },
            index=[1],
        ).to_excel(writer, sheet_name="config", index=False)

    return buffer.getvalue()


def export_job_artifact(job_result: dict, fmt: str) -> tuple[bytes, str, str]:
    """Build downloadable output bytes from a finished job result."""
    artifacts = job_result.get("_artifacts")
    if not artifacts:
        raise ValueError("Download artifact metadata is unavailable for this job.")

    fmt_key = fmt.lower().strip()

    if fmt_key == "xlsx":
        analysis, ts1_label, ts2_label, *_ = _restore_analysis(artifacts)
        ts1_token = _sanitize_filename_token(ts1_label, "ts1")
        ts2_token = _sanitize_filename_token(ts2_label, "ts2")
        filename = f"sdc_{ts1_token}_{ts2_token}_{analysis.fragment_size}.xlsx"
        return (
            _build_excel_bytes(analysis),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename,
        )

    if fmt_key in {"png", "svg"}:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        analysis, ts1_label, ts2_label, alpha, min_lag, max_lag = _restore_analysis(artifacts)
        fig = analysis.combi_plot(
            alpha=alpha,
            xlabel=ts1_label,
            ylabel=ts2_label,
            min_lag=min_lag,
            max_lag=max_lag,
            figsize=(7, 7),
            dpi=250,
            wspace=0.08,
            hspace=0.08,
            n_ticks=6,
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt_key, dpi=250, bbox_inches="tight")
        plt.close(fig)

        ts1_token = _sanitize_filename_token(ts1_label, "ts1")
        ts2_token = _sanitize_filename_token(ts2_label, "ts2")
        media = "image/png" if fmt_key == "png" else "image/svg+xml"
        filename = f"sdc_{ts1_token}_{ts2_token}_{analysis.fragment_size}.{fmt_key}"
        return buffer.getvalue(), media, filename

    raise ValueError(f"Unsupported download format: {fmt}")


def export_sdc_map_artifact(job_result: dict, fmt: str) -> tuple[bytes, str, str]:
    """Build downloadable output bytes from a finished SDC map job result."""
    artifacts = job_result.get("_artifacts_map")
    if not artifacts:
        raise ValueError("Download artifact metadata is unavailable for this map job.")

    driver_token = _sanitize_filename_token(artifacts.get("driver_dataset", "driver"), "driver")
    field_token = _sanitize_filename_token(artifacts.get("field_dataset", "field"), "field")
    fragment_size = int(artifacts.get("fragment_size", 0))
    basename = f"sdcmap_{driver_token}_{field_token}_{fragment_size}"

    fmt_key = fmt.lower().strip()
    if fmt_key == "png":
        return artifacts["png"], "image/png", f"{basename}.png"
    if fmt_key == "nc":
        return artifacts["nc"], "application/x-netcdf", f"{basename}.nc"
    raise ValueError(f"Unsupported download format: {fmt}")


def _canonical_psl_download_url(url: str) -> str:
    """Normalize NOAA PSL fileServer links to downloads host for stability."""
    prefix = "https://psl.noaa.gov/thredds/fileServer/"
    if url.startswith(prefix):
        return f"https://downloads.psl.noaa.gov/{url[len(prefix):]}"
    return url


def _map_asset_filename(kind: str, key: str, url: str) -> str:
    """Build unique filenames to avoid collisions (e.g., multiple sst.mnmean.nc files)."""
    name = Path(urlparse(url).path).name or "dataset.bin"
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip()) if key else "asset"
    safe_kind = re.sub(r"[^A-Za-z0-9_.-]+", "_", kind.strip()) if kind else "asset"
    return f"{safe_kind}_{safe_key}_{name}"


def _serialize_optional_array(values: np.ndarray) -> list[float | None]:
    arr = np.asarray(values, dtype=float)
    return [float(v) if np.isfinite(v) else None for v in arr.tolist()]


def _serialize_optional_matrix(values: np.ndarray) -> list[list[float | None]]:
    arr = np.asarray(values, dtype=float)
    matrix: list[list[float | None]] = []
    for row in arr:
        matrix.append([float(v) if np.isfinite(v) else None for v in row.tolist()])
    return matrix


def _serialize_optional_cube(values: np.ndarray) -> list[list[list[float | None]]]:
    arr = np.asarray(values, dtype=float)
    cube: list[list[list[float | None]]] = []
    for frame in arr:
        cube.append(_serialize_optional_matrix(frame))
    return cube


def _build_map_netcdf_bytes(
    layers: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    attrs: dict[str, str | int | float],
) -> bytes:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ValueError(
            "Could not serialize map arrays as NetCDF because xarray is unavailable."
        ) from exc

    dataset = xr.Dataset(
        data_vars={
            key: (("lat", "lon"), np.asarray(values, dtype=float)) for key, values in layers.items()
        },
        coords={
            "lat": np.asarray(lats, dtype=float),
            "lon": np.asarray(lons, dtype=float),
        },
        attrs=dict(attrs),
    )

    for engine in ("h5netcdf", "netcdf4", "scipy", None):
        try:
            if engine is None:
                payload = dataset.to_netcdf()
            else:
                payload = dataset.to_netcdf(engine=engine)
            if isinstance(payload, bytes):
                return payload
            if isinstance(payload, memoryview):
                return payload.tobytes()
            return bytes(payload)
        except Exception:
            continue

    raise ValueError(
        "Could not serialize map arrays as NetCDF. Install one NetCDF backend (h5netcdf, netCDF4, or scipy)."
    )


def _iter_coastline_coords(geometry) -> list[list[tuple[float, float]]]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []

    geom_type = getattr(geometry, "geom_type", "")
    if geom_type in {"LineString", "LinearRing"}:
        coords = list(getattr(geometry, "coords", []))
        if not coords:
            return []
        return [[(float(x), float(y)) for x, y, *_ in coords]]
    if geom_type == "Polygon":
        exterior = getattr(geometry, "exterior", None)
        if exterior is None:
            return []
        coords = list(getattr(exterior, "coords", []))
        if not coords:
            return []
        return [[(float(x), float(y)) for x, y, *_ in coords]]
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return []

    out: list[list[tuple[float, float]]] = []
    for part in geoms:
        out.extend(_iter_coastline_coords(part))
    return out


def _serialize_coastline_trace(coastline) -> dict[str, list[float | None]]:
    lon_values: list[float | None] = []
    lat_values: list[float | None] = []
    for geom in getattr(coastline, "geometry", []):
        for coords in _iter_coastline_coords(geom):
            if not coords:
                continue
            for lon, lat in coords:
                lon_values.append(float(lon))
                lat_values.append(float(lat))
            lon_values.append(None)
            lat_values.append(None)
    if lon_values and lon_values[-1] is None:
        lon_values.pop()
        lat_values.pop()
    return {"lon": lon_values, "lat": lat_values}


def _build_map_layer_payload(
    layers: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline,
) -> dict:
    payload_layers: list[dict] = []
    for spec in _MAP_LAYER_DEFS:
        key = str(spec["key"])
        if key not in layers:
            continue
        payload_layers.append(
            {
                "key": key,
                "label": spec["label"],
                "colorscale": spec["colorscale"],
                "zmin": spec["zmin"],
                "zmax": spec["zmax"],
                "values": _serialize_optional_matrix(np.asarray(layers[key], dtype=float)),
            }
        )
    return {
        "lat": [float(v) for v in np.asarray(lats, dtype=float).tolist()],
        "lon": [float(v) for v in np.asarray(lons, dtype=float).tolist()],
        "coastline": _serialize_coastline_trace(coastline),
        "layers": payload_layers,
    }


def fetch_sdc_map_assets(
    data_dir: Path | str,
    driver_key: str,
    field_key: str,
    include_coastline: bool = True,
    progress_hook: Callable[[int, int, str], None] | None = None,
    progress_start: int = 0,
    progress_total: int = 1,
) -> dict[str, Path]:
    """Fetch map assets using collision-safe filenames and stable source URLs."""
    try:
        from sdcpy_map.datasets import (
            COASTLINE_URL,
            DRIVER_DATASETS,
            FIELD_DATASETS,
            download_if_missing,
        )
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc

    if driver_key not in DRIVER_DATASETS:
        supported = ", ".join(sorted(DRIVER_DATASETS))
        raise ValueError(f"Unknown driver dataset '{driver_key}'. Supported: {supported}.")
    if field_key not in FIELD_DATASETS:
        supported = ", ".join(sorted(FIELD_DATASETS))
        raise ValueError(f"Unknown field dataset '{field_key}'. Supported: {supported}.")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    driver_spec = DRIVER_DATASETS[driver_key]
    field_spec = FIELD_DATASETS[field_key]

    driver_url = _canonical_psl_download_url(driver_spec.url)
    field_url = _canonical_psl_download_url(field_spec.url)
    coastline_url = _canonical_psl_download_url(COASTLINE_URL)

    def _resolve_asset(url: str, destination: Path) -> Path:
        return _download_map_asset(download_if_missing, url, destination)

    out: dict[str, Path] = {}
    _emit_progress(progress_hook, progress_start, progress_total, f"Ensuring driver dataset ({driver_key})")
    out["driver"] = _resolve_asset(
        driver_url,
        data_dir / _map_asset_filename("driver", driver_key, driver_url),
    )
    _emit_progress(progress_hook, progress_start + 1, progress_total, f"Ensuring field dataset ({field_key})")
    out["field"] = _resolve_asset(
        field_url,
        data_dir / _map_asset_filename("field", field_key, field_url),
    )
    if include_coastline:
        _emit_progress(progress_hook, progress_start + 2, progress_total, "Ensuring coastline dataset")
        out["coastline"] = _resolve_asset(
            coastline_url,
            data_dir / _map_asset_filename("coastline", "ne_110m", coastline_url),
        )

    return out


def _get_driver_data_coverage(driver_key: str, data_dir: Path | str | None = None) -> dict[str, object]:
    with _DRIVER_COVERAGE_CACHE_LOCK:
        cached = _DRIVER_COVERAGE_CACHE.get(driver_key)
    if cached is not None:
        return cached

    try:
        from sdcpy_map import load_driver_series
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc

    if data_dir is None:
        data_dir = _resolve_map_cache_dir()

    paths = fetch_sdc_map_assets(
        data_dir=data_dir,
        driver_key=driver_key,
        field_key="ncep_air",
        include_coastline=False,
    )
    driver = load_driver_series(
        paths["driver"],
        config=_full_driver_config(),
        driver_key=driver_key,
    )
    defaults = _derive_driver_peak_window(driver, years=3)
    coverage = {
        "time_start": defaults["driver_min_date"],
        "time_end": defaults["driver_max_date"],
        "n_points": int(len(driver)),
        "peak_date": defaults["peak_date"],
        "peak_value": defaults["peak_value"],
    }
    with _DRIVER_COVERAGE_CACHE_LOCK:
        _DRIVER_COVERAGE_CACHE[driver_key] = coverage
    return coverage


def _get_field_data_coverage(field_key: str, data_dir: Path | str | None = None) -> dict[str, object]:
    with _FIELD_COVERAGE_CACHE_LOCK:
        cached = _FIELD_COVERAGE_CACHE.get(field_key)
    if cached is not None:
        return cached

    try:
        from sdcpy_map import load_field_anomaly_subset
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc

    if data_dir is None:
        data_dir = _resolve_map_cache_dir()

    paths = fetch_sdc_map_assets(
        data_dir=data_dir,
        driver_key="pdo",
        field_key=field_key,
        include_coastline=False,
    )
    mapped_field = load_field_anomaly_subset(
        paths["field"],
        config=_full_driver_config(),
        field_key=field_key,
    )
    time_values = np.asarray(mapped_field["time"].values)
    parsed_dates = [_as_calendar_date(value) for value in time_values]
    valid_dates = [value for value in parsed_dates if value is not None]
    bounds = _get_field_data_bounds(field_key, data_dir=data_dir)
    coverage: dict[str, object] = {
        "n_time": int(mapped_field.sizes.get("time", 0)),
        "n_lat": int(mapped_field.sizes.get("lat", 0)),
        "n_lon": int(mapped_field.sizes.get("lon", 0)),
        "lat_min": float(bounds["lat_min"]),
        "lat_max": float(bounds["lat_max"]),
        "lon_min": float(bounds["lon_min"]),
        "lon_max": float(bounds["lon_max"]),
    }
    if valid_dates:
        coverage["time_start"] = min(valid_dates).isoformat()
        coverage["time_end"] = max(valid_dates).isoformat()
    with _FIELD_COVERAGE_CACHE_LOCK:
        _FIELD_COVERAGE_CACHE[field_key] = coverage
    return coverage


def get_sdc_map_catalog() -> dict:
    """Return available map datasets and human-readable descriptions."""
    try:
        from sdcpy_map.datasets import DRIVER_DATASETS, FIELD_DATASETS
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc

    coverage_mode = os.getenv("SDCPY_STUDIO_MAP_CATALOG_COVERAGE", "cached").strip().lower()
    # Modes:
    # - none: no coverage fields in catalog payload (fastest)
    # - cached: include only already-computed in-memory coverage (default)
    # - eager: compute coverage for every driver/field (can be slow/network-heavy)
    include_cached = coverage_mode in {"cached", "eager"}
    include_eager = coverage_mode == "eager"

    def _driver_coverage(key: str) -> dict[str, object]:
        if include_eager:
            try:
                return _get_driver_data_coverage(key)
            except Exception:
                return {}
        if include_cached:
            with _DRIVER_COVERAGE_CACHE_LOCK:
                return dict(_DRIVER_COVERAGE_CACHE.get(key, {}))
        return {}

    def _field_coverage(key: str) -> dict[str, object]:
        if include_eager:
            try:
                return _get_field_data_coverage(key)
            except Exception:
                return {}
        if include_cached:
            with _FIELD_COVERAGE_CACHE_LOCK:
                return dict(_FIELD_COVERAGE_CACHE.get(key, {}))
        return {}

    drivers = [
        (
            {
                "key": key,
                "description": str(getattr(spec, "description", "")),
            }
            | _driver_coverage(key)
        )
        for key, spec in sorted(DRIVER_DATASETS.items())
    ]
    fields = [
        (
            {
                "key": key,
                "description": str(getattr(spec, "description", "")),
                "variable": str(getattr(spec, "variable", "")),
            }
            | _field_coverage(key)
        )
        for key, spec in sorted(FIELD_DATASETS.items())
    ]
    return {"drivers": drivers, "fields": fields}


def get_sdc_map_driver_defaults(driver_key: str, window_years: int = 3) -> dict:
    """Return dynamic peak-date defaults and a +/- window around it."""
    try:
        from sdcpy_map import load_driver_series
        from sdcpy_map.datasets import DRIVER_DATASETS, download_if_missing
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc

    if driver_key not in DRIVER_DATASETS:
        supported = ", ".join(sorted(DRIVER_DATASETS))
        raise ValueError(f"Unknown driver dataset '{driver_key}'. Supported: {supported}.")

    data_dir = _resolve_map_cache_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    driver_spec = DRIVER_DATASETS[driver_key]
    driver_url = _canonical_psl_download_url(driver_spec.url)
    driver_path = _download_map_asset(
        download_if_missing,
        driver_url,
        data_dir / _map_asset_filename("driver", driver_key, driver_url),
    )

    full_driver = load_driver_series(
        driver_path,
        config=_full_driver_config(),
        driver_key=driver_key,
    )
    defaults = _derive_driver_peak_window(full_driver, years=max(1, int(window_years)))
    defaults["driver_dataset"] = driver_key
    return defaults


def _get_field_data_bounds(field_key: str, data_dir: Path | str = None) -> dict[str, float]:
    """Compute the geographic bounds of a field dataset without constraints.
    
    Returns a dictionary with keys: lat_min, lat_max, lon_min, lon_max
    """
    with _FIELD_BOUNDS_CACHE_LOCK:
        if field_key in _FIELD_BOUNDS_CACHE:
            return _FIELD_BOUNDS_CACHE[field_key]
    
    try:
        from sdcpy_map import SDCMapConfig, grid_coordinates, load_field_anomaly_subset
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc
    
    if data_dir is None:
        data_dir = _resolve_map_cache_dir()
    
    # Fetch the field dataset
    paths = fetch_sdc_map_assets(
        data_dir=data_dir,
        driver_key="pdo",  # Dummy driver; only fetching field
        field_key=field_key,
        include_coastline=False,
    )
    
    # Load field with completely unconstrained bounds
    config = SDCMapConfig(
        lat_min=-90, lat_max=90,
        lon_min=-180, lon_max=180,
        lat_stride=1, lon_stride=1,
    )
    mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=field_key)
    lats, lons = grid_coordinates(mapped_field)
    
    # Compute actual data extent
    lat_array = np.asarray(lats, dtype=float)
    lon_array = np.asarray(lons, dtype=float)
    bounds = {
        "lat_min": float(np.nanmin(lat_array)),
        "lat_max": float(np.nanmax(lat_array)),
        "lon_min": float(np.nanmin(lon_array)),
        "lon_max": float(np.nanmax(lon_array)),
    }
    
    with _FIELD_BOUNDS_CACHE_LOCK:
        _FIELD_BOUNDS_CACHE[field_key] = bounds
    
    return bounds


def _resolve_map_bounds(
    request: SDCMapJobRequest,
    data_dir: Path | str | None = None,
) -> tuple[float, float, float, float, bool]:
    bounds = (request.lat_min, request.lat_max, request.lon_min, request.lon_max)
    if all(value is not None for value in bounds):
        return (
            float(request.lat_min),
            float(request.lat_max),
            float(request.lon_min),
            float(request.lon_max),
            False,
        )

    field_bounds = _get_field_data_bounds(request.field_dataset, data_dir=data_dir)
    return (
        float(field_bounds["lat_min"]),
        float(field_bounds["lat_max"]),
        float(field_bounds["lon_min"]),
        float(field_bounds["lon_max"]),
        True,
    )


def _resolve_map_temporal_window(
    request: SDCMapJobRequest,
    driver_full: pd.Series,
) -> tuple[str, str, str]:
    defaults = _derive_driver_peak_window(driver_full, years=3)
    peak_date = _as_calendar_date(request.peak_date or defaults["peak_date"])
    time_start = _as_calendar_date(request.time_start or defaults["time_start"])
    time_end = _as_calendar_date(request.time_end or defaults["time_end"])
    if peak_date is None:
        raise ValueError("`peak_date` is invalid.")
    if time_start is None or time_end is None:
        raise ValueError("`time_start` and `time_end` must be valid dates.")
    if time_start > time_end:
        raise ValueError("`time_start` must be <= `time_end`.")
    return peak_date.isoformat(), time_start.isoformat(), time_end.isoformat()


def _compute_sdcmap_layers_with_progress(
    *,
    driver: pd.Series,
    mapped_field,
    config,
    progress_hook: Callable[[int, int, str], None] | None,
    progress_base_current: int,
    progress_total: int,
) -> tuple[dict[str, np.ndarray], int]:
    try:
        from sdcpy_map.layers import _summarize_gridpoint as summarize_gridpoint
    except Exception:
        from sdcpy_map import compute_sdcmap_layers

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            layers = compute_sdcmap_layers(driver=driver, mapped_field=mapped_field, config=config)
        nlat = int(mapped_field.sizes.get("lat", 0))
        nlon = int(mapped_field.sizes.get("lon", 0))
        return layers, max(0, nlat * nlon)

    time_values = np.asarray(mapped_field["time"].values)
    peak_idx = _nearest_time_index(time_values, str(config.peak_date))
    driver_vals = driver.to_numpy(dtype=float)
    field_values = np.asarray(mapped_field.values, dtype=float)

    nlat = int(mapped_field.sizes.get("lat", 0))
    nlon = int(mapped_field.sizes.get("lon", 0))
    total_cells = max(0, nlat * nlon)

    layers = {
        "corr_mean": np.full((nlat, nlon), np.nan, dtype=float),
        "driver_rel_time_mean": np.full((nlat, nlon), np.nan, dtype=float),
        "lag_mean": np.full((nlat, nlon), np.nan, dtype=float),
        "timing_combo": np.full((nlat, nlon), np.nan, dtype=float),
        "strong_span": np.full((nlat, nlon), np.nan, dtype=float),
        "strong_start": np.full((nlat, nlon), np.nan, dtype=float),
        "dominant_sign": np.full((nlat, nlon), np.nan, dtype=float),
        "n_selected": np.full((nlat, nlon), np.nan, dtype=float),
    }

    processed = 0
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        for i in range(nlat):
            for j in range(nlon):
                local_vals = np.asarray(field_values[:, i, j], dtype=float)
                summary = summarize_gridpoint(
                    driver_vals=driver_vals,
                    local_vals=local_vals,
                    config=config,
                    peak_idx=peak_idx,
                )
                if summary is not None:
                    for key, value in summary.items():
                        layers[key][i, j] = value

                processed += 1
                _emit_progress(
                    progress_hook,
                    progress_base_current + processed,
                    progress_total,
                    f"Computing SDC map ({processed}/{total_cells} cells)",
                )

    return layers, total_cells


def build_sdc_map_exploration(payload: dict) -> dict:
    """Load map datasets and return exploration-ready arrays for interactive UI."""
    request = SDCMapJobRequest.model_validate(payload)
    try:
        from sdcpy_map import (
            SDCMapConfig,
            align_driver_to_field,
            grid_coordinates,
            load_coastline,
            load_driver_series,
            load_field_anomaly_subset,
        )
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc

    data_dir = _resolve_map_cache_dir()
    lat_min, lat_max, lon_min, lon_max, using_full_bounds = _resolve_map_bounds(
        request,
        data_dir=data_dir,
    )
    cache_key = (
        str(data_dir.resolve()),
        request.driver_dataset,
        request.field_dataset,
    )
    with _MAP_DATA_PATH_CACHE_LOCK:
        cached_paths = _MAP_DATA_PATH_CACHE.get(cache_key)
    if cached_paths and all(path.exists() and path.stat().st_size > 0 for path in cached_paths.values()):
        paths = cached_paths
    else:
        paths = fetch_sdc_map_assets(
            data_dir=data_dir,
            driver_key=request.driver_dataset,
            field_key=request.field_dataset,
        )
        with _MAP_DATA_PATH_CACHE_LOCK:
            _MAP_DATA_PATH_CACHE[cache_key] = paths

    full_driver = load_driver_series(
        paths["driver"],
        config=_full_driver_config(),
        driver_key=request.driver_dataset,
    )
    peak_date, time_start, time_end = _resolve_map_temporal_window(request, full_driver)

    config = SDCMapConfig(
        fragment_size=request.fragment_size,
        n_permutations=request.n_permutations,
        two_tailed=request.two_tailed,
        min_lag=request.min_lag,
        max_lag=request.max_lag,
        alpha=request.alpha,
        top_fraction=request.top_fraction,
        peak_date=peak_date,
        time_start=time_start,
        time_end=time_end,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_stride=request.lat_stride,
        lon_stride=request.lon_stride,
    )

    driver = load_driver_series(paths["driver"], config=config, driver_key=request.driver_dataset)
    mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=request.field_dataset)
    driver = align_driver_to_field(driver, mapped_field)
    coastline = load_coastline(paths["coastline"])
    lats, lons = grid_coordinates(mapped_field)

    time_values = np.asarray(mapped_field["time"].values)
    field_values = np.asarray(mapped_field.values, dtype=float)
    driver_values = driver.to_numpy(dtype=float)

    finite_mask = np.isfinite(field_values)
    valid_count = int(finite_mask.sum())
    total_count = int(field_values.size)
    first_valid_idx: list[int] | None = None
    field_lat_min: float | None = None
    field_lat_max: float | None = None
    field_lon_min: float | None = None
    field_lon_max: float | None = None
    field_value_min: float | None = None
    field_value_max: float | None = None
    if valid_count > 0:
        first_valid_idx = list(np.argwhere(finite_mask)[0].tolist())
        valid_cell_mask = np.any(finite_mask, axis=0)
        valid_cell_coords = np.argwhere(valid_cell_mask)
        if valid_cell_coords.size:
            lat_idx = valid_cell_coords[:, 0]
            lon_idx = valid_cell_coords[:, 1]
            lat_values = np.asarray(lats, dtype=float)[lat_idx]
            lon_values = np.asarray(lons, dtype=float)[lon_idx]
            field_lat_min = float(np.nanmin(lat_values))
            field_lat_max = float(np.nanmax(lat_values))
            field_lon_min = float(np.nanmin(lon_values))
            field_lon_max = float(np.nanmax(lon_values))

        finite_values = field_values[finite_mask]
        if finite_values.size:
            field_value_min = float(np.nanmin(finite_values))
            field_value_max = float(np.nanmax(finite_values))

    return {
        "summary": {
            "driver_dataset": request.driver_dataset,
            "field_dataset": request.field_dataset,
            "time_start": time_start,
            "time_end": time_end,
            "peak_date": peak_date,
            "n_time": int(mapped_field.sizes.get("time", 0)),
            "n_lat": int(mapped_field.sizes.get("lat", 0)),
            "n_lon": int(mapped_field.sizes.get("lon", 0)),
            "valid_values": valid_count,
            "valid_rate": float(valid_count / total_count) if total_count else 0.0,
            "first_valid_index": first_valid_idx,
            "field_lat_min": field_lat_min,
            "field_lat_max": field_lat_max,
            "field_lon_min": field_lon_min,
            "field_lon_max": field_lon_max,
            "field_value_min": field_value_min,
            "field_value_max": field_value_max,
            "used_lat_min": lat_min,
            "used_lat_max": lat_max,
            "used_lon_min": lon_min,
            "used_lon_max": lon_max,
            "full_bounds_selected": using_full_bounds,
        },
        "time_index": _serialize_time_labels(time_values),
        "driver_values": _serialize_optional_array(driver_values),
        "lat": [float(v) for v in np.asarray(lats, dtype=float).tolist()],
        "lon": [float(v) for v in np.asarray(lons, dtype=float).tolist()],
        "field_frames": _serialize_optional_cube(field_values),
        "coastline": _serialize_coastline_trace(coastline),
    }


def run_sdc_map_job(
    payload: dict,
    progress_hook: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run a beta SDC map job and return a compact JSON-friendly payload."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    request = SDCMapJobRequest.model_validate(payload)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        # Keep job execution resilient if backend selection is unavailable.
        pass

    try:
        from sdcpy_map import (
            SDCMapConfig,
            align_driver_to_field,
            grid_coordinates,
            load_coastline,
            load_driver_series,
            load_field_anomaly_subset,
            plot_layer_maps_compact,
        )
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable. Install optional dependencies with `pip install .[map]`."
        ) from exc

    initial_progress_total = 8
    _emit_progress(progress_hook, 0, initial_progress_total, "Preparing SDC map inputs")
    data_dir = _resolve_map_cache_dir()
    lat_min, lat_max, lon_min, lon_max, using_full_bounds = _resolve_map_bounds(
        request,
        data_dir=data_dir,
    )

    started = perf_counter()
    _emit_progress(
        progress_hook,
        1,
        initial_progress_total,
        f"Ensuring driver dataset ({request.driver_dataset})",
    )
    cache_key = (
        str(data_dir.resolve()),
        request.driver_dataset,
        request.field_dataset,
    )
    with _MAP_DATA_PATH_CACHE_LOCK:
        cached_paths = _MAP_DATA_PATH_CACHE.get(cache_key)

    if cached_paths and all(path.exists() and path.stat().st_size > 0 for path in cached_paths.values()):
        _emit_progress(
            progress_hook,
            1,
            initial_progress_total,
            f"Using in-memory driver cache ({request.driver_dataset})",
        )
        _emit_progress(
            progress_hook,
            2,
            initial_progress_total,
            f"Using in-memory field cache ({request.field_dataset})",
        )
        _emit_progress(progress_hook, 3, initial_progress_total, "Using in-memory coastline cache")
        paths = cached_paths
    else:
        paths = fetch_sdc_map_assets(
            data_dir=data_dir,
            driver_key=request.driver_dataset,
            field_key=request.field_dataset,
            progress_hook=progress_hook,
            progress_start=1,
            progress_total=initial_progress_total,
        )
        with _MAP_DATA_PATH_CACHE_LOCK:
            _MAP_DATA_PATH_CACHE[cache_key] = paths

    full_driver = load_driver_series(
        paths["driver"],
        config=_full_driver_config(),
        driver_key=request.driver_dataset,
    )
    peak_date, time_start, time_end = _resolve_map_temporal_window(request, full_driver)
    config = SDCMapConfig(
        fragment_size=request.fragment_size,
        n_permutations=request.n_permutations,
        two_tailed=request.two_tailed,
        min_lag=request.min_lag,
        max_lag=request.max_lag,
        alpha=request.alpha,
        top_fraction=request.top_fraction,
        peak_date=peak_date,
        time_start=time_start,
        time_end=time_end,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_stride=request.lat_stride,
        lon_stride=request.lon_stride,
    )

    _emit_progress(progress_hook, 4, initial_progress_total, "Loading and aligning series")
    driver = load_driver_series(paths["driver"], config=config, driver_key=request.driver_dataset)
    mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=request.field_dataset)
    driver = align_driver_to_field(driver, mapped_field)
    lats, lons = grid_coordinates(mapped_field)

    n_lat = int(mapped_field.sizes.get("lat", 0))
    n_lon = int(mapped_field.sizes.get("lon", 0))
    total_cells = max(1, n_lat * n_lon)
    progress_total = total_cells + 7
    _emit_progress(progress_hook, 4, progress_total, f"Grid ready ({n_lat}x{n_lon}). Starting cell loop")
    layers, _ = _compute_sdcmap_layers_with_progress(
        driver=driver,
        mapped_field=mapped_field,
        config=config,
        progress_hook=progress_hook,
        progress_base_current=4,
        progress_total=progress_total,
    )
    coastline = load_coastline(paths["coastline"])

    _emit_progress(progress_hook, progress_total - 2, progress_total, "Rendering map figure")
    fig, *_ = plot_layer_maps_compact(
        layers=layers,
        lats=lats,
        lons=lons,
        coastline=coastline,
        title="",
        return_handles=True,
    )
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    png_bytes = png_buffer.getvalue()

    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass

    _emit_progress(progress_hook, progress_total - 1, progress_total, "Packing outputs")
    nc_bytes = _build_map_netcdf_bytes(
        layers=layers,
        lats=lats,
        lons=lons,
        attrs={
            "driver_dataset": request.driver_dataset,
            "field_dataset": request.field_dataset,
            "fragment_size": int(request.fragment_size),
            "alpha": float(request.alpha),
            "top_fraction": float(request.top_fraction),
            "min_lag": int(request.min_lag),
            "max_lag": int(request.max_lag),
        },
    )

    runtime_seconds = perf_counter() - started
    corr_mean = np.asarray(layers["corr_mean"], dtype=float)
    dominant_sign = np.asarray(layers["dominant_sign"], dtype=float)
    total_cells_result = int(corr_mean.size)
    valid_cells = int(np.isfinite(corr_mean).sum())
    field_lat_min: float | None = None
    field_lat_max: float | None = None
    field_lon_min: float | None = None
    field_lon_max: float | None = None
    finite_cell_coords = np.argwhere(np.isfinite(corr_mean))
    if finite_cell_coords.size:
        lat_idx = finite_cell_coords[:, 0]
        lon_idx = finite_cell_coords[:, 1]
        lat_values = np.asarray(lats, dtype=float)[lat_idx]
        lon_values = np.asarray(lons, dtype=float)[lon_idx]
        field_lat_min = float(np.nanmin(lat_values))
        field_lat_max = float(np.nanmax(lat_values))
        field_lon_min = float(np.nanmin(lon_values))
        field_lon_max = float(np.nanmax(lon_values))
    elif len(lats) and len(lons):
        field_lat_min = float(np.nanmin(np.asarray(lats, dtype=float)))
        field_lat_max = float(np.nanmax(np.asarray(lats, dtype=float)))
        field_lon_min = float(np.nanmin(np.asarray(lons, dtype=float)))
        field_lon_max = float(np.nanmax(np.asarray(lons, dtype=float)))

    positive_cells = int(np.nansum(dominant_sign > 0))
    negative_cells = int(np.nansum(dominant_sign < 0))
    mean_abs_corr = (
        float(np.nanmean(np.abs(corr_mean))) if np.isfinite(corr_mean).any() else None
    )

    notes: list[str] = []
    if valid_cells == 0:
        notes.append("No valid grid cells passed filtering with the current parameters.")
    if request.n_permutations > 99:
        notes.append("High permutation count selected; map runs may take substantially longer.")
    if using_full_bounds:
        notes.append("Full geographic map was computed; this mode is the most computationally expensive.")

    _emit_progress(progress_hook, progress_total, progress_total, "Completed")
    layer_maps = _build_map_layer_payload(layers=layers, lats=lats, lons=lons, coastline=coastline)
    return {
        "summary": {
            "driver_dataset": request.driver_dataset,
            "field_dataset": request.field_dataset,
            "time_start": time_start,
            "time_end": time_end,
            "peak_date": peak_date,
            "fragment_size": int(request.fragment_size),
            "n_permutations": int(request.n_permutations),
            "alpha": float(request.alpha),
            "top_fraction": float(request.top_fraction),
            "min_lag": int(request.min_lag),
            "max_lag": int(request.max_lag),
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "lat_stride": int(request.lat_stride),
            "lon_stride": int(request.lon_stride),
            "n_time": int(mapped_field.sizes.get("time", 0)),
            "n_lat": int(mapped_field.sizes.get("lat", 0)),
            "n_lon": int(mapped_field.sizes.get("lon", 0)),
            "total_cells": total_cells_result,
            "valid_cells": valid_cells,
            "valid_cell_rate": float(valid_cells / total_cells_result) if total_cells_result else 0.0,
            "field_lat_min": field_lat_min,
            "field_lat_max": field_lat_max,
            "field_lon_min": field_lon_min,
            "field_lon_max": field_lon_max,
            "positive_dominant_cells": positive_cells,
            "negative_dominant_cells": negative_cells,
            "mean_abs_corr": mean_abs_corr,
            "full_bounds_selected": using_full_bounds,
        },
        "notes": notes,
        "runtime_seconds": float(runtime_seconds),
        "figure_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "layer_maps": layer_maps,
        "download_formats": ["png", "nc"],
        "_artifacts_map": {
            "png": png_bytes,
            "nc": nc_bytes,
            "driver_dataset": request.driver_dataset,
            "field_dataset": request.field_dataset,
            "fragment_size": int(request.fragment_size),
        },
    }


def run_sdc_job(
    payload: dict,
    progress_hook: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run a full SDC job and return a compact JSON-friendly payload."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    request = SDCJobRequest.model_validate(payload)

    ts1 = np.asarray(request.ts1, dtype=float)
    ts2 = np.asarray(request.ts2, dtype=float)
    if request.index_values is not None and len(request.index_values) == len(ts1):
        series_index: list[int | str] = [str(v) for v in request.index_values]
    else:
        series_index = [int(i) for i in range(len(ts1))]

    _emit_progress(progress_hook, 0, 1, "Preparing inputs")

    original_tqdm = sdc_core.tqdm

    def reporting_tqdm(*args, **kwargs):
        bar = original_tqdm(*args, **kwargs)
        total = int(getattr(bar, "total", 0) or 1)
        description = kwargs.get("desc") or getattr(bar, "desc", "Computing")
        _emit_progress(progress_hook, 0, total, description)

        original_update = bar.update

        def _update(n: int = 1):
            original_update(n)
            _emit_progress(progress_hook, int(getattr(bar, "n", 0)), total, description)

        bar.update = _update  # type: ignore[assignment]
        return bar

    sdc_core.tqdm = reporting_tqdm

    started = perf_counter()
    try:
        sdc_df = compute_sdc(
            ts1=ts1,
            ts2=ts2,
            fragment_size=request.fragment_size,
            n_permutations=request.n_permutations,
            method=request.method,
            two_tailed=request.two_tailed,
            permutations=request.permutations,
            min_lag=request.min_lag,
            max_lag=request.max_lag,
            max_memory_gb=request.max_memory_gb,
        )
    finally:
        sdc_core.tqdm = original_tqdm

    runtime_seconds = perf_counter() - started

    _emit_progress(progress_hook, 0, 1, "Summarizing outputs")

    valid = sdc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["r", "p_value"]).copy()
    significant = valid.loc[valid["p_value"] <= request.alpha].copy()

    if valid.empty:
        raise ValueError("No valid SDC rows were generated.")

    strongest = significant.copy()
    if not strongest.empty:
        strongest["abs_r"] = strongest["r"].abs()
        strongest = (
            strongest.nlargest(MAX_STRONGEST_LINKS, "abs_r")
            .drop(columns=["abs_r"])
            .loc[:, ["start_1", "stop_1", "start_2", "stop_2", "lag", "r", "p_value"]]
        )

    r_matrix_full = valid.pivot(index="start_2", columns="start_1", values="r")
    p_matrix_full = valid.pivot(index="start_2", columns="start_1", values="p_value")
    p_matrix_full = p_matrix_full.reindex_like(r_matrix_full)
    r_matrix_full, p_matrix_full = _mask_matrix_outside_lag_window(
        r_matrix_full,
        p_matrix_full,
        request.min_lag,
        request.max_lag,
    )

    r_matrix = _downsample_matrix(r_matrix_full, request.heatmap_step)
    p_matrix = _downsample_matrix(p_matrix_full, request.heatmap_step)
    p_matrix = p_matrix.reindex(index=r_matrix.index, columns=r_matrix.columns)

    lag_r_full = valid.pivot(index="lag", columns="start_1", values="r").sort_index(axis=0)
    lag_r_full = lag_r_full.sort_index(axis=1)
    lag_p_full = valid.pivot(index="lag", columns="start_1", values="p_value").sort_index(axis=0)
    lag_p_full = lag_p_full.sort_index(axis=1).reindex_like(lag_r_full)
    if not r_matrix_full.empty:
        min_start2 = int(r_matrix_full.index.min())
        max_start2 = int(r_matrix_full.index.max())
        lag_r_full, lag_p_full = _mask_lag_matrix_outside_start2_range(
            lag_r_full,
            lag_p_full,
            min_start2=min_start2,
            max_start2=max_start2,
        )

    lag_r_matrix = _downsample_matrix_columns(lag_r_full, request.heatmap_step)
    lag_p_matrix = _downsample_matrix_columns(lag_p_full, request.heatmap_step)
    lag_p_matrix = lag_p_matrix.reindex(index=lag_r_matrix.index, columns=lag_r_matrix.columns)

    lag_default: int | None
    if lag_r_matrix.empty:
        lag_default = None
    else:
        lag_values = [int(v) for v in lag_r_matrix.index.to_numpy()]
        if 0 in lag_values:
            lag_default = 0
        else:
            lag_default = min(lag_values, key=lambda value: abs(value))

    lag0_corr = pd.Series(ts1, dtype=float).corr(pd.Series(ts2, dtype=float), method=request.method)

    summary = {
        "series_length": int(len(ts1)),
        "fragment_size": int(request.fragment_size),
        "n_pairs": int(len(valid)),
        "n_significant": int(len(significant)),
        "significant_rate": float(len(significant) / len(valid)) if len(valid) else 0.0,
        "r_min": float(valid["r"].min()),
        "r_max": float(valid["r"].max()),
        "full_series_corr_lag0": float(lag0_corr) if pd.notna(lag0_corr) else None,
        "method": request.method,
        "ts1_label": request.ts1_label,
        "ts2_label": request.ts2_label,
        "alpha": float(request.alpha),
        "permutations": bool(request.permutations),
        "n_permutations": int(request.n_permutations),
        "two_tailed": bool(request.two_tailed),
    }

    notes: list[str] = []
    if len(ts1) > 1200:
        notes.append("Large input length detected; job runtime may increase substantially.")
    if request.n_permutations > 199:
        notes.append("High permutation count selected; consider reducing for exploratory work.")
    if significant.empty:
        notes.append("No significant links found at the selected alpha threshold.")

    _emit_progress(progress_hook, 1, 1, "Completed")

    return {
        "summary": summary,
        "series": {
            "index": series_index,
            "ts1": [float(v) for v in ts1],
            "ts2": [float(v) for v in ts2],
        },
        "matrix_r": _matrix_payload(r_matrix),
        "matrix_p": _matrix_payload(p_matrix),
        "lag_matrix_r": _matrix_payload(lag_r_matrix),
        "lag_matrix_p": _matrix_payload(lag_p_matrix),
        "lag_default": lag_default,
        "strongest_links": strongest.to_dict(orient="records") if not strongest.empty else [],
        "notes": notes,
        "runtime_seconds": float(runtime_seconds),
        "_artifacts": _serialize_artifacts(request, valid),
    }
