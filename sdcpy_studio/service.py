"""Service-layer helpers for sdcpy-studio computations and exports."""

from __future__ import annotations

import base64
import csv
import io
import os
import re
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import sdcpy.core as sdc_core
from sdcpy import compute_sdc
from sdcpy.scale_dependent_correlation import SDCAnalysis

from sdcpy_studio.schemas import (
    DriverDefaults,
    FieldDims,
    FieldNormalizationInfo,
    FieldSelectorDefinition,
    FieldSelectorOption,
    FieldVariableOption,
    IncompatibleVariable,
    InspectWarning,
    SDCJobFromDatasetRequest,
    SDCJobRequest,
    SDCMapJobRequest,
)

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
_PU_OR: tuple[tuple[float, str], ...] = (
    (0.0, "#7f3b08"),
    (0.1, "#b35806"),
    (0.2, "#e08214"),
    (0.3, "#fdb863"),
    (0.4, "#fee0b6"),
    (0.5, "#f7f7f7"),
    (0.6, "#d8daeb"),
    (0.7, "#b2abd2"),
    (0.8, "#8073ac"),
    (0.9, "#542788"),
    (1.0, "#2d004b"),
)
_COOLWARM: tuple[tuple[float, str], ...] = (
    (0.0, "#3b4cc0"),
    (0.1, "#5d7ce6"),
    (0.2, "#7b9ff9"),
    (0.3, "#9ebeff"),
    (0.4, "#c0d4f5"),
    (0.5, "#dddcdc"),
    (0.6, "#f2cbb7"),
    (0.7, "#f7a889"),
    (0.8, "#ee8468"),
    (0.9, "#d65244"),
    (1.0, "#b40426"),
)
_BR_BG: tuple[tuple[float, str], ...] = (
    (0.0, "#543005"),
    (0.1, "#8c510a"),
    (0.2, "#bf812d"),
    (0.3, "#dfc27d"),
    (0.4, "#f6e8c3"),
    (0.5, "#f5f5f5"),
    (0.6, "#c7eae5"),
    (0.7, "#80cdc1"),
    (0.8, "#35978f"),
    (0.9, "#01665e"),
    (1.0, "#003c30"),
)
_CSV_DELIMITERS: tuple[tuple[str, str], ...] = (
    (",", "comma"),
    (";", "semicolon"),
    ("\t", "tab"),
    ("|", "pipe"),
)
_MAP_LAYER_DEFS: tuple[dict[str, object], ...] = (
    {
        "key": "corr_mean",
        "label": "A. Correlation",
        "description": "Strongest significant event-conditioned correlation kept at each grid cell.",
        "colorscale": [[stop, color] for stop, color in _RD_BU_WHITE_CENTER],
        "zmin": -1.0,
        "zmax": 1.0,
    },
    {
        "key": "driver_rel_time_mean",
        "label": "B. Position",
        "description": "Center of the retained driver fragment relative to the selected event peak.",
        "colorscale": [[stop, color] for stop, color in _PU_OR],
        "zmin": None,
        "zmax": None,
    },
    {
        "key": "lag_mean",
        "label": "C. Lag",
        "description": "Lag of the retained field fragment relative to the driver fragment.",
        "colorscale": [[stop, color] for stop, color in _COOLWARM],
        "zmin": None,
        "zmax": None,
    },
    {
        "key": "timing_combo",
        "label": "D. Timing",
        "description": "Retained field-fragment timing with respect to the selected event peak.",
        "colorscale": [[stop, color] for stop, color in _BR_BG],
        "zmin": None,
        "zmax": None,
    },
)


class SDCMapJobCancelledError(RuntimeError):
    """Raised when a running SDC map job is cancelled cooperatively."""


def _raise_if_cancelled(cancel_event: Event | None, description: str = "SDC map job cancelled.") -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise SDCMapJobCancelledError(description)


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


def _load_map_dataset_registry():
    try:
        from sdcpy_map.datasets import (
            COASTLINE_URL,
            DRIVER_DATASETS,
            FIELD_DATASETS,
            download_if_missing,
        )
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc
    return COASTLINE_URL, DRIVER_DATASETS, FIELD_DATASETS, download_if_missing


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


def _infer_series_cadence(index: pd.DatetimeIndex) -> str | None:
    clean = pd.DatetimeIndex(index).sort_values()
    if len(clean) < 3:
        return None
    inferred = pd.infer_freq(clean)
    if inferred is None:
        return None
    freq = str(inferred).upper()
    if freq in {"M", "ME", "MS"} or freq.startswith(("M-", "ME-", "MS-")):
        return "monthly"
    if freq in {"A", "AS", "Y", "YS", "YE"} or freq.startswith(("A-", "AS-", "Y-", "YS-", "YE-")):
        return "yearly"
    if freq.startswith("W-"):
        return "weekly"
    return None


def _canonicalize_window_bounds(
    start_value: date,
    end_value: date,
    *,
    cadence: str | None,
) -> tuple[date, date]:
    if cadence == "monthly":
        start_value = pd.Period(start_value.isoformat(), freq="M").start_time.date()
        end_value = pd.Period(end_value.isoformat(), freq="M").end_time.date()
    elif cadence == "yearly":
        start_value = pd.Period(start_value.isoformat(), freq="Y").start_time.date()
        end_value = pd.Period(end_value.isoformat(), freq="Y").end_time.date()
    return start_value, end_value


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


def _infer_time_step_info(time_values: np.ndarray) -> dict[str, str]:
    parsed_dates = [_as_calendar_date(value) for value in np.asarray(time_values)]
    valid_dates = [value for value in parsed_dates if value is not None]
    if len(valid_dates) < 2:
        return {"singular": "time step", "plural": "time steps"}

    diffs = np.diff(np.asarray(valid_dates, dtype="datetime64[D]")).astype("timedelta64[D]").astype(int)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return {"singular": "time step", "plural": "time steps"}

    median_days = float(np.median(diffs))
    if 27.0 <= median_days <= 32.0:
        return {"singular": "month", "plural": "months"}
    if 6.0 <= median_days <= 8.0:
        return {"singular": "week", "plural": "weeks"}
    if 0.5 <= median_days <= 1.5:
        return {"singular": "day", "plural": "days"}
    if 300.0 <= median_days <= 400.0:
        return {"singular": "year", "plural": "years"}
    return {"singular": "time step", "plural": "time steps"}


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


def _derive_driver_defaults(driver: pd.Series) -> dict[str, object]:
    clean = driver.dropna().sort_index()
    if clean.empty:
        raise ValueError("Driver dataset does not contain valid values.")

    parsed_index_dates = [_as_calendar_date(value) for value in clean.index]
    valid_dates = [value for value in parsed_index_dates if value is not None]
    if not valid_dates:
        raise ValueError("Driver dataset does not contain parseable datetime values.")

    min_date = min(valid_dates)
    max_date = max(valid_dates)
    cadence = _infer_series_cadence(pd.DatetimeIndex(clean.index))
    time_start, time_end = _canonicalize_window_bounds(min_date, max_date, cadence=cadence)
    return {
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "driver_min_date": min_date.isoformat(),
        "driver_max_date": max_date.isoformat(),
        "n_points": int(len(clean)),
    }


def _public_event_catalog(catalog: dict[str, object] | None) -> dict[str, object]:
    raw = catalog or {}
    def _normalize_event_item(item: object) -> dict[str, object]:
        payload = dict(item or {})
        return {
            "index": int(payload["index"]) if payload.get("index") is not None else None,
            "date": str(payload.get("date") or ""),
            "value": float(payload.get("value") or 0.0),
            "sign": str(payload.get("sign") or ""),
            "source": str(payload.get("source") or "auto"),
        }

    return {
        "selected_positive": [_normalize_event_item(item) for item in (raw.get("selected_positive") or [])],
        "selected_negative": [_normalize_event_item(item) for item in (raw.get("selected_negative") or [])],
        "ignored_positive": [_normalize_event_item(item) for item in (raw.get("ignored_positive") or [])],
        "ignored_negative": [_normalize_event_item(item) for item in (raw.get("ignored_negative") or [])],
        "base_state_threshold": raw.get("base_state_threshold"),
        "base_state_count": int(raw.get("base_state_count") or 0),
        "warnings": [str(item) for item in (raw.get("warnings") or [])],
        "selection_mode": str(raw.get("selection_mode") or "auto"),
    }


def _detect_driver_event_catalog(
    driver: pd.Series,
    *,
    correlation_width: int = 12,
    n_positive_peaks: int = 3,
    n_negative_peaks: int = 3,
    base_state_beta: float = 0.5,
    manual_event_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    try:
        from sdcpy_map import SDCMapConfig, resolve_driver_event_catalog
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    config = SDCMapConfig(
        correlation_width=int(correlation_width),
        n_positive_peaks=int(n_positive_peaks),
        n_negative_peaks=int(n_negative_peaks),
        base_state_beta=float(base_state_beta),
    )
    return _public_event_catalog(
        resolve_driver_event_catalog(
            driver.sort_index(),
            config,
            manual_event_selection=manual_event_selection,
        )
    )


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


def _rank_csv_delimiter_candidates(content: bytes) -> list[tuple[str, str]]:
    sample = content.decode("utf-8-sig", errors="replace")[:4096]
    ranked: list[tuple[str, str]] = []
    seen: set[str] = set()
    if sample.strip():
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters="".join(sep for sep, _ in _CSV_DELIMITERS))
            for sep, label in _CSV_DELIMITERS:
                if sniffed.delimiter == sep:
                    ranked.append((sep, label))
                    seen.add(sep)
                    break
        except csv.Error:
            pass

    ranked.extend(
        (sep, label)
        for sep, label in sorted(
            _CSV_DELIMITERS,
            key=lambda item: sample.count(item[0]),
            reverse=True,
        )
        if sep not in seen
    )
    return ranked or list(_CSV_DELIMITERS)


def _read_csv_with_fallbacks(content: bytes) -> tuple[pd.DataFrame, str | None, str | None]:
    candidates = _rank_csv_delimiter_candidates(content)
    attempts: list[tuple[pd.DataFrame, str, str]] = []
    last_error: Exception | None = None

    for sep, label in candidates:
        try:
            frame = pd.read_csv(io.BytesIO(content), sep=sep)
        except Exception as exc:
            last_error = exc
            continue
        attempts.append((frame, sep, label))

    if not attempts:
        raise ValueError(
            "Could not parse CSV. Supported delimiters: comma, semicolon, tab, pipe."
        ) from last_error

    frame, sep, label = max(
        attempts,
        key=lambda item: (int(item[0].shape[1]), -candidates.index((item[1], item[2]))),
    )
    return frame, sep, label


def _coerce_datetime_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def inspect_dataset_csv(content: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """Inspect uploaded dataset and infer useful column types for the UI workflow."""
    frame, detected_delimiter, delimiter_name = _read_csv_with_fallbacks(content)
    if frame.empty:
        raise ValueError("Uploaded dataset is empty.")

    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]

    datetime_columns: list[str] = []
    numeric_columns: list[str] = []
    warnings: list[InspectWarning] = []
    rejected_numeric_columns: list[str] = []

    for col in frame.columns:
        series = frame[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(col)
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            parsed = _coerce_datetime_series(series)
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
            continue
        if non_null and parsed_count > 0:
            rejected_numeric_columns.append(str(col))

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
        "detected_delimiter": detected_delimiter,
        "delimiter_name": delimiter_name,
        "warnings": warnings,
        "rejected_numeric_columns": rejected_numeric_columns,
        "preview_rows": preview,
    }
    return frame, metadata


def _load_custom_driver_series_from_frame(
    frame: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    time_start: str | None = None,
    time_end: str | None = None,
) -> pd.Series:
    if date_column not in frame.columns:
        raise ValueError(f"Driver date column '{date_column}' not found in uploaded CSV.")
    if value_column not in frame.columns:
        raise ValueError(f"Driver value column '{value_column}' not found in uploaded CSV.")

    parsed_dates = _coerce_datetime_series(frame[date_column])
    values = pd.to_numeric(frame[value_column], errors="coerce")
    valid = parsed_dates.notna() & values.notna()
    if not bool(valid.any()):
        raise ValueError("Uploaded driver CSV does not contain valid datetime/value pairs.")

    series = pd.Series(values.loc[valid].to_numpy(dtype=float), index=parsed_dates.loc[valid])
    series = series.groupby(level=0).mean().sort_index()

    if time_start:
        series = series.loc[series.index >= pd.Timestamp(time_start)]
    if time_end:
        series = series.loc[series.index <= pd.Timestamp(time_end)]
    if series.empty:
        raise ValueError("Uploaded driver CSV has no values inside the selected time window.")
    return series


def load_custom_map_driver_series(
    csv_path: Path | str,
    *,
    date_column: str,
    value_column: str,
    time_start: str | None = None,
    time_end: str | None = None,
) -> pd.Series:
    frame, *_ = _read_csv_with_fallbacks(Path(csv_path).read_bytes())
    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    return _load_custom_driver_series_from_frame(
        frame,
        date_column=date_column,
        value_column=value_column,
        time_start=time_start,
        time_end=time_end,
    )


def inspect_sdc_map_driver_csv(content: bytes, filename: str) -> dict:
    """Inspect uploaded map-driver CSV and infer defaults for the map workflow."""
    frame, metadata = inspect_dataset_csv(content, filename=filename)
    if not metadata["datetime_columns"]:
        raise ValueError("Driver CSV must contain a parseable date column.")
    if not metadata["numeric_columns"]:
        raise ValueError("Driver CSV must contain at least one numeric time-series column.")

    date_column = str(metadata["suggested_date_column"] or metadata["datetime_columns"][0])
    numeric_candidates = [str(col) for col in metadata["numeric_columns"] if col != date_column]
    if not numeric_candidates:
        numeric_candidates = [str(metadata["numeric_columns"][0])]
    value_column = numeric_candidates[0]
    first_non_date_column = next(
        (str(col) for col in metadata["columns"] if str(col) != date_column),
        None,
    )
    warnings = list(metadata.get("warnings") or [])
    if first_non_date_column and value_column != first_non_date_column:
        warnings.append(
            InspectWarning(
                code="series_auto_selected",
                message=(
                    f"Selected '{value_column}' because '{first_non_date_column}' "
                    "did not contain enough numeric values."
                ),
                columns=[first_non_date_column, value_column],
            )
        )
    driver_series = _load_custom_driver_series_from_frame(
        frame,
        date_column=date_column,
        value_column=value_column,
    )
    defaults = DriverDefaults.model_validate(_derive_driver_defaults(driver_series))
    event_catalog = _detect_driver_event_catalog(driver_series)
    return {
        **metadata,
        "suggested_date_column": date_column,
        "suggested_value_column": value_column,
        "warnings": warnings,
        "defaults": defaults,
        "event_catalog": event_catalog,
    }


def _normalize_netcdf_open_error(filename: str) -> str:
    return (
        f"Could not open NetCDF file '{filename}'. "
        "The file is unreadable with the bundled NetCDF backends in this environment."
    )


def _manually_decode_time_coordinate(dataset):
    time_name = _pick_coord_name(["time", "date", "datetime", "t"], [str(name) for name in dataset.coords])
    if time_name is None:
        return dataset, False

    time_coord = dataset.coords[time_name]
    if np.issubdtype(time_coord.dtype, np.datetime64):
        return dataset, False
    if not np.issubdtype(time_coord.dtype, np.number):
        return dataset, False

    units = str(time_coord.attrs.get("units") or "").strip()
    if not units:
        raise ValueError(
            "This NetCDF file could be opened, but its time axis could not be decoded automatically."
        )

    try:
        import cftime
    except ImportError as exc:
        raise ValueError(
            "This NetCDF file needs the standard project dependencies to decode its time axis. "
            "Install them with `uv sync`."
        ) from exc

    calendar = str(time_coord.attrs.get("calendar") or "standard")
    try:
        decoded = cftime.num2date(
            np.asarray(time_coord.values),
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=True,
        )
    except Exception as exc:
        raise ValueError(
            "This NetCDF file could be opened, but its time axis could not be decoded automatically."
        ) from exc
    dataset = dataset.assign_coords({time_name: (time_coord.dims, np.asarray(decoded, dtype=object))})
    dataset.attrs["_sdcpy_studio_time_decode_mode"] = "manual"
    return dataset, True


def _open_netcdf_dataset(dataset_path: Path | str):
    try:
        import xarray as xr
    except ImportError as exc:
        raise ValueError(
            "NetCDF support requires the standard project dependencies. Install them with `uv sync`."
        ) from exc

    filename = Path(dataset_path).name
    last_error: Exception | None = None
    for engine in (None, "h5netcdf", "netcdf4", "scipy"):
        try:
            if engine is None:
                return xr.open_dataset(dataset_path)
            return xr.open_dataset(dataset_path, engine=engine)
        except Exception as exc:  # pragma: no cover - depends on installed backends
            last_error = exc
            continue

    decode_error: Exception | None = None
    for engine in (None, "h5netcdf", "netcdf4", "scipy"):
        dataset = None
        try:
            if engine is None:
                dataset = xr.open_dataset(dataset_path, decode_times=False)
            else:
                dataset = xr.open_dataset(dataset_path, engine=engine, decode_times=False)
        except Exception as exc:  # pragma: no cover - depends on installed backends
            last_error = exc
            if dataset is not None:
                try:
                    dataset.close()
                except Exception:
                    pass
            continue
        try:
            dataset, _ = _manually_decode_time_coordinate(dataset)
            return dataset
        except ValueError as exc:
            decode_error = exc
            if dataset is not None:
                try:
                    dataset.close()
                except Exception:
                    pass

    if decode_error is not None:
        raise decode_error from last_error
    raise ValueError(_normalize_netcdf_open_error(filename)) from last_error


def _pick_coord_name(candidates: list[str], names: list[str]) -> str | None:
    name_map = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        found = name_map.get(candidate.lower())
        if found:
            return found
    return None


def _find_dimension_coordinate(data_array, dim_name: str):
    if dim_name in data_array.coords and tuple(str(item) for item in data_array.coords[dim_name].dims) == (dim_name,):
        return data_array.coords[dim_name]
    for coord_name in data_array.coords:
        coord = data_array.coords[coord_name]
        if tuple(str(item) for item in coord.dims) == (dim_name,):
            return coord
    return None


def _climatology_time_axis_error(data_array) -> str | None:
    time_dim = _pick_coord_name(["time", "date", "datetime", "t"], [str(dim) for dim in data_array.dims])
    if time_dim is None:
        return None

    time_coord = _find_dimension_coordinate(data_array, time_dim)
    if time_coord is None:
        return None

    climatology_name = str(time_coord.attrs.get("climatology") or "").strip()
    climo_period = str(time_coord.attrs.get("climo_period") or "").strip()
    time_values = np.asarray(time_coord.values)
    parsed_dates = [_as_calendar_date(value) for value in time_values]
    valid_dates = [value for value in parsed_dates if value is not None]

    looks_like_month_bins = False
    if len(valid_dates) == int(time_coord.size) == 12:
        months = sorted({value.month for value in valid_dates})
        years = {value.year for value in valid_dates}
        looks_like_month_bins = months == list(range(1, 13)) and len(years) == 1 and min(years) <= 1

    if not climatology_name and not climo_period and not looks_like_month_bins:
        return None

    variable_label = f"'{data_array.name}'" if getattr(data_array, "name", None) else "This field variable"
    if int(time_coord.size) == 12:
        bin_text = "12 month-of-year climatology bins"
    else:
        bin_text = f"{int(time_coord.size)} climatology time bins"
    period_text = f" for {climo_period}" if climo_period else ""
    return (
        f"{variable_label} uses a climatology-style time axis{period_text}. "
        f"It provides {bin_text}, not a dated time series, so it cannot be aligned with the driver for SDC Map. "
        "Use a time-resolved NetCDF such as '*.mon.mean.nc' instead."
    )


def _serialize_selector_value(value: object) -> str:
    parsed = _as_calendar_date(value)
    if parsed is not None:
        return parsed.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, int):
        return str(value)
    return str(value)


def _suggest_selector_value(dim_name: str, coord_values: list[object]) -> str | None:
    if not coord_values:
        return None

    lowered = str(dim_name).lower()
    numeric_pairs: list[tuple[float, object]] = []
    for value in coord_values:
        try:
            numeric_pairs.append((float(value), value))
        except Exception:
            continue

    if numeric_pairs:
        if any(token in lowered for token in ("level", "lev", "plev", "pressure", "isobaric")):
            return _serialize_selector_value(max(numeric_pairs, key=lambda item: item[0])[1])
        if any(token in lowered for token in ("depth", "dep", "z")):
            return _serialize_selector_value(min(numeric_pairs, key=lambda item: abs(item[0]))[1])
    return _serialize_selector_value(coord_values[0])


def _build_field_selector_definition(data_array, dim_name: str) -> FieldSelectorDefinition:
    coord = _find_dimension_coordinate(data_array, dim_name)
    if coord is None:
        raise ValueError(
            f"Field variable dimension '{dim_name}' requires a selectable coordinate axis, but none was found."
        )
    if int(coord.size) > 100:
        raise ValueError(
            f"Field variable dimension '{dim_name}' has {int(coord.size)} values, which is too many to expose safely in the upload UI."
        )

    raw_values = list(np.asarray(coord.values).tolist())
    if not raw_values:
        raise ValueError(
            f"Field variable dimension '{dim_name}' requires a selectable coordinate axis, but it is empty."
        )

    units = str(coord.attrs.get("units") or "").strip()
    options = [
        FieldSelectorOption(
            value=_serialize_selector_value(value),
            label=f"{_serialize_selector_value(value)} {units}".strip(),
        )
        for value in raw_values
    ]
    return FieldSelectorDefinition(
        dimension=dim_name,
        label=f"{dim_name}{f' ({units})' if units else ''}",
        options=options,
        suggested_value=_suggest_selector_value(dim_name, raw_values),
    )


def _describe_custom_field_variable(data_array) -> tuple[FieldNormalizationInfo, list[FieldSelectorDefinition]]:
    original_dims = [str(dim) for dim in data_array.dims]
    time_dim = _pick_coord_name(["time", "date", "datetime", "t"], original_dims)
    lat_dim = _pick_coord_name(["lat", "latitude", "y"], original_dims)
    lon_dim = _pick_coord_name(["lon", "longitude", "x"], original_dims)
    if time_dim is None or lat_dim is None or lon_dim is None:
        raise ValueError(
            "Field variable must have time/lat/lon dimensions (accepted aliases: time/date, lat/latitude, lon/longitude)."
        )

    climatology_error = _climatology_time_axis_error(data_array)
    if climatology_error:
        raise ValueError(climatology_error)

    selectors: list[FieldSelectorDefinition] = []
    squeezed_dims: list[str] = []
    for dim_name in original_dims:
        if dim_name in {time_dim, lat_dim, lon_dim}:
            continue
        size = int(data_array.sizes.get(dim_name, 0))
        if size == 1:
            squeezed_dims.append(dim_name)
            continue
        selectors.append(_build_field_selector_definition(data_array, dim_name))

    return (
        FieldNormalizationInfo(
            original_dims=original_dims,
            squeezed_dims=squeezed_dims,
            selected_dimensions={},
        ),
        selectors,
    )


def _apply_field_dimension_selection(data_array, dim_name: str, raw_value: str):
    coord = _find_dimension_coordinate(data_array, dim_name)
    if coord is None:
        raise ValueError(f"Field dimension '{dim_name}' does not expose coordinate values for selection.")
    coord_values = list(np.asarray(coord.values).tolist())
    serialized = [_serialize_selector_value(value) for value in coord_values]
    try:
        selected_index = serialized.index(str(raw_value))
    except ValueError as exc:
        raise ValueError(
            f"Selected value '{raw_value}' is not available for field dimension '{dim_name}'."
        ) from exc
    return data_array.sel({dim_name: coord_values[selected_index]})


def _normalize_custom_field_dataarray(
    data_array,
    *,
    apply_config=None,
    return_metadata: bool = False,
    dimension_selections: dict[str, str] | None = None,
):
    original_dims = [str(dim) for dim in data_array.dims]
    dims = list(original_dims)
    rename_map: dict[str, str] = {}
    time_dim = _pick_coord_name(["time", "date", "datetime", "t"], dims)
    lat_dim = _pick_coord_name(["lat", "latitude", "y"], dims)
    lon_dim = _pick_coord_name(["lon", "longitude", "x"], dims)
    if time_dim is None or lat_dim is None or lon_dim is None:
        raise ValueError(
            "Field variable must have time/lat/lon dimensions (accepted aliases: time/date, lat/latitude, lon/longitude)."
        )
    rename_map[time_dim] = "time"
    rename_map[lat_dim] = "lat"
    rename_map[lon_dim] = "lon"

    normalization, selectors = _describe_custom_field_variable(data_array)
    da = data_array
    selected_values: dict[str, str] = {}
    selection_map = {str(key): str(value) for key, value in (dimension_selections or {}).items() if value is not None}
    for selector in selectors:
        selected_value = selection_map.get(selector.dimension) or selector.suggested_value
        if not selected_value:
            raise ValueError(
                f"Field variable requires a selection for dimension '{selector.dimension}' before it can be used."
            )
        da = _apply_field_dimension_selection(da, selector.dimension, selected_value)
        selected_values[selector.dimension] = selected_value

    da = da.rename(rename_map)
    extra_dims = [str(dim) for dim in da.dims if str(dim) not in {"time", "lat", "lon"}]
    squeezed_dims = [dim for dim in extra_dims if int(da.sizes.get(dim, 0)) == 1]
    non_singleton_dims = [dim for dim in extra_dims if int(da.sizes.get(dim, 0)) != 1]
    if non_singleton_dims:
        formatted = ", ".join(f"{dim} ({int(da.sizes.get(dim, 0))})" for dim in non_singleton_dims)
        raise ValueError(
            "Field variable has unsupported non-singleton extra dimensions: "
            f"{formatted}. Only singleton extra dimensions can be normalized automatically."
        )
    if squeezed_dims:
        da = da.squeeze(dim=squeezed_dims, drop=True)

    da = da.transpose("time", "lat", "lon")

    for coord_name in ("time", "lat", "lon"):
        if coord_name not in da.coords:
            raise ValueError(f"Field variable is missing required '{coord_name}' coordinate values.")

    # Normalize and sort longitude into [-180, 180) when data appears in [0, 360].
    lon_values = np.asarray(da["lon"].values, dtype=float)
    if lon_values.size and np.nanmax(lon_values) > 180.0:
        normalized = ((lon_values + 180.0) % 360.0) - 180.0
        da = da.assign_coords(lon=normalized)

    da = da.sortby("time").sortby("lat").sortby("lon")

    if apply_config is not None:
        if getattr(apply_config, "time_start", None) is not None or getattr(apply_config, "time_end", None) is not None:
            try:
                start = getattr(apply_config, "time_start", None) or None
                end = getattr(apply_config, "time_end", None) or None
                da = da.sel(time=slice(start, end))
            except Exception:
                pass
        da = da.sel(
            lat=slice(float(apply_config.lat_min), float(apply_config.lat_max)),
            lon=slice(float(apply_config.lon_min), float(apply_config.lon_max)),
        )
        da = da.isel(
            lat=slice(None, None, max(1, int(getattr(apply_config, "lat_stride", 1)))),
            lon=slice(None, None, max(1, int(getattr(apply_config, "lon_stride", 1)))),
        )

    if int(da.sizes.get("time", 0)) <= 0 or int(da.sizes.get("lat", 0)) <= 0 or int(da.sizes.get("lon", 0)) <= 0:
        raise ValueError("Selected field variable has no data after applying bounds/time filters.")

    normalized = da.astype(float)
    if not return_metadata:
        return normalized
    return normalized, FieldNormalizationInfo(
        original_dims=normalization.original_dims,
        squeezed_dims=sorted({*normalization.squeezed_dims, *squeezed_dims}),
        selected_dimensions=selected_values,
    )


def inspect_sdc_map_field_netcdf(dataset_path: Path | str, filename: str) -> dict:
    """Inspect an uploaded NetCDF and list compatible field variables."""
    dataset = _open_netcdf_dataset(dataset_path)
    try:
        variables = [str(name) for name in dataset.data_vars]
        compatible: list[str] = []
        variable_options: list[FieldVariableOption] = []
        incompatible: list[IncompatibleVariable] = []
        coverage_by_var: dict[str, dict[str, object]] = {}
        warnings: list[InspectWarning] = []
        for name in variables:
            try:
                normalization_preview, selectors = _describe_custom_field_variable(dataset[name])
                selected_dimensions = {
                    selector.dimension: selector.suggested_value
                    for selector in selectors
                    if selector.suggested_value is not None
                }
                da, normalization = _normalize_custom_field_dataarray(
                    dataset[name],
                    return_metadata=True,
                    dimension_selections=selected_dimensions,
                )
            except Exception as exc:
                incompatible.append(IncompatibleVariable(name=name, reason=str(exc)))
                continue
            compatible.append(name)
            variable_warnings: list[InspectWarning] = []
            if normalization_preview.squeezed_dims:
                variable_warnings.append(
                    InspectWarning(
                        code="singleton_dimensions_squeezed",
                        message=(
                            f"Normalized singleton dimensions for '{name}': "
                            + ", ".join(str(dim) for dim in normalization_preview.squeezed_dims)
                        ),
                        columns=[name],
                    )
                )
            if selectors:
                selector_summary = ", ".join(
                    f"{selector.dimension}={selector.suggested_value}" for selector in selectors if selector.suggested_value
                )
                variable_warnings.append(
                    InspectWarning(
                        code="dimension_selection_available",
                        message=(
                            f"'{name}' requires selecting extra dimensions before analysis."
                            + (f" Suggested: {selector_summary}." if selector_summary else "")
                        ),
                        columns=[name],
                    )
                )
            variable_options.append(
                FieldVariableOption(
                    name=name,
                    selectors=selectors,
                    normalization=normalization,
                    warnings=variable_warnings,
                )
            )
            time_values = np.asarray(da["time"].values)
            parsed_dates = [_as_calendar_date(value) for value in time_values]
            valid_dates = [value for value in parsed_dates if value is not None]
            lat_vals = np.asarray(da["lat"].values, dtype=float)
            lon_vals = np.asarray(da["lon"].values, dtype=float)
            coverage_by_var[name] = {
                "dims": {
                    "time": int(da.sizes.get("time", 0)),
                    "lat": int(da.sizes.get("lat", 0)),
                    "lon": int(da.sizes.get("lon", 0)),
                },
                "time_start": min(valid_dates).isoformat() if valid_dates else None,
                "time_end": max(valid_dates).isoformat() if valid_dates else None,
                "lat_min": float(np.nanmin(lat_vals)) if lat_vals.size else None,
                "lat_max": float(np.nanmax(lat_vals)) if lat_vals.size else None,
                "lon_min": float(np.nanmin(lon_vals)) if lon_vals.size else None,
                "lon_max": float(np.nanmax(lon_vals)) if lon_vals.size else None,
                "normalization": normalization,
            }
        if not compatible:
            climatology_reason = next(
                (
                    item.reason
                    for item in incompatible
                    if "climatology-style time axis" in str(item.reason)
                ),
                None,
            )
            if climatology_reason:
                raise ValueError(climatology_reason)
            reason_preview = "; ".join(
                f"{item.name}: {item.reason}" for item in incompatible[:3]
            )
            raise ValueError(
                "No compatible variable found in NetCDF. Expected a variable with time/lat/lon dimensions, "
                "optionally with singleton extra dimensions."
                + (f" Rejections: {reason_preview}" if reason_preview else "")
            )
        suggested = compatible[0]
        coverage = coverage_by_var.get(suggested, {})
        selected_option = next(
            (option for option in variable_options if option.name == suggested),
            FieldVariableOption(name=suggested),
        )
        normalization = coverage.get("normalization") or FieldNormalizationInfo()
        warnings.extend(selected_option.warnings)
        if str(dataset.attrs.get("_sdcpy_studio_time_decode_mode") or "") == "manual":
            warnings.append(
                InspectWarning(
                    code="time_axis_manually_decoded",
                    message="Decoded the NetCDF time axis using cftime compatibility fallback.",
                    columns=[suggested],
                )
            )
        return {
            "filename": filename,
            "variables": variables,
            "compatible_variables": compatible,
            "variable_options": variable_options,
            "incompatible_variables": incompatible,
            "suggested_variable": suggested,
            "dims": FieldDims.model_validate(dict(coverage.get("dims") or {})),
            "normalization": normalization,
            "warnings": warnings,
            "time_start": coverage.get("time_start"),
            "time_end": coverage.get("time_end"),
            "lat_min": coverage.get("lat_min"),
            "lat_max": coverage.get("lat_max"),
            "lon_min": coverage.get("lon_min"),
            "lon_max": coverage.get("lon_max"),
        }
    finally:
        try:
            dataset.close()
        except Exception:
            pass


def load_custom_map_field_subset(
    dataset_path: Path | str,
    *,
    variable: str,
    config,
    dimension_selections: dict[str, str] | None = None,
):
    dataset = _open_netcdf_dataset(dataset_path)
    try:
        if variable not in dataset.data_vars:
            available = ", ".join(str(name) for name in dataset.data_vars)
            raise ValueError(f"Field variable '{variable}' not found. Available variables: {available}.")
        da = _normalize_custom_field_dataarray(
            dataset[variable],
            apply_config=config,
            dimension_selections=dimension_selections,
        )
        return da.load()
    finally:
        try:
            dataset.close()
        except Exception:
            pass


def get_custom_field_bounds(
    dataset_path: Path | str,
    *,
    variable: str,
    dimension_selections: dict[str, str] | None = None,
) -> dict[str, float]:
    dataset = _open_netcdf_dataset(dataset_path)
    try:
        if variable not in dataset.data_vars:
            raise ValueError(f"Field variable '{variable}' not found in uploaded NetCDF.")
        da = _normalize_custom_field_dataarray(
            dataset[variable],
            dimension_selections=dimension_selections,
        )
        lat_vals = np.asarray(da["lat"].values, dtype=float)
        lon_vals = np.asarray(da["lon"].values, dtype=float)
        return {
            "lat_min": float(np.nanmin(lat_vals)),
            "lat_max": float(np.nanmax(lat_vals)),
            "lon_min": float(np.nanmin(lon_vals)),
            "lon_max": float(np.nanmax(lon_vals)),
        }
    finally:
        try:
            dataset.close()
        except Exception:
            pass


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


def export_sdc_map_artifact(job_result: dict, fmt: str, sign: str | None = None) -> tuple[bytes, str, str]:
    """Build downloadable output bytes from a finished SDC map job result."""
    artifacts = job_result.get("_artifacts_map")
    if not artifacts:
        raise ValueError("Download artifact metadata is unavailable for this map job.")

    driver_token = _sanitize_filename_token(artifacts.get("driver_dataset", "driver"), "driver")
    field_token = _sanitize_filename_token(artifacts.get("field_dataset", "field"), "field")
    correlation_width = int(artifacts.get("correlation_width", 0))
    sign_key = str(sign or "").strip().lower()
    if sign_key not in {"", "positive", "negative"}:
        raise ValueError("Unsupported map sign. Expected 'positive' or 'negative'.")

    basename = (
        f"sdcmap_{driver_token}_{field_token}_rw{correlation_width}_{sign_key}"
        if sign_key
        else f"sdcmap_{driver_token}_{field_token}_rw{correlation_width}_positive_negative"
    )

    fmt_key = fmt.lower().strip()
    if fmt_key == "png":
        if sign_key:
            payload = artifacts.get(f"png_{sign_key}") or artifacts["png"]
            return payload, "image/png", f"{basename}.png"
        return artifacts["png"], "image/png", f"{basename}.png"
    if fmt_key == "pdf":
        if sign_key:
            payload = artifacts.get(f"pdf_{sign_key}") or artifacts["pdf"]
            return payload, "application/pdf", f"{basename}.pdf"
        return artifacts["pdf"], "application/pdf", f"{basename}.pdf"
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


def _event_window_bounds(event_idx: int, width: int, series_len: int) -> tuple[int, int] | None:
    half_before = (int(width) - 1) // 2
    half_after = int(width) - 1 - half_before
    start = int(event_idx) - half_before
    stop = int(event_idx) + half_after + 1
    if start < 0 or stop > int(series_len):
        return None
    return start, stop


def _selected_event_count(event_catalog: dict[str, object] | None) -> int:
    catalog = event_catalog or {}
    return len(catalog.get("selected_positive") or []) + len(catalog.get("selected_negative") or [])


def _require_selected_map_events(event_catalog: dict[str, object] | None, *, action: str) -> None:
    if _selected_event_count(event_catalog) > 0:
        return
    raise ValueError(
        f"Select at least one positive or negative driver event before {action}. "
        "You can seed events with N+/N- or click the preview chart to add them manually."
    )


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


def _build_event_map_netcdf_bytes(
    class_results: dict[str, dict[str, object]],
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

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    dataset_attrs = dict(attrs)
    lag_values: np.ndarray | None = None
    for sign_key, class_result in class_results.items():
        layers = class_result.get("layers") or {}
        for key, values in layers.items():
            data_vars[f"{sign_key}_{key}"] = (("lat", "lon"), np.asarray(values, dtype=float))
        lag_maps = class_result.get("lag_maps") or {}
        class_lags = np.asarray(lag_maps.get("lags") or [], dtype=int)
        if class_lags.size:
            if lag_values is None:
                lag_values = class_lags
            elif not np.array_equal(lag_values, class_lags):
                raise ValueError("Positive and negative lag coordinates must match for NetCDF export.")
            corr_by_lag = np.asarray(lag_maps.get("corr_by_lag"), dtype=float)
            event_count_by_lag = np.asarray(lag_maps.get("event_count_by_lag"), dtype=float)
            data_vars[f"{sign_key}_corr_by_lag"] = (("lag", "lat", "lon"), corr_by_lag)
            data_vars[f"{sign_key}_event_count_by_lag"] = (("lag", "lat", "lon"), event_count_by_lag)

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": np.asarray(lats, dtype=float),
            "lon": np.asarray(lons, dtype=float),
            **({"lag": lag_values.astype(int)} if lag_values is not None else {}),
        },
        attrs=dataset_attrs,
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
    *,
    time_step_info: dict[str, str] | None = None,
) -> dict:
    unit_plural = str((time_step_info or {}).get("plural") or "time steps")
    payload_layers: list[dict] = []
    for spec in _MAP_LAYER_DEFS:
        key = str(spec["key"])
        if key not in layers:
            continue
        payload_layers.append(
            {
                "key": key,
                "label": spec["label"],
                "description": str(spec.get("description") or "").replace("{unit}", unit_plural),
                "unit_label": unit_plural,
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


def _build_map_lag_payload(
    lag_maps: dict[str, object],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline,
) -> dict:
    lags = np.asarray(lag_maps.get("lags") or [], dtype=int)
    corr_by_lag = lag_maps.get("corr_by_lag")
    event_count_by_lag = lag_maps.get("event_count_by_lag")
    corr_cube = (
        np.asarray(corr_by_lag, dtype=float)
        if corr_by_lag is not None
        else np.full((len(lags), len(lats), len(lons)), np.nan, dtype=float)
    )
    event_count_cube = (
        np.asarray(event_count_by_lag, dtype=float)
        if event_count_by_lag is not None
        else np.zeros((len(lags), len(lats), len(lons)), dtype=float)
    )
    return {
        "lat": [float(v) for v in np.asarray(lats, dtype=float).tolist()],
        "lon": [float(v) for v in np.asarray(lons, dtype=float).tolist()],
        "lags": [int(v) for v in lags.tolist()],
        "coastline": _serialize_coastline_trace(coastline),
        "corr_by_lag": _serialize_optional_cube(corr_cube),
        "event_count_by_lag": _serialize_optional_cube(event_count_cube),
    }


def _build_map_class_payloads(
    class_results: dict[str, dict[str, object]],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline,
    *,
    time_step_info: dict[str, str] | None = None,
) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for sign_key, class_result in class_results.items():
        payloads[sign_key] = {
            "summary_layers": _build_map_layer_payload(
                layers=class_result.get("layers") or {},
                lats=lats,
                lons=lons,
                coastline=coastline,
                time_step_info=time_step_info,
            ),
            "lag_maps": _build_map_lag_payload(
                lag_maps=class_result.get("lag_maps") or {},
                lats=lats,
                lons=lons,
                coastline=coastline,
            ),
        }
    return payloads


def _map_cell_bounds_from_layers(
    layers: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[int, float | None, float | None, float | None, float | None]:
    corr_mean = np.asarray(layers.get("corr_mean"), dtype=float)
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

    return valid_cells, field_lat_min, field_lat_max, field_lon_min, field_lon_max


def _build_map_class_summary(
    *,
    sign_key: str,
    class_result: dict[str, object],
    lats: np.ndarray,
    lons: np.ndarray,
) -> dict[str, object]:
    layers = class_result.get("layers") or {}
    valid_cells, field_lat_min, field_lat_max, field_lon_min, field_lon_max = _map_cell_bounds_from_layers(
        layers,
        lats,
        lons,
    )
    corr_mean = np.asarray(layers.get("corr_mean"), dtype=float)
    public_summary = dict(class_result.get("summary") or {})
    public_summary.update(
        {
            "sign": sign_key,
            "valid_cells": valid_cells,
            "total_cells": int(corr_mean.size),
            "field_lat_min": field_lat_min,
            "field_lat_max": field_lat_max,
            "field_lon_min": field_lon_min,
            "field_lon_max": field_lon_max,
        }
    )
    return public_summary


def _render_map_class_png(
    *,
    sign_label: str,
    lag_maps: dict[str, object],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline,
) -> bytes:
    from sdcpy_map import plot_correlation_maps_by_lag

    fig, *_ = plot_correlation_maps_by_lag(
        lag_maps=lag_maps,
        lats=lats,
        lons=lons,
        coastline=coastline,
        title=f"{sign_label} events: peak-averaged correlation by lag",
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
    return png_bytes


def _render_map_class_static_png(
    *,
    sign_label: str,
    layers: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline,
) -> bytes:
    from sdcpy_map import plot_layer_maps_compact

    fig, *_ = plot_layer_maps_compact(
        layers=layers,
        lats=lats,
        lons=lons,
        coastline=coastline,
        title=f"{sign_label} events · A/B/C/D",
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
    return png_bytes


def _resolve_report_event_index(
    event: dict[str, object],
    *,
    date_to_index: dict[str, int],
    series_len: int,
) -> int | None:
    raw_index = event.get("index")
    if raw_index is not None:
        try:
            event_index = int(raw_index)
        except (TypeError, ValueError):
            event_index = None
        else:
            if 0 <= event_index < series_len:
                return event_index
    event_date = str(event.get("date") or "").strip()
    if not event_date:
        return None
    return date_to_index.get(event_date)


def _render_map_driver_events_panel_png(
    *,
    sign_label: str,
    correlation_width: int,
    time_index: list[str],
    driver_values: list[float | None],
    event_catalog: dict[str, object],
    active_sign: str,
) -> bytes:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = pd.to_datetime(time_index)
    values = np.asarray(
        [float(value) if value is not None and np.isfinite(value) else np.nan for value in driver_values],
        dtype=float,
    )
    date_to_index = {stamp.date().isoformat(): idx for idx, stamp in enumerate(dates)}
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(dates, values, color="#22384d", linewidth=1.8)
    ax.axhline(0.0, color="#8b98a6", linewidth=0.9, linestyle="--")

    sign_order = ("positive", "negative")
    sign_colors = {
        "positive": {"fill": "#f3b2ae"},
        "negative": {"fill": "#bfd6ff"},
    }
    active_sign_key = str(active_sign).strip().lower()
    for sign_key in sign_order:
        events = list(event_catalog.get(f"selected_{sign_key}") or [])
        if not events:
            continue
        shade_alpha = 0.10 if sign_key == active_sign_key else 0.045
        for event in events:
            event_index = _resolve_report_event_index(event, date_to_index=date_to_index, series_len=len(dates))
            if event_index is None:
                continue
            bounds = _event_window_bounds(event_index, int(correlation_width), len(dates))
            if bounds is not None:
                start_idx, stop_idx = bounds
                ax.axvspan(
                    dates[start_idx],
                    dates[stop_idx - 1],
                    color=sign_colors[sign_key]["fill"],
                    alpha=shade_alpha,
                    linewidth=0,
                )

    ax.set_title(f"{sign_label} events · driver time series with selected event windows", fontsize=12)
    ax.set_ylabel("Driver")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout(pad=1.0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return buffer.getvalue()


def _combine_map_report_images(images: list[tuple[str, bytes]], *, fmt: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    valid_images = [(label, data) for label, data in images if data]
    if not valid_images:
        raise ValueError("No PNG images were available to combine.")

    fig, axes = plt.subplots(
        len(valid_images),
        1,
        figsize=(12, 6.8 * len(valid_images)),
        gridspec_kw={"hspace": 0.14},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for axis, (label, image_bytes) in zip(axes, valid_images, strict=False):
        axis.imshow(mpimg.imread(io.BytesIO(image_bytes), format="png"))
        axis.set_title(label, pad=10)
        axis.axis("off")

    fig.subplots_adjust(top=0.975, bottom=0.028, left=0.02, right=0.98, hspace=0.16)
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return buffer.getvalue()


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
    out: dict[str, Path] = {}
    _emit_progress(progress_hook, progress_start, progress_total, f"Ensuring driver dataset ({driver_key})")
    out["driver"] = fetch_sdc_map_driver_asset(
        data_dir,
        driver_key,
    )
    _emit_progress(progress_hook, progress_start + 1, progress_total, f"Ensuring field dataset ({field_key})")
    out["field"] = fetch_sdc_map_field_asset(
        data_dir,
        field_key,
    )
    if include_coastline:
        _emit_progress(progress_hook, progress_start + 2, progress_total, "Ensuring coastline dataset")
        out["coastline"] = fetch_sdc_map_coastline_asset(data_dir)

    return out


def fetch_sdc_map_driver_asset(data_dir: Path | str, driver_key: str) -> Path:
    coastline_url, driver_datasets, _field_datasets, download_if_missing = _load_map_dataset_registry()
    _ = coastline_url
    if driver_key not in driver_datasets:
        supported = ", ".join(sorted(driver_datasets))
        raise ValueError(f"Unknown driver dataset '{driver_key}'. Supported: {supported}.")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    driver_spec = driver_datasets[driver_key]
    driver_url = _canonical_psl_download_url(driver_spec.url)
    return _download_map_asset(
        download_if_missing,
        driver_url,
        data_dir / _map_asset_filename("driver", driver_key, driver_url),
    )


def fetch_sdc_map_field_asset(data_dir: Path | str, field_key: str) -> Path:
    coastline_url, _driver_datasets, field_datasets, download_if_missing = _load_map_dataset_registry()
    _ = coastline_url
    if field_key not in field_datasets:
        supported = ", ".join(sorted(field_datasets))
        raise ValueError(f"Unknown field dataset '{field_key}'. Supported: {supported}.")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    field_spec = field_datasets[field_key]
    field_url = _canonical_psl_download_url(field_spec.url)
    return _download_map_asset(
        download_if_missing,
        field_url,
        data_dir / _map_asset_filename("field", field_key, field_url),
    )


def fetch_sdc_map_coastline_asset(data_dir: Path | str) -> Path:
    coastline_url, _driver_datasets, _field_datasets, download_if_missing = _load_map_dataset_registry()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    source_url = _canonical_psl_download_url(coastline_url)
    return _download_map_asset(
        download_if_missing,
        source_url,
        data_dir / _map_asset_filename("coastline", "ne_110m", source_url),
    )


def _get_driver_data_coverage(driver_key: str, data_dir: Path | str | None = None) -> dict[str, object]:
    with _DRIVER_COVERAGE_CACHE_LOCK:
        cached = _DRIVER_COVERAGE_CACHE.get(driver_key)
    if cached is not None:
        return cached

    try:
        from sdcpy_map import load_driver_series
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    if data_dir is None:
        data_dir = _resolve_map_cache_dir()
    driver_path = fetch_sdc_map_driver_asset(data_dir=data_dir, driver_key=driver_key)
    driver = load_driver_series(
        driver_path,
        config=_full_driver_config(),
        driver_key=driver_key,
    )
    defaults = _derive_driver_defaults(driver)
    coverage = {
        "time_start": defaults["driver_min_date"],
        "time_end": defaults["driver_max_date"],
        "n_points": int(len(driver)),
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
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    if data_dir is None:
        data_dir = _resolve_map_cache_dir()

    field_path = fetch_sdc_map_field_asset(data_dir=data_dir, field_key=field_key)
    mapped_field = load_field_anomaly_subset(
        field_path,
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
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
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
    """Return default driver coverage and a preview of detected events."""
    try:
        from sdcpy_map import load_driver_series
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    _coastline_url, driver_datasets, _field_datasets, _download_if_missing = _load_map_dataset_registry()
    if driver_key not in driver_datasets:
        supported = ", ".join(sorted(driver_datasets))
        raise ValueError(f"Unknown driver dataset '{driver_key}'. Supported: {supported}.")

    data_dir = _resolve_map_cache_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    driver_path = fetch_sdc_map_driver_asset(data_dir=data_dir, driver_key=driver_key)

    full_driver = load_driver_series(
        driver_path,
        config=_full_driver_config(),
        driver_key=driver_key,
    )
    defaults = _derive_driver_defaults(full_driver)
    defaults["driver_dataset"] = driver_key
    defaults["event_catalog"] = _detect_driver_event_catalog(
        full_driver,
        correlation_width=12,
        n_positive_peaks=3,
        n_negative_peaks=3,
        base_state_beta=0.5,
    )
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
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    if data_dir is None:
        data_dir = _resolve_map_cache_dir()

    # Fetch the field dataset
    field_path = fetch_sdc_map_field_asset(data_dir=data_dir, field_key=field_key)

    # Load field with completely unconstrained bounds
    config = SDCMapConfig(
        lat_min=-90, lat_max=90,
        lon_min=-180, lon_max=180,
        lat_stride=1, lon_stride=1,
    )
    mapped_field = load_field_anomaly_subset(field_path, config=config, field_key=field_key)
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

    if request.field_source_type == "upload":
        if not request.field_upload_path or not request.field_variable:
            raise ValueError("Uploaded field source is missing resolved path or variable selection.")
        field_bounds = get_custom_field_bounds(
            request.field_upload_path,
            variable=request.field_variable,
            dimension_selections=request.field_dimension_selections,
        )
    else:
        field_bounds = _get_field_data_bounds(request.field_dataset, data_dir=data_dir)
    return (
        float(field_bounds["lat_min"]),
        float(field_bounds["lat_max"]),
        float(field_bounds["lon_min"]),
        float(field_bounds["lon_max"]),
        True,
    )


def _is_catalog_driver(request: SDCMapJobRequest) -> bool:
    return str(request.driver_source_type or "catalog") == "catalog"


def _is_catalog_field(request: SDCMapJobRequest) -> bool:
    return str(request.field_source_type or "catalog") == "catalog"


def _resolve_map_paths_for_request(
    request: SDCMapJobRequest,
    *,
    data_dir: Path,
    progress_hook: Callable[[int, int, str], None] | None = None,
    progress_start: int = 0,
    progress_total: int = 1,
) -> dict[str, Path]:
    """Resolve any catalog-backed assets plus coastline, using cache when both sources are catalog."""
    driver_catalog = _is_catalog_driver(request)
    field_catalog = _is_catalog_field(request)

    if driver_catalog and field_catalog:
        cache_key = (str(data_dir.resolve()), request.driver_dataset, request.field_dataset)
        with _MAP_DATA_PATH_CACHE_LOCK:
            cached_paths = _MAP_DATA_PATH_CACHE.get(cache_key)
        if cached_paths and all(path.exists() and path.stat().st_size > 0 for path in cached_paths.values()):
            return cached_paths
        paths = fetch_sdc_map_assets(
            data_dir=data_dir,
            driver_key=request.driver_dataset,
            field_key=request.field_dataset,
            progress_hook=progress_hook,
            progress_start=progress_start,
            progress_total=progress_total,
        )
        with _MAP_DATA_PATH_CACHE_LOCK:
            _MAP_DATA_PATH_CACHE[cache_key] = paths
        return paths

    paths: dict[str, Path] = {
        "coastline": fetch_sdc_map_coastline_asset(data_dir),
    }
    if driver_catalog:
        _emit_progress(progress_hook, progress_start, progress_total, f"Ensuring driver dataset ({request.driver_dataset})")
        paths["driver"] = fetch_sdc_map_driver_asset(data_dir, request.driver_dataset)
    if field_catalog:
        field_progress = progress_start + (1 if driver_catalog else 0)
        _emit_progress(progress_hook, field_progress, progress_total, f"Ensuring field dataset ({request.field_dataset})")
        paths["field"] = fetch_sdc_map_field_asset(data_dir, request.field_dataset)
    return paths


def _resolve_map_time_window(
    request: SDCMapJobRequest,
    driver_full: pd.Series,
) -> tuple[str, str]:
    defaults = _derive_driver_defaults(driver_full)
    time_start = _as_calendar_date(request.time_start or defaults["time_start"])
    time_end = _as_calendar_date(request.time_end or defaults["time_end"])
    if time_start is None or time_end is None:
        raise ValueError("`time_start` and `time_end` must be valid dates.")
    cadence = _infer_series_cadence(pd.DatetimeIndex(driver_full.dropna().sort_index().index))
    time_start, time_end = _canonicalize_window_bounds(time_start, time_end, cadence=cadence)
    if time_start > time_end:
        raise ValueError("`time_start` must be <= `time_end`.")
    return time_start.isoformat(), time_end.isoformat()


def _slice_driver_to_window(
    driver_full: pd.Series,
    *,
    time_start: str,
    time_end: str,
) -> pd.Series:
    driver = driver_full.sort_index()
    driver = driver.loc[driver.index >= pd.Timestamp(time_start)]
    driver = driver.loc[driver.index <= pd.Timestamp(time_end)]
    if driver.empty:
        raise ValueError("Selected driver time window contains no data.")
    return driver


def build_sdc_map_driver_preview(payload: dict) -> dict:
    """Return a lightweight driver-only preview for event selection controls."""
    request = SDCMapJobRequest.model_validate(payload)
    try:
        from sdcpy_map import load_driver_series
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    data_dir = _resolve_map_cache_dir()
    if _is_catalog_driver(request):
        paths = _resolve_map_paths_for_request(request, data_dir=data_dir, progress_total=1)
        driver_full = load_driver_series(
            paths["driver"],
            config=_full_driver_config(),
            driver_key=request.driver_dataset,
        )
    else:
        if not request.driver_upload_path or not request.driver_date_column or not request.driver_value_column:
            raise ValueError("Uploaded driver is missing file path or selected columns.")
        driver_full = load_custom_map_driver_series(
            request.driver_upload_path,
            date_column=request.driver_date_column,
            value_column=request.driver_value_column,
        )

    time_start, time_end = _resolve_map_time_window(request, driver_full)
    driver = _slice_driver_to_window(driver_full, time_start=time_start, time_end=time_end)
    manual_event_selection = (
        request.manual_event_selection.model_dump(mode="python")
        if request.manual_event_selection is not None
        else None
    )
    event_catalog = _detect_driver_event_catalog(
        driver,
        correlation_width=request.correlation_width,
        n_positive_peaks=request.n_positive_peaks,
        n_negative_peaks=request.n_negative_peaks,
        base_state_beta=request.base_state_beta,
        manual_event_selection=manual_event_selection,
    )
    return {
        "summary": {
            "driver_dataset": request.driver_upload_filename or request.driver_dataset,
            "driver_source_type": request.driver_source_type,
            "time_start": time_start,
            "time_end": time_end,
            "correlation_width": int(request.correlation_width),
            "n_positive_peaks": int(request.n_positive_peaks),
            "n_negative_peaks": int(request.n_negative_peaks),
            "base_state_beta": float(request.base_state_beta),
            "n_points": int(len(driver)),
        },
        "event_catalog": event_catalog,
        "time_index": _serialize_time_labels(np.asarray(driver.index.to_numpy())),
        "driver_values": _serialize_optional_array(driver.to_numpy(dtype=float)),
    }


def _compute_sdcmap_event_layers_with_progress(
    *,
    driver: pd.Series,
    mapped_field,
    config,
    manual_event_selection: dict[str, object] | None,
    progress_hook: Callable[[int, int, str], None] | None,
    cancel_event: Event | None,
    progress_base_current: int,
    progress_total: int,
) -> tuple[dict[str, object], int]:
    from sdcpy_map import compute_sdcmap_event_layers

    nlat = int(mapped_field.sizes.get("lat", 0))
    nlon = int(mapped_field.sizes.get("lon", 0))
    total_cells = max(0, nlat * nlon)

    def _cell_progress(completed_cells: int, expected_total_cells: int) -> None:
        _raise_if_cancelled(cancel_event, "SDC map run cancelled during cell computation.")
        if expected_total_cells <= 0:
            return
        current = min(progress_total - 3, progress_base_current + int(completed_cells))
        _emit_progress(
            progress_hook,
            current,
            progress_total,
            f"Computing event-conditioned SDC map ({expected_total_cells} cells)",
        )

    _emit_progress(
        progress_hook,
        progress_base_current,
        progress_total,
        f"Computing event-conditioned SDC map ({total_cells} cells)",
    )
    _raise_if_cancelled(cancel_event, "SDC map run cancelled before cell computation started.")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        result = compute_sdcmap_event_layers(
            driver=driver,
            mapped_field=mapped_field,
            config=config,
            manual_event_selection=manual_event_selection,
            progress_callback=_cell_progress if progress_hook else None,
        )

    _emit_progress(
        progress_hook,
        max(progress_base_current, progress_total - 3),
        progress_total,
        "Event-conditioned map layers computed",
    )
    _raise_if_cancelled(cancel_event, "SDC map run cancelled after cell computation.")
    return result, total_cells


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
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    data_dir = _resolve_map_cache_dir()
    lat_min, lat_max, lon_min, lon_max, using_full_bounds = _resolve_map_bounds(
        request,
        data_dir=data_dir,
    )
    paths = _resolve_map_paths_for_request(request, data_dir=data_dir)

    if _is_catalog_driver(request):
        full_driver = load_driver_series(
            paths["driver"],
            config=_full_driver_config(),
            driver_key=request.driver_dataset,
        )
    else:
        if not request.driver_upload_path or not request.driver_date_column or not request.driver_value_column:
            raise ValueError("Uploaded driver is missing file path or selected columns.")
        full_driver = load_custom_map_driver_series(
            request.driver_upload_path,
            date_column=request.driver_date_column,
            value_column=request.driver_value_column,
        )
    time_start, time_end = _resolve_map_time_window(request, full_driver)

    config = SDCMapConfig(
        correlation_width=request.correlation_width,
        n_positive_peaks=request.n_positive_peaks,
        n_negative_peaks=request.n_negative_peaks,
        base_state_beta=request.base_state_beta,
        n_permutations=request.n_permutations,
        two_tailed=request.two_tailed,
        min_lag=request.min_lag,
        max_lag=request.max_lag,
        alpha=request.alpha,
        time_start=time_start,
        time_end=time_end,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_stride=request.lat_stride,
        lon_stride=request.lon_stride,
    )

    if _is_catalog_driver(request):
        driver = load_driver_series(paths["driver"], config=config, driver_key=request.driver_dataset)
    else:
        driver = load_custom_map_driver_series(
            request.driver_upload_path or "",
            date_column=request.driver_date_column or "",
            value_column=request.driver_value_column or "",
            time_start=time_start,
            time_end=time_end,
        )
    if _is_catalog_field(request):
        mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=request.field_dataset)
    else:
        if not request.field_upload_path or not request.field_variable:
            raise ValueError("Uploaded field is missing file path or selected variable.")
        mapped_field = load_custom_map_field_subset(
            request.field_upload_path,
            variable=request.field_variable,
            config=config,
            dimension_selections=request.field_dimension_selections,
        )
    driver = align_driver_to_field(driver, mapped_field)
    coastline = load_coastline(paths["coastline"])
    lats, lons = grid_coordinates(mapped_field)
    manual_event_selection = (
        request.manual_event_selection.model_dump(mode="python")
        if request.manual_event_selection is not None
        else None
    )
    event_catalog = _detect_driver_event_catalog(
        driver,
        correlation_width=request.correlation_width,
        n_positive_peaks=request.n_positive_peaks,
        n_negative_peaks=request.n_negative_peaks,
        base_state_beta=request.base_state_beta,
        manual_event_selection=manual_event_selection,
    )
    _require_selected_map_events(event_catalog, action="loading the dataset")

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
            "driver_dataset": request.driver_upload_filename or request.driver_dataset,
            "field_dataset": request.field_upload_filename or request.field_dataset,
            "driver_source_type": request.driver_source_type,
            "field_source_type": request.field_source_type,
            "field_variable": request.field_variable if request.field_source_type == "upload" else request.field_dataset,
            "field_dimension_selections": (
                dict(request.field_dimension_selections)
                if request.field_source_type == "upload"
                else {}
            ),
            "time_start": time_start,
            "time_end": time_end,
            "correlation_width": int(request.correlation_width),
            "n_positive_peaks": int(request.n_positive_peaks),
            "n_negative_peaks": int(request.n_negative_peaks),
            "base_state_beta": float(request.base_state_beta),
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
            "selected_positive_events": len(event_catalog["selected_positive"]),
            "selected_negative_events": len(event_catalog["selected_negative"]),
            "base_state_count": int(event_catalog["base_state_count"]),
            "base_state_threshold": event_catalog["base_state_threshold"],
        },
        "event_catalog": event_catalog,
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
    cancel_event: Event | None = None,
) -> dict:
    """Run an event-conditioned SDC map job and return a JSON-friendly payload."""
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
            derive_compact_layers,
            grid_coordinates,
            load_coastline,
            load_driver_series,
            load_field_anomaly_subset,
        )
    except ImportError as exc:
        raise ValueError(
            "SDC Map dependencies are unavailable in this environment. Install project dependencies with `uv sync`."
        ) from exc

    initial_progress_total = 8
    _emit_progress(progress_hook, 0, initial_progress_total, "Preparing SDC map inputs")
    _raise_if_cancelled(cancel_event, "SDC map run cancelled before preparation.")
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
        f"Ensuring map assets ({request.driver_dataset} / {request.field_dataset})",
    )
    _raise_if_cancelled(cancel_event, "SDC map run cancelled before asset resolution.")
    paths = _resolve_map_paths_for_request(
        request,
        data_dir=data_dir,
        progress_hook=progress_hook,
        progress_start=1,
        progress_total=initial_progress_total,
    )
    _raise_if_cancelled(cancel_event, "SDC map run cancelled after asset resolution.")

    if _is_catalog_driver(request):
        full_driver = load_driver_series(
            paths["driver"],
            config=_full_driver_config(),
            driver_key=request.driver_dataset,
        )
    else:
        if not request.driver_upload_path or not request.driver_date_column or not request.driver_value_column:
            raise ValueError("Uploaded driver is missing file path or selected columns.")
        full_driver = load_custom_map_driver_series(
            request.driver_upload_path,
            date_column=request.driver_date_column,
            value_column=request.driver_value_column,
        )
    time_start, time_end = _resolve_map_time_window(request, full_driver)
    config = SDCMapConfig(
        correlation_width=request.correlation_width,
        n_positive_peaks=request.n_positive_peaks,
        n_negative_peaks=request.n_negative_peaks,
        base_state_beta=request.base_state_beta,
        n_permutations=request.n_permutations,
        two_tailed=request.two_tailed,
        min_lag=request.min_lag,
        max_lag=request.max_lag,
        alpha=request.alpha,
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
    _raise_if_cancelled(cancel_event, "SDC map run cancelled before loading datasets.")
    if _is_catalog_driver(request):
        driver = load_driver_series(paths["driver"], config=config, driver_key=request.driver_dataset)
    else:
        driver = load_custom_map_driver_series(
            request.driver_upload_path or "",
            date_column=request.driver_date_column or "",
            value_column=request.driver_value_column or "",
            time_start=time_start,
            time_end=time_end,
        )
    if _is_catalog_field(request):
        mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=request.field_dataset)
    else:
        if not request.field_upload_path or not request.field_variable:
            raise ValueError("Uploaded field is missing file path or selected variable.")
        mapped_field = load_custom_map_field_subset(
            request.field_upload_path,
            variable=request.field_variable,
            config=config,
            dimension_selections=request.field_dimension_selections,
        )
    _raise_if_cancelled(cancel_event, "SDC map run cancelled during dataset loading.")
    driver = align_driver_to_field(driver, mapped_field)
    _raise_if_cancelled(cancel_event, "SDC map run cancelled during series alignment.")
    lats, lons = grid_coordinates(mapped_field)
    time_step_info = _infer_time_step_info(mapped_field["time"].values)
    manual_event_selection = (
        request.manual_event_selection.model_dump(mode="python")
        if request.manual_event_selection is not None
        else None
    )
    preflight_event_catalog = _detect_driver_event_catalog(
        driver,
        correlation_width=request.correlation_width,
        n_positive_peaks=request.n_positive_peaks,
        n_negative_peaks=request.n_negative_peaks,
        base_state_beta=request.base_state_beta,
        manual_event_selection=manual_event_selection,
    )
    _require_selected_map_events(preflight_event_catalog, action="running the map")

    n_lat = int(mapped_field.sizes.get("lat", 0))
    n_lon = int(mapped_field.sizes.get("lon", 0))
    total_cells = max(1, n_lat * n_lon)
    progress_total = total_cells + 7
    _emit_progress(progress_hook, 4, progress_total, f"Grid ready ({n_lat}x{n_lon}). Starting cell loop")
    event_result, _ = _compute_sdcmap_event_layers_with_progress(
        driver=driver,
        mapped_field=mapped_field,
        config=config,
        manual_event_selection=manual_event_selection,
        progress_hook=progress_hook,
        cancel_event=cancel_event,
        progress_base_current=4,
        progress_total=progress_total,
    )
    _raise_if_cancelled(cancel_event, "SDC map run cancelled after cell computation.")
    coastline = load_coastline(paths["coastline"])
    compact_layers = derive_compact_layers(event_result)
    public_catalog = _public_event_catalog(event_result.get("event_catalog"))
    class_layer_maps = _build_map_class_payloads(
        {
            "positive": dict(event_result["positive"]),
            "negative": dict(event_result["negative"]),
        },
        lats=lats,
        lons=lons,
        coastline=coastline,
        time_step_info=time_step_info,
    )

    _emit_progress(progress_hook, progress_total - 2, progress_total, "Rendering event-class figures")
    _raise_if_cancelled(cancel_event, "SDC map run cancelled before figure rendering.")
    positive_static_png = _render_map_class_static_png(
        sign_label="Positive",
        layers=event_result["positive"]["layers"],
        lats=lats,
        lons=lons,
        coastline=coastline,
    )
    negative_static_png = _render_map_class_static_png(
        sign_label="Negative",
        layers=event_result["negative"]["layers"],
        lats=lats,
        lons=lons,
        coastline=coastline,
    )
    driver_time_index = _serialize_time_labels(np.asarray(driver.index.to_numpy()))
    driver_series_values = _serialize_optional_array(driver.to_numpy(dtype=float))
    positive_driver_panel_png = _render_map_driver_events_panel_png(
        sign_label="Positive",
        correlation_width=int(request.correlation_width),
        time_index=driver_time_index,
        driver_values=driver_series_values,
        event_catalog=public_catalog,
        active_sign="positive",
    )
    negative_driver_panel_png = _render_map_driver_events_panel_png(
        sign_label="Negative",
        correlation_width=int(request.correlation_width),
        time_index=driver_time_index,
        driver_values=driver_series_values,
        event_catalog=public_catalog,
        active_sign="negative",
    )
    positive_report_png = _combine_map_report_images(
        [
            ("Positive driver events", positive_driver_panel_png),
            ("Positive events · A/B/C/D", positive_static_png),
        ],
        fmt="png",
    )
    negative_report_png = _combine_map_report_images(
        [
            ("Negative driver events", negative_driver_panel_png),
            ("Negative events · A/B/C/D", negative_static_png),
        ],
        fmt="png",
    )
    positive_report_pdf = _combine_map_report_images(
        [
            ("Positive driver events", positive_driver_panel_png),
            ("Positive events · A/B/C/D", positive_static_png),
        ],
        fmt="pdf",
    )
    negative_report_pdf = _combine_map_report_images(
        [
            ("Negative driver events", negative_driver_panel_png),
            ("Negative events · A/B/C/D", negative_static_png),
        ],
        fmt="pdf",
    )
    combined_report_png = _combine_map_report_images(
        [
            ("Positive map report", positive_report_png),
            ("Negative map report", negative_report_png),
        ],
        fmt="png",
    )
    combined_report_pdf = _combine_map_report_images(
        [
            ("Positive map report", positive_report_png),
            ("Negative map report", negative_report_png),
        ],
        fmt="pdf",
    )
    png_bytes = combined_report_png

    _emit_progress(progress_hook, progress_total - 1, progress_total, "Packing outputs")
    _raise_if_cancelled(cancel_event, "SDC map run cancelled before packing outputs.")
    nc_bytes = _build_event_map_netcdf_bytes(
        class_results={
            "positive": dict(event_result["positive"]),
            "negative": dict(event_result["negative"]),
        },
        lats=lats,
        lons=lons,
        attrs={
            "driver_dataset": request.driver_upload_filename or request.driver_dataset,
            "field_dataset": request.field_upload_filename or request.field_dataset,
            "correlation_width": int(request.correlation_width),
            "n_positive_peaks": int(request.n_positive_peaks),
            "n_negative_peaks": int(request.n_negative_peaks),
            "base_state_beta": float(request.base_state_beta),
            "alpha": float(request.alpha),
            "min_lag": int(request.min_lag),
            "max_lag": int(request.max_lag),
            "base_state_threshold": float(public_catalog["base_state_threshold"])
            if public_catalog["base_state_threshold"] is not None
            else "",
            "selected_positive_events_json": str(public_catalog["selected_positive"]),
            "selected_negative_events_json": str(public_catalog["selected_negative"]),
        },
    )

    runtime_seconds = perf_counter() - started
    corr_mean = np.asarray(compact_layers["corr_mean"], dtype=float)
    total_cells_result = int(corr_mean.size)
    valid_cells, field_lat_min, field_lat_max, field_lon_min, field_lon_max = _map_cell_bounds_from_layers(
        compact_layers,
        lats,
        lons,
    )
    mean_abs_corr = (
        float(np.nanmean(np.abs(corr_mean))) if np.isfinite(corr_mean).any() else None
    )

    class_results_payload: dict[str, dict[str, object]] = {}
    for sign_key in ("positive", "negative"):
        class_result = dict(event_result[sign_key])
        class_summary = _build_map_class_summary(
            sign_key=sign_key,
            class_result=class_result,
            lats=lats,
            lons=lons,
        )
        selected_event_count = int(class_summary.get("selected_event_count") or 0)
        valid_class_cells = int(class_summary.get("valid_cells") or 0)
        empty_reason = None
        if selected_event_count == 0:
            empty_reason = f"No usable {sign_key} driver events were detected for the selected time window."
        elif valid_class_cells == 0:
            empty_reason = f"No valid {sign_key} map cells passed filtering with the current parameters."
        class_results_payload[sign_key] = {
            "summary": class_summary,
            "events": list(class_result.get("events") or []),
            "layer_maps": class_layer_maps[sign_key].get("summary_layers") or {},
            "lag_maps": class_layer_maps[sign_key].get("lag_maps") or {},
            "empty_reason": empty_reason,
        }

    notes: list[str] = []
    if valid_cells == 0:
        notes.append("No valid grid cells passed filtering with the current parameters.")
    if int(class_results_payload["positive"]["summary"].get("selected_event_count") or 0) == 0:
        notes.append("No usable positive driver events were available in the selected window.")
    if int(class_results_payload["negative"]["summary"].get("selected_event_count") or 0) == 0:
        notes.append("No usable negative driver events were available in the selected window.")
    if request.n_permutations > 99:
        notes.append("High permutation count selected; map runs may take substantially longer.")
    if using_full_bounds:
        notes.append("Full geographic map was computed; this mode is the most computationally expensive.")
    notes.extend(str(item) for item in public_catalog["warnings"])

    _emit_progress(progress_hook, progress_total, progress_total, "Completed")
    layer_maps = _build_map_layer_payload(
        layers=compact_layers,
        lats=lats,
        lons=lons,
        coastline=coastline,
        time_step_info=time_step_info,
    )
    return {
        "summary": {
            "driver_dataset": request.driver_upload_filename or request.driver_dataset,
            "field_dataset": request.field_upload_filename or request.field_dataset,
            "driver_source_type": request.driver_source_type,
            "field_source_type": request.field_source_type,
            "field_variable": request.field_variable if request.field_source_type == "upload" else request.field_dataset,
            "field_dimension_selections": (
                dict(request.field_dimension_selections)
                if request.field_source_type == "upload"
                else {}
            ),
            "time_start": time_start,
            "time_end": time_end,
            "correlation_width": int(request.correlation_width),
            "n_positive_peaks": int(request.n_positive_peaks),
            "n_negative_peaks": int(request.n_negative_peaks),
            "base_state_beta": float(request.base_state_beta),
            "n_permutations": int(request.n_permutations),
            "alpha": float(request.alpha),
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
            "mean_abs_corr": mean_abs_corr,
            "full_bounds_selected": using_full_bounds,
            "selected_positive_events": len(public_catalog["selected_positive"]),
            "selected_negative_events": len(public_catalog["selected_negative"]),
            "base_state_count": int(public_catalog["base_state_count"]),
            "base_state_threshold": public_catalog["base_state_threshold"],
            "time_step_unit_singular": time_step_info["singular"],
            "time_step_unit_plural": time_step_info["plural"],
        },
        "event_catalog": public_catalog,
        "class_results": class_results_payload,
        "notes": notes,
        "runtime_seconds": float(runtime_seconds),
        "figure_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "layer_maps": layer_maps,
        "download_formats": ["png", "pdf", "nc"],
        "_artifacts_map": {
            "png": combined_report_png,
            "png_positive": positive_report_png,
            "png_negative": negative_report_png,
            "pdf": combined_report_pdf,
            "pdf_positive": positive_report_pdf,
            "pdf_negative": negative_report_pdf,
            "nc": nc_bytes,
            "driver_dataset": request.driver_dataset,
            "field_dataset": request.field_dataset,
            "correlation_width": int(request.correlation_width),
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
