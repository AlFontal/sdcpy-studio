"""Service-layer helpers for sdcpy-studio computations and exports."""

from __future__ import annotations

import base64
import io
import os
import re
from collections.abc import Callable
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

    out: dict[str, Path] = {}
    _emit_progress(progress_hook, progress_start, progress_total, f"Ensuring driver dataset ({driver_key})")
    out["driver"] = download_if_missing(
        driver_url,
        data_dir / _map_asset_filename("driver", driver_key, driver_url),
    )
    _emit_progress(progress_hook, progress_start + 1, progress_total, f"Ensuring field dataset ({field_key})")
    out["field"] = download_if_missing(
        field_url,
        data_dir / _map_asset_filename("field", field_key, field_url),
    )
    if include_coastline:
        _emit_progress(progress_hook, progress_start + 2, progress_total, "Ensuring coastline dataset")
        out["coastline"] = download_if_missing(
            coastline_url,
            data_dir / _map_asset_filename("coastline", "ne_110m", coastline_url),
        )

    return out


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

    config = SDCMapConfig(
        fragment_size=request.fragment_size,
        n_permutations=request.n_permutations,
        two_tailed=request.two_tailed,
        min_lag=request.min_lag,
        max_lag=request.max_lag,
        alpha=request.alpha,
        top_fraction=request.top_fraction,
        peak_date=request.peak_date,
        time_start=request.time_start,
        time_end=request.time_end,
        lat_min=request.lat_min,
        lat_max=request.lat_max,
        lon_min=request.lon_min,
        lon_max=request.lon_max,
        lat_stride=request.lat_stride,
        lon_stride=request.lon_stride,
    )

    data_dir = Path(
        os.getenv(
            "SDCPY_STUDIO_SDCPY_MAP_DATA_DIR",
            str(Path.home() / ".cache" / "sdcpy-studio" / "sdcpy-map"),
        )
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

    driver = load_driver_series(paths["driver"], config=config, driver_key=request.driver_dataset)
    mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=request.field_dataset)
    driver = align_driver_to_field(driver, mapped_field)
    coastline = load_coastline(paths["coastline"])
    lats, lons = grid_coordinates(mapped_field)

    time_index = pd.to_datetime(mapped_field["time"].values)
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
            "time_start": request.time_start,
            "time_end": request.time_end,
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
        },
        "time_index": [ts.strftime("%Y-%m-%d") for ts in time_index],
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
            compute_sdcmap_layers,
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

    total_steps = 8
    _emit_progress(progress_hook, 0, total_steps, "Preparing SDC map inputs")
    config = SDCMapConfig(
        fragment_size=request.fragment_size,
        n_permutations=request.n_permutations,
        two_tailed=request.two_tailed,
        min_lag=request.min_lag,
        max_lag=request.max_lag,
        alpha=request.alpha,
        top_fraction=request.top_fraction,
        peak_date=request.peak_date,
        time_start=request.time_start,
        time_end=request.time_end,
        lat_min=request.lat_min,
        lat_max=request.lat_max,
        lon_min=request.lon_min,
        lon_max=request.lon_max,
        lat_stride=request.lat_stride,
        lon_stride=request.lon_stride,
    )

    data_dir = Path(
        os.getenv(
            "SDCPY_STUDIO_SDCPY_MAP_DATA_DIR",
            str(Path.home() / ".cache" / "sdcpy-studio" / "sdcpy-map"),
        )
    )

    started = perf_counter()
    _emit_progress(progress_hook, 1, total_steps, f"Ensuring driver dataset ({request.driver_dataset})")
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
            total_steps,
            f"Using in-memory driver cache ({request.driver_dataset})",
        )
        _emit_progress(
            progress_hook,
            2,
            total_steps,
            f"Using in-memory field cache ({request.field_dataset})",
        )
        _emit_progress(progress_hook, 3, total_steps, "Using in-memory coastline cache")
        paths = cached_paths
    else:
        paths = fetch_sdc_map_assets(
            data_dir=data_dir,
            driver_key=request.driver_dataset,
            field_key=request.field_dataset,
            progress_hook=progress_hook,
            progress_start=1,
            progress_total=total_steps,
        )
        with _MAP_DATA_PATH_CACHE_LOCK:
            _MAP_DATA_PATH_CACHE[cache_key] = paths

    _emit_progress(progress_hook, 4, total_steps, "Loading and aligning series")
    driver = load_driver_series(paths["driver"], config=config, driver_key=request.driver_dataset)
    mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=request.field_dataset)
    driver = align_driver_to_field(driver, mapped_field)

    _emit_progress(progress_hook, 5, total_steps, "Computing map layers")
    layers = compute_sdcmap_layers(driver=driver, mapped_field=mapped_field, config=config)
    lats, lons = grid_coordinates(mapped_field)

    _emit_progress(progress_hook, 6, total_steps, "Rendering map figure")
    coastline = load_coastline(paths["coastline"])
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

    _emit_progress(progress_hook, 7, total_steps, "Packing outputs")
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
    total_cells = int(corr_mean.size)
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

    _emit_progress(progress_hook, total_steps, total_steps, "Completed")
    layer_maps = _build_map_layer_payload(layers=layers, lats=lats, lons=lons, coastline=coastline)
    return {
        "summary": {
            "driver_dataset": request.driver_dataset,
            "field_dataset": request.field_dataset,
            "time_start": request.time_start,
            "time_end": request.time_end,
            "peak_date": request.peak_date,
            "fragment_size": int(request.fragment_size),
            "n_permutations": int(request.n_permutations),
            "alpha": float(request.alpha),
            "top_fraction": float(request.top_fraction),
            "min_lag": int(request.min_lag),
            "max_lag": int(request.max_lag),
            "lat_min": float(request.lat_min),
            "lat_max": float(request.lat_max),
            "lon_min": float(request.lon_min),
            "lon_max": float(request.lon_max),
            "lat_stride": int(request.lat_stride),
            "lon_stride": int(request.lon_stride),
            "n_time": int(mapped_field.sizes.get("time", 0)),
            "n_lat": int(mapped_field.sizes.get("lat", 0)),
            "n_lon": int(mapped_field.sizes.get("lon", 0)),
            "valid_cells": valid_cells,
            "valid_cell_rate": float(valid_cells / total_cells) if total_cells else 0.0,
            "field_lat_min": field_lat_min,
            "field_lat_max": field_lat_max,
            "field_lon_min": field_lon_min,
            "field_lon_max": field_lon_max,
            "positive_dominant_cells": positive_cells,
            "negative_dominant_cells": negative_cells,
            "mean_abs_corr": mean_abs_corr,
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

    r_matrix = _downsample_matrix(r_matrix_full, request.heatmap_step)
    p_matrix = _downsample_matrix(p_matrix_full, request.heatmap_step)
    p_matrix = p_matrix.reindex(index=r_matrix.index, columns=r_matrix.columns)
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
        "strongest_links": strongest.to_dict(orient="records") if not strongest.empty else [],
        "notes": notes,
        "runtime_seconds": float(runtime_seconds),
        "_artifacts": _serialize_artifacts(request, valid),
    }
