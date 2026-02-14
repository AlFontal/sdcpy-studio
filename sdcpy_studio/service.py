"""Service-layer helpers for sdcpy-studio computations and exports."""

from __future__ import annotations

import io
import os
import re
from collections.abc import Callable
from time import perf_counter

import numpy as np
import pandas as pd
import sdcpy.core as sdc_core
from sdcpy import compute_sdc
from sdcpy.scale_dependent_correlation import SDCAnalysis

from sdcpy_studio.schemas import SDCJobFromDatasetRequest, SDCJobRequest

MAX_STRONGEST_LINKS = 100


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
