"""Service-layer helpers for heavy studio computations."""

from __future__ import annotations

import io
import math
import os
from time import perf_counter

import numpy as np
import pandas as pd
from sdcpy import compute_sdc
from sdcpy.scale_dependent_correlation import SDCAnalysis

from sdcpy_studio.schemas import SDCJobRequest

MAX_HEATMAP_SIDE = 120
MAX_STRONGEST_LINKS = 100


def parse_series_csv(content: bytes) -> list[float]:
    """Parse a CSV payload into a numeric series.

    Accepted layouts:
    - one numeric column,
    - a `value` column,
    - a two-column table where second column is numeric.
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


def _downsample_matrix(matrix: pd.DataFrame, max_side: int = MAX_HEATMAP_SIDE) -> pd.DataFrame:
    if matrix.empty:
        return matrix

    step_rows = max(1, math.ceil(matrix.shape[0] / max_side))
    step_cols = max(1, math.ceil(matrix.shape[1] / max_side))
    return matrix.iloc[::step_rows, ::step_cols]


def _build_ranges_panel(sdc_df: pd.DataFrame, ts1: np.ndarray, ts2: np.ndarray, fragment_size: int) -> dict:
    """Compute a compact `get_ranges_df` side-panel payload for TS1 bins."""
    ts1_series = pd.Series(ts1, index=np.arange(len(ts1)))
    ts2_series = pd.Series(ts2, index=np.arange(len(ts2)))

    sdc_augmented = sdc_df.copy()
    index_1 = ts1_series.index.to_numpy()
    index_2 = ts2_series.index.to_numpy()
    sdc_augmented["date_1"] = sdc_augmented["start_1"].astype(int).map(lambda i: index_1[i])
    sdc_augmented["date_2"] = sdc_augmented["start_2"].astype(int).map(lambda i: index_2[i])

    analysis = SDCAnalysis(
        ts1=ts1_series,
        ts2=ts2_series,
        fragment_size=fragment_size,
        sdc_df=sdc_augmented,
    )
    ranges = analysis.get_ranges_df(
        ts=1,
        alpha=0.05,
        agg_func="mean",
        bin_size=max(float(np.nanstd(ts1)) / 8.0, 0.1),
    )

    if ranges.empty:
        return {
            "bin_center": [],
            "positive_freq": [],
            "negative_freq": [],
            "ns_freq": [],
        }

    ranges = ranges.copy()
    ranges["bin_center"] = ranges["cat_value"].map(
        lambda interval: float((interval.left + interval.right) / 2.0)
    )
    pivot = ranges.pivot_table(
        index="bin_center",
        columns="direction",
        values="freq",
        fill_value=0.0,
        observed=False,
    ).sort_index()

    for column_name in ("Positive", "Negative", "NS"):
        if column_name not in pivot.columns:
            pivot[column_name] = 0.0

    return {
        "bin_center": [float(v) for v in pivot.index.to_numpy()],
        "positive_freq": [float(v) for v in pivot["Positive"].to_numpy()],
        "negative_freq": [float(v) for v in pivot["Negative"].to_numpy()],
        "ns_freq": [float(v) for v in pivot["NS"].to_numpy()],
    }


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


def run_sdc_job(payload: dict) -> dict:
    """Run a full SDC job and return a compact JSON-friendly payload."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    request = SDCJobRequest.model_validate(payload)

    ts1 = np.asarray(request.ts1, dtype=float)
    ts2 = np.asarray(request.ts2, dtype=float)

    started = perf_counter()
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
    runtime_seconds = perf_counter() - started

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

    r_matrix = _downsample_matrix(r_matrix_full)
    p_matrix = _downsample_matrix(p_matrix_full)
    p_matrix = p_matrix.reindex(index=r_matrix.index, columns=r_matrix.columns)

    ranges_panel = _build_ranges_panel(
        valid,
        ts1=ts1,
        ts2=ts2,
        fragment_size=request.fragment_size,
    )

    summary = {
        "series_length": int(len(ts1)),
        "fragment_size": int(request.fragment_size),
        "n_pairs": int(len(valid)),
        "n_significant": int(len(significant)),
        "significant_rate": float(len(significant) / len(valid)) if len(valid) else 0.0,
        "r_min": float(valid["r"].min()),
        "r_max": float(valid["r"].max()),
        "lag_min": int(valid["lag"].min()),
        "lag_max": int(valid["lag"].max()),
        "method": request.method,
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

    return {
        "summary": summary,
        "series": {
            "index": [int(i) for i in range(len(ts1))],
            "ts1": [float(v) for v in ts1],
            "ts2": [float(v) for v in ts2],
        },
        "matrix_r": _matrix_payload(r_matrix),
        "matrix_p": _matrix_payload(p_matrix),
        "ranges_panel": ranges_panel,
        "strongest_links": strongest.to_dict(orient="records") if not strongest.empty else [],
        "notes": notes,
        "runtime_seconds": float(runtime_seconds),
    }
