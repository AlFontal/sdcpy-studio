"""Service-layer helpers for heavy studio computations."""

from __future__ import annotations

import io
import math
import os
from time import perf_counter

import numpy as np
import pandas as pd
from sdcpy import compute_sdc

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

    all_matrix = valid.pivot(index="start_2", columns="start_1", values="r")
    sig_matrix = valid.assign(
        r_sig=np.where(valid["p_value"] <= request.alpha, valid["r"], np.nan)
    ).pivot(index="start_2", columns="start_1", values="r_sig")

    all_matrix = _downsample_matrix(all_matrix)
    sig_matrix = _downsample_matrix(sig_matrix)

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
        "heatmap_all": _matrix_payload(all_matrix),
        "heatmap_significant": _matrix_payload(sig_matrix),
        "strongest_links": strongest.to_dict(orient="records") if not strongest.empty else [],
        "notes": notes,
        "runtime_seconds": float(runtime_seconds),
    }
