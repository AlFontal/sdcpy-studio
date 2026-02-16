#!/usr/bin/env python3
"""Pre-download sdcpy-map datasets used by the Studio map workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_parser(driver_keys: list[str], field_keys: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prewarm local cache for sdcpy-map datasets used by sdcpy-studio."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Cache directory. Defaults to SDCPY_STUDIO_SDCPY_MAP_DATA_DIR or ~/.cache/sdcpy-studio/sdcpy-map",
    )
    parser.add_argument(
        "--driver-key",
        action="append",
        choices=driver_keys,
        help="Driver dataset key(s) to prewarm. Defaults to all.",
    )
    parser.add_argument(
        "--field-key",
        action="append",
        choices=field_keys,
        help="Field dataset key(s) to prewarm. Defaults to all.",
    )
    parser.add_argument(
        "--skip-coastline",
        action="store_true",
        help="Skip coastline asset download.",
    )
    return parser


def main() -> int:
    try:
        from sdcpy_map.datasets import DRIVER_DATASETS, FIELD_DATASETS

        from sdcpy_studio.service import fetch_sdc_map_assets
    except ImportError:
        print(
            "sdcpy-map dependencies are unavailable. Install them with: uv sync --extra map",
            file=sys.stderr,
        )
        return 1

    driver_keys = sorted(DRIVER_DATASETS.keys())
    field_keys = sorted(FIELD_DATASETS.keys())
    parser = _build_parser(driver_keys, field_keys)
    args = parser.parse_args()

    data_dir = args.data_dir or Path(
        os.getenv(
            "SDCPY_STUDIO_SDCPY_MAP_DATA_DIR",
            str(Path.home() / ".cache" / "sdcpy-studio" / "sdcpy-map"),
        )
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_drivers = args.driver_key or driver_keys
    selected_fields = args.field_key or field_keys

    print(f"Prewarming SDC map cache at: {data_dir}")
    total_bytes = 0
    downloaded_labels: set[str] = set()
    failures: list[str] = []

    for driver_key in selected_drivers:
        for field_key in selected_fields:
            try:
                paths = fetch_sdc_map_assets(
                    data_dir=data_dir,
                    driver_key=driver_key,
                    field_key=field_key,
                    include_coastline=not args.skip_coastline,
                )
            except Exception as exc:
                failures.append(f"{driver_key} × {field_key}: {exc}")
                continue

            for label, path in (
                (f"driver:{driver_key}", paths["driver"]),
                (f"field:{field_key}", paths["field"]),
            ):
                if label in downloaded_labels:
                    continue
                downloaded_labels.add(label)
                size = path.stat().st_size if path.exists() else 0
                total_bytes += size
                print(f"  - {label:<16} {path.name:<36} {size / (1024 * 1024):7.2f} MB")

            if not args.skip_coastline and "coastline" in paths and "coastline" not in downloaded_labels:
                downloaded_labels.add("coastline")
                path = paths["coastline"]
                size = path.stat().st_size if path.exists() else 0
                total_bytes += size
                print(f"  - {'coastline':<16} {path.name:<36} {size / (1024 * 1024):7.2f} MB")

    print(f"Cached {len(downloaded_labels)} files. Total size on disk: {total_bytes / (1024 * 1024):.2f} MB")
    if failures:
        print("\nSome downloads failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
