#!/usr/bin/env python3
"""Pre-download sdcpy-map datasets used by the Studio map workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prewarm local cache for sdcpy-map datasets used by sdcpy-studio."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Cache directory. Defaults to SDCPY_STUDIO_SDCPY_MAP_DATA_DIR or ~/.cache/sdcpy-studio/sdcpy-map",
    )
    return parser


def main() -> int:
    try:
        from sdcpy_studio.map_cache import prewarm_map_cache
    except ImportError:
        print(
            "sdcpy-map dependencies are unavailable. Install them with: uv sync",
            file=sys.stderr,
        )
        return 1

    parser = _build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir or Path(
        os.getenv(
            "SDCPY_STUDIO_SDCPY_MAP_DATA_DIR",
            str(Path.home() / ".cache" / "sdcpy-studio" / "sdcpy-map"),
        )
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prewarming SDC map cache at: {data_dir}")
    manifest = prewarm_map_cache(data_dir)
    files = manifest.get("files", {})
    total_paths = [data_dir / str(files.get("coastline", ""))]
    total_paths.extend(data_dir / str(name) for name in files.get("drivers", {}).values())
    total_paths.extend(data_dir / str(name) for name in files.get("fields", {}).values())
    total_bytes = sum(path.stat().st_size for path in total_paths if path.exists())
    print(
        f"Cached {len(total_paths)} files. Total size on disk: {total_bytes / (1024 * 1024):.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
