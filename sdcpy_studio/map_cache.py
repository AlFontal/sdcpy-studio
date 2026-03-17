"""Startup cache warmup helpers for the SDC Map workflow."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import sleep
from typing import Any

CACHE_SCHEMA_VERSION = 1
LOCK_STALE_SECONDS = 60 * 60


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version() -> str:
    try:
        return version("sdcpy-studio")
    except PackageNotFoundError:
        return "0.1.0"


def _read_prewarm_mode() -> str:
    mode = os.getenv("SDCPY_STUDIO_MAP_PREWARM_MODE", "off").strip().lower() or "off"
    if mode not in {"auto", "off", "force"}:
        return "off"
    return mode


def _load_catalog_specs() -> tuple[list[str], list[str]]:
    from sdcpy_map.datasets import DRIVER_DATASETS, FIELD_DATASETS

    return sorted(DRIVER_DATASETS.keys()), sorted(FIELD_DATASETS.keys())


def _manifest_path(data_dir: Path) -> Path:
    return data_dir / "cache-manifest.json"


def _lock_path(data_dir: Path) -> Path:
    return data_dir / ".cache-warmup.lock"


def _read_manifest(data_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(data_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_manifest(data_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = _manifest_path(data_dir)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    return payload


def _manifest_is_valid(
    data_dir: Path,
    manifest: dict[str, Any] | None,
    *,
    driver_keys: list[str],
    field_keys: list[str],
) -> bool:
    if not manifest:
        return False
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        return False
    if str(manifest.get("studio_version", "")).strip() != _package_version():
        return False
    if sorted(str(item) for item in manifest.get("driver_keys", [])) != driver_keys:
        return False
    if sorted(str(item) for item in manifest.get("field_keys", [])) != field_keys:
        return False

    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return False

    coastline_name = str(files.get("coastline", "")).strip()
    if not coastline_name:
        return False
    if not (data_dir / coastline_name).exists():
        return False

    driver_files = files.get("drivers", {})
    field_files = files.get("fields", {})
    if not isinstance(driver_files, dict) or not isinstance(field_files, dict):
        return False
    for key in driver_keys:
        filename = str(driver_files.get(key, "")).strip()
        if not filename or not (data_dir / filename).exists():
            return False
    for key in field_keys:
        filename = str(field_files.get(key, "")).strip()
        if not filename or not (data_dir / filename).exists():
            return False
    return True


def _build_manifest(
    *,
    data_dir: Path,
    driver_paths: dict[str, Path],
    field_paths: dict[str, Path],
    coastline_path: Path,
) -> dict[str, Any]:
    driver_keys = sorted(driver_paths.keys())
    field_keys = sorted(field_paths.keys())
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "studio_version": _package_version(),
        "driver_keys": driver_keys,
        "field_keys": field_keys,
        "coastline_ready": coastline_path.exists(),
        "completed_at": _utc_now_iso(),
        "files": {
            "coastline": coastline_path.name,
            "drivers": {key: path.name for key, path in sorted(driver_paths.items())},
            "fields": {key: path.name for key, path in sorted(field_paths.items())},
        },
    }


def prewarm_map_cache(data_dir: Path) -> dict[str, Any]:
    """Download all bundled SDC Map catalog assets into the target cache directory."""
    from sdcpy_studio.service import (
        fetch_sdc_map_coastline_asset,
        fetch_sdc_map_driver_asset,
        fetch_sdc_map_field_asset,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    driver_keys, field_keys = _load_catalog_specs()
    coastline_path = fetch_sdc_map_coastline_asset(data_dir)
    driver_paths = {key: fetch_sdc_map_driver_asset(data_dir, key) for key in driver_keys}
    field_paths = {key: fetch_sdc_map_field_asset(data_dir, key) for key in field_keys}
    manifest = _build_manifest(
        data_dir=data_dir,
        driver_paths=driver_paths,
        field_paths=field_paths,
        coastline_path=coastline_path,
    )
    return _write_manifest(data_dir, manifest)


@dataclass
class _WarmupLock:
    path: Path
    acquired: bool = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = max(0.0, datetime.now(timezone.utc).timestamp() - self.path.stat().st_mtime)
            if age > LOCK_STALE_SECONDS:
                try:
                    self.path.unlink()
                except OSError:
                    pass
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": _utc_now_iso()}))
        self.acquired = True
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self.acquired = False


class MapCacheWarmupManager:
    """Track and optionally warm the SDC Map cache in the background."""

    def __init__(
        self,
        *,
        data_dir: Path,
        mode: str | None = None,
        prewarm_fn=prewarm_map_cache,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.mode = (mode or _read_prewarm_mode()).strip().lower()
        if self.mode not in {"auto", "off", "force"}:
            self.mode = "off"
        self._prewarm_fn = prewarm_fn
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = "ready"
        self._error: str | None = None
        self._detail: str | None = None
        self._manifest: dict[str, Any] | None = None

    def start(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_from_manifest()
        if self.mode == "off":
            with self._state_lock:
                self._detail = "Automatic map-cache warmup is disabled."
            return
        needs_warmup = self.mode == "force" or not self._manifest_ready()
        if not needs_warmup:
            return
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return
            self._status = "warming"
            self._error = None
            self._detail = "Warming bundled SDC Map catalog assets in the background."
            self._thread = threading.Thread(
                target=self._run_warmup,
                name="sdcpy-studio-map-cache-warmup",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        thread = None
        with self._state_lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=0.2)

    def snapshot(self) -> dict[str, Any]:
        self._refresh_from_manifest()
        with self._state_lock:
            manifest = dict(self._manifest or {})
            return {
                "status": self._status,
                "mode": self.mode,
                "cache_dir": str(self.data_dir),
                "manifest_present": _manifest_path(self.data_dir).exists(),
                "catalog_ready": self._manifest_ready(),
                "coastline_ready": bool(manifest.get("coastline_ready")),
                "completed_at": manifest.get("completed_at"),
                "detail": self._detail,
                "error": self._error,
            }

    def _manifest_ready(self) -> bool:
        driver_keys, field_keys = _load_catalog_specs()
        return _manifest_is_valid(
            self.data_dir,
            self._manifest,
            driver_keys=driver_keys,
            field_keys=field_keys,
        )

    def _refresh_from_manifest(self) -> None:
        manifest = _read_manifest(self.data_dir)
        driver_keys, field_keys = _load_catalog_specs()
        is_ready = _manifest_is_valid(
            self.data_dir,
            manifest,
            driver_keys=driver_keys,
            field_keys=field_keys,
        )
        with self._state_lock:
            self._manifest = manifest
            if is_ready:
                self._status = "ready"
                self._error = None
                self._detail = "Bundled SDC Map catalog assets are ready."
            elif self.mode == "off" and self._status != "error":
                self._status = "ready"
                self._detail = "Automatic map-cache warmup is disabled."
            elif self._status != "warming" and self._error is None:
                self._status = "ready"
                self._detail = "Bundled SDC Map catalog assets have not been warmed yet."

    def _run_warmup(self) -> None:
        lock = _WarmupLock(_lock_path(self.data_dir))
        if not lock.acquire():
            with self._state_lock:
                self._status = "warming"
                self._detail = "Another process is warming the map cache."
            for _ in range(180):
                sleep(1)
                self._refresh_from_manifest()
                with self._state_lock:
                    if self._status == "ready":
                        return
                if not lock.path.exists():
                    break
            with self._state_lock:
                if self._status != "ready":
                    self._status = "error"
                    self._error = "Timed out waiting for another process to finish warming the map cache."
                    self._detail = "The cache lock did not clear and no valid cache manifest appeared."
            return
        try:
            manifest = self._prewarm_fn(self.data_dir)
        except Exception as exc:
            with self._state_lock:
                self._status = "error"
                self._error = str(exc)
                self._detail = "Bundled SDC Map catalog assets could not be warmed."
            return
        finally:
            lock.release()
        with self._state_lock:
            self._manifest = manifest
            self._status = "ready"
            self._error = None
            self._detail = "Bundled SDC Map catalog assets are ready."
