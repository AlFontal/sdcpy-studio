from __future__ import annotations

import threading
import time
from pathlib import Path

from sdcpy_studio import map_cache
from sdcpy_studio.map_cache import MapCacheWarmupManager
from sdcpy_studio.schemas import SDCMapJobRequest
from sdcpy_studio.service import _resolve_map_paths_for_request


def _write_fake_manifest(data_dir: Path, driver_keys: list[str], field_keys: list[str]) -> dict:
    coastline = data_dir / "coastline.geojson"
    coastline.write_text("{}", encoding="utf-8")
    driver_paths = {}
    field_paths = {}
    for key in driver_keys:
        path = data_dir / f"{key}.nc"
        path.write_text(key, encoding="utf-8")
        driver_paths[key] = path
    for key in field_keys:
        path = data_dir / f"{key}.nc"
        path.write_text(key, encoding="utf-8")
        field_paths[key] = path
    manifest = map_cache._build_manifest(
        data_dir=data_dir,
        driver_paths=driver_paths,
        field_paths=field_paths,
        coastline_path=coastline,
    )
    manifest["studio_version"] = map_cache._package_version()
    return map_cache._write_manifest(data_dir, manifest)


def test_map_cache_warmup_writes_manifest_and_transitions_ready(tmp_path, monkeypatch):
    driver_keys = ["pdo"]
    field_keys = ["ncep_air"]
    monkeypatch.setattr(map_cache, "_load_catalog_specs", lambda: (driver_keys, field_keys))

    calls = {"count": 0}

    def fake_prewarm(data_dir: Path) -> dict:
        calls["count"] += 1
        return _write_fake_manifest(data_dir, driver_keys, field_keys)

    manager = MapCacheWarmupManager(data_dir=tmp_path, mode="auto", prewarm_fn=fake_prewarm)
    manager.start()
    for _ in range(20):
        snapshot = manager.snapshot()
        if snapshot["status"] == "ready":
            break
        time.sleep(0.1)
    manager.stop()

    assert calls["count"] == 1
    assert snapshot["status"] == "ready"
    assert snapshot["catalog_ready"] is True
    assert snapshot["coastline_ready"] is True
    assert (tmp_path / "cache-manifest.json").exists()


def test_map_cache_existing_manifest_skips_warmup(tmp_path, monkeypatch):
    driver_keys = ["pdo"]
    field_keys = ["ncep_air"]
    monkeypatch.setattr(map_cache, "_load_catalog_specs", lambda: (driver_keys, field_keys))
    _write_fake_manifest(tmp_path, driver_keys, field_keys)

    calls = {"count": 0}

    def fake_prewarm(_data_dir: Path) -> dict:
        calls["count"] += 1
        raise AssertionError("warmup should have been skipped")

    manager = MapCacheWarmupManager(data_dir=tmp_path, mode="auto", prewarm_fn=fake_prewarm)
    manager.start()
    snapshot = manager.snapshot()
    manager.stop()

    assert calls["count"] == 0
    assert snapshot["status"] == "ready"
    assert snapshot["catalog_ready"] is True


def test_map_cache_corrupt_manifest_triggers_rewarm(tmp_path, monkeypatch):
    driver_keys = ["pdo"]
    field_keys = ["ncep_air"]
    monkeypatch.setattr(map_cache, "_load_catalog_specs", lambda: (driver_keys, field_keys))
    (tmp_path / "cache-manifest.json").write_text("{not-json", encoding="utf-8")

    calls = {"count": 0}

    def fake_prewarm(data_dir: Path) -> dict:
        calls["count"] += 1
        return _write_fake_manifest(data_dir, driver_keys, field_keys)

    manager = MapCacheWarmupManager(data_dir=tmp_path, mode="auto", prewarm_fn=fake_prewarm)
    manager.start()
    for _ in range(20):
        snapshot = manager.snapshot()
        if snapshot["status"] == "ready":
            break
        time.sleep(0.1)
    manager.stop()

    assert calls["count"] == 1
    assert snapshot["catalog_ready"] is True


def test_map_cache_off_mode_skips_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(map_cache, "_load_catalog_specs", lambda: (["pdo"], ["ncep_air"]))

    calls = {"count": 0}

    def fake_prewarm(_data_dir: Path) -> dict:
        calls["count"] += 1
        return {}

    manager = MapCacheWarmupManager(data_dir=tmp_path, mode="off", prewarm_fn=fake_prewarm)
    manager.start()
    snapshot = manager.snapshot()
    manager.stop()

    assert calls["count"] == 0
    assert snapshot["status"] == "ready"
    assert snapshot["catalog_ready"] is False


def test_map_cache_force_mode_reruns_even_with_manifest(tmp_path, monkeypatch):
    driver_keys = ["pdo"]
    field_keys = ["ncep_air"]
    monkeypatch.setattr(map_cache, "_load_catalog_specs", lambda: (driver_keys, field_keys))
    _write_fake_manifest(tmp_path, driver_keys, field_keys)

    calls = {"count": 0}

    def fake_prewarm(data_dir: Path) -> dict:
        calls["count"] += 1
        return _write_fake_manifest(data_dir, driver_keys, field_keys)

    manager = MapCacheWarmupManager(data_dir=tmp_path, mode="force", prewarm_fn=fake_prewarm)
    manager.start()
    for _ in range(20):
        snapshot = manager.snapshot()
        if snapshot["status"] == "ready":
            break
        time.sleep(0.1)
    manager.stop()

    assert calls["count"] == 1
    assert snapshot["catalog_ready"] is True


def test_map_cache_concurrent_start_avoids_duplicate_warmups(tmp_path, monkeypatch):
    driver_keys = ["pdo"]
    field_keys = ["ncep_air"]
    monkeypatch.setattr(map_cache, "_load_catalog_specs", lambda: (driver_keys, field_keys))

    calls = {"count": 0}
    entered = threading.Event()
    release = threading.Event()

    def fake_prewarm(data_dir: Path) -> dict:
        calls["count"] += 1
        entered.set()
        release.wait(timeout=2)
        return _write_fake_manifest(data_dir, driver_keys, field_keys)

    manager_a = MapCacheWarmupManager(data_dir=tmp_path, mode="auto", prewarm_fn=fake_prewarm)
    manager_b = MapCacheWarmupManager(data_dir=tmp_path, mode="auto", prewarm_fn=fake_prewarm)
    manager_a.start()
    assert entered.wait(timeout=1)
    manager_b.start()
    release.set()

    for _ in range(30):
        if manager_a.snapshot()["status"] == "ready" and manager_b.snapshot()["status"] == "ready":
            break
        time.sleep(0.1)
    manager_a.stop()
    manager_b.stop()

    assert calls["count"] == 1
    assert manager_a.snapshot()["catalog_ready"] is True
    assert manager_b.snapshot()["catalog_ready"] is True


def test_custom_upload_map_paths_only_require_coastline(tmp_path, monkeypatch):
    calls = {"driver": 0, "field": 0, "coastline": 0}

    monkeypatch.setattr(
        "sdcpy_studio.service.fetch_sdc_map_driver_asset",
        lambda data_dir, driver_key: calls.__setitem__("driver", calls["driver"] + 1) or Path(data_dir) / f"{driver_key}.nc",
    )
    monkeypatch.setattr(
        "sdcpy_studio.service.fetch_sdc_map_field_asset",
        lambda data_dir, field_key: calls.__setitem__("field", calls["field"] + 1) or Path(data_dir) / f"{field_key}.nc",
    )
    monkeypatch.setattr(
        "sdcpy_studio.service.fetch_sdc_map_coastline_asset",
        lambda data_dir: calls.__setitem__("coastline", calls["coastline"] + 1) or Path(data_dir) / "coastline.geojson",
    )

    request = SDCMapJobRequest(
        driver_source_type="upload",
        field_source_type="upload",
        driver_upload_id="driver-1",
        driver_date_column="date",
        driver_value_column="idx",
        field_upload_id="field-1",
        field_variable="air",
    )
    paths = _resolve_map_paths_for_request(request, data_dir=tmp_path)

    assert calls == {"driver": 0, "field": 0, "coastline": 1}
    assert "coastline" in paths
    assert "driver" not in paths
    assert "field" not in paths
