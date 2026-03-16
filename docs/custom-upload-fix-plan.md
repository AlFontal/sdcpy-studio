# Custom Upload Fix Plan

## Goal

Make custom CSV and NetCDF uploads predictable, forgiving where safe, and explicit when the app cannot proceed.

This plan is based on the current upload logic in `sdcpy_studio/service.py` and the map upload UI in `sdcpy_studio/static/app.js`, plus the recent test pass against local CSV variants and NOAA NCEP/DOE Reanalysis 2 samples.

## Findings To Address

1. CSV parsing is strict about delimiter and silently falls back to whichever numeric column still parses.
2. NetCDF compatibility rejects NOAA files that are effectively surface fields but stored as `time, level, lat, lon` with a singleton `level`.
3. Some NetCDF files fail during open/decode with backend-specific errors that are not useful to end users.
4. The UI keeps stale error text after a later successful upload.
5. End-to-end runs can succeed while producing zero valid cells, which currently looks too much like success.
6. The runtime must install `cftime` and `netCDF4` as part of the main project dependencies, because some supported NetCDF files depend on them for decoding/opening.

## Recommended Order

1. Improve validation and error messaging first.
2. Fix NetCDF runtime capability and compatibility second.
3. Improve run-result UX for empty outputs third.
4. Add regression coverage around all of the above before merging.
5. Use typed Pydantic models for upload-inspection payloads as these payloads grow.

## P0: Make Upload Errors Actionable

### P0.0 Type the upload-inspection payloads with Pydantic

Problem:

- Upload-inspection responses are currently assembled as loose dicts.
- We are about to add delimiter metadata, fallback-series warnings, typed defaults, and richer NetCDF inspection metadata.
- That is exactly the point where backend/frontend drift becomes likely.

Change:

- Replace generic dict-shaped substructures in upload-inspection responses with explicit Pydantic models.
- Keep the numerical processing path unchanged; limit this to API payloads.

Recommended models:

- `DriverDefaults`
- `InspectWarning`
- `FieldDims`

Implementation touchpoints:

- `sdcpy_studio/schemas.py`
- `sdcpy_studio/service.py`
- `sdcpy_studio/main.py`

Acceptance criteria:

- Upload-inspection responses still serialize to the same JSON shape for existing fields.
- New inspection metadata is validated at construction time.
- Generic fields like `defaults: dict` and `dims: dict[str, int]` are replaced with typed models.

### P0.1 CSV delimiter detection

Problem:

- `pd.read_csv()` is currently used with default comma parsing in `inspect_dataset_csv`.
- Semicolon-delimited files fail with `Driver CSV must contain a parseable date column.`
- That error is technically true but not helpful.

Change:

- Detect delimiter before parsing, or attempt a small fallback sequence for common delimiters: `,`, `;`, `\t`, `|`.
- If fallback parsing succeeds, continue normally.
- If parsing still fails, return an explicit error such as:
  - `Could not parse CSV. Supported delimiters: comma, semicolon, tab, pipe.`

Implementation touchpoints:

- `sdcpy_studio/service.py`
  - `inspect_dataset_csv`
  - `load_custom_map_driver_series`

Acceptance criteria:

- The semicolon CSV used in recent testing is accepted.
- Existing comma-delimited CSV behavior remains unchanged.
- A truly malformed file produces a parse-focused error, not a misleading date-column error.

### P0.2 Explain numeric-column auto-selection

Problem:

- When one numeric candidate fails parsing but another succeeds, the app silently switches to the other series.
- This is reasonable backend behavior but poor UX because the user may analyze the wrong series unintentionally.

Change:

- Keep auto-selection, but expose it explicitly in the inspect payload and UI.
- Add metadata such as:
  - `rejected_numeric_columns`
  - `selection_warnings`
- Show a warning in the upload panel when the suggested series is not the obvious primary column.

Implementation touchpoints:

- `sdcpy_studio/service.py`
  - `inspect_dataset_csv`
  - `inspect_sdc_map_driver_csv`
- `sdcpy_studio/static/app.js`
  - `inspectMapDriverUploadFile`
  - upload status / metadata rendering

Acceptance criteria:

- A CSV like `driver_bad_numeric.csv` still uploads.
- The UI explains that `driver_index` was skipped and `alt_series` was selected instead.
- The user can still override the choice manually.

### P0.3 Clear stale errors on subsequent success

Problem:

- The map status line can continue to display a previous upload error after a later successful upload.

Change:

- On successful driver or field inspection, clear previous global error state tied to upload inspection.
- Keep success and error states scoped to each panel where possible.

Implementation touchpoints:

- `sdcpy_studio/static/app.js`
  - `inspectMapDriverUploadFile`
  - `inspectMapFieldUploadFile`
  - shared status rendering

Acceptance criteria:

- Reproduce:
  1. Upload invalid CSV.
  2. Upload valid CSV.
  3. Upload invalid NetCDF.
  4. Upload valid NetCDF.
- Result:
  - panel-specific status is correct
  - global status no longer shows the stale earlier error

## P1: Accept More Real-World NetCDF Files

### P1.0 Ship the right NetCDF runtime dependencies

Problem:

- The project previously relied on an optional `map` extra and did not consistently bundle `cftime` or `netCDF4` in the default install path.
- In practice, some NOAA files need:
  - `cftime` for non-standard or out-of-bounds time decoding
  - `netCDF4` for robust opening of NetCDF4/HDF5-backed files in environments where `h5netcdf` or `scipy` are insufficient
- That means the app can reject files that should be supported simply because the deployment is missing expected runtime pieces.

Change:

- Add the map stack and NetCDF runtime dependencies to the main dependency set.
- Treat `cftime` and `netCDF4` as part of the supported runtime for custom NetCDF uploads.
- Update setup docs so `uv sync --extra dev` provides full map and NetCDF support.

Implementation touchpoints:

- `pyproject.toml`
- `README.md`

Acceptance criteria:

- A fresh `uv sync --extra dev` environment includes `cftime` and `netCDF4`.
- The app no longer fails on supported files solely because those libraries are absent.
- The UI never tells the user to install a missing package that should already be part of the supported deployment profile.

### P1.1 Accept singleton non-spatial dimensions

Problem:

- `_normalize_custom_field_dataarray` currently assumes the variable can be renamed directly to `time/lat/lon` and then transposed to exactly those three dims.
- NOAA files like `air.2m.mon.mean.nc` and `wspd.10m.mon.mean.nc` are valid field products but store the data as `time, level, lat, lon` with `level=1`.

Change:

- Allow extra dimensions if their size is `1`.
- Squeeze singleton dimensions before the final `transpose("time", "lat", "lon")`.
- Include the squeezed dimensions in returned metadata so the UI can explain what was normalized.

Implementation touchpoints:

- `sdcpy_studio/service.py`
  - `_normalize_custom_field_dataarray`
  - `inspect_sdc_map_field_netcdf`

Acceptance criteria:

- `air.2m.mon.mean.nc` passes inspection.
- `wspd.10m.mon.mean.nc` passes inspection.
- A true 4D field with multiple pressure levels still fails unless a level-selection feature is added later.

### P1.2 Improve NetCDF open/decode behavior

Problem:

- `_open_netcdf_dataset` cycles engines, but some files still fail with backend-specific trace text that leaks directly to the UI.
- Some climatology/LTM files require `cftime` for correct decoding, and should succeed once the runtime includes it.
- We still need a graceful fallback path for files whose times remain undecodable even after `cftime` is available.

Change:

- Prefer normal decoding with `cftime`/`netCDF4` available.
- Add a fallback open path using `decode_times=False` when normal open still fails.
- Return a normalized user-facing error when the dataset is unreadable.
- Prefer messages like:
  - `This NetCDF file could be opened, but its time axis could not be decoded automatically.`
  - `This NetCDF file requires an unsupported backend in the current deployment.`

Implementation touchpoints:

- `sdcpy_studio/service.py`
  - `_open_netcdf_dataset`
  - `inspect_sdc_map_field_netcdf`

Acceptance criteria:

- Backend/internal exception text is not shown raw in the main UI.
- Files that should decode with `cftime` actually do so in the supported runtime.
- Files with undecodable time metadata still fail with a short, actionable explanation.
- If `decode_times=False` makes a file usable for compatibility inspection, the app should proceed and convert dates more carefully later.

### P1.3 Return richer incompatibility reasons

Problem:

- `No compatible variable found in NetCDF` is too coarse.
- Users need to know whether the failure is due to missing coords, wrong dims, multiple levels, bad time metadata, or backend support.

Change:

- During inspect, collect rejection reasons per variable.
- Return a payload like:
  - `variables`
  - `compatible_variables`
  - `incompatible_variables: [{name, reason}]`

Implementation touchpoints:

- `sdcpy_studio/service.py`
  - `inspect_sdc_map_field_netcdf`
- `sdcpy_studio/schemas.py`
  - `SDCMapFieldUploadInspectResponse`
- `sdcpy_studio/static/app.js`
  - `inspectMapFieldUploadFile`
  - metadata rendering

Acceptance criteria:

- A user uploading `air.2m.mon.mean.nc` sees a reason like:
  - `Variable air has an extra singleton dimension level; this file is now supported` after P1.1
  - or, before support, `Variable air uses extra dimension level.`

## P1: Improve Result UX When The Run Produces No Valid Cells

### P1.4 Distinguish compute success from analytical empty result

Problem:

- The job status is `succeeded`, but the actual scientific result can be empty: `0 valid cells`.
- The current UI reads as success without enough warning.

Change:

- Keep job status `succeeded`, but render an explicit result state such as:
  - `Run completed, but no grid cells passed the current filters.`
- Promote API notes into a visible warning panel in results mode.
- Include likely causes:
  - bounds too small
  - alpha too strict
  - top fraction too restrictive
  - fragment/lag combination too restrictive

Implementation touchpoints:

- `sdcpy_studio/static/app.js`
  - result rendering for SDC map jobs
- Possibly schema/result rendering helpers if needed

Acceptance criteria:

- The NOAA `prate` run still completes.
- Results view clearly marks `0 valid cells` as a warning state, not a clean success state.

### P1.5 Suggest next actions for empty-result runs

Problem:

- Users currently have to guess how to recover from an empty map.

Change:

- When `valid_cells == 0`, show 2 to 4 concrete suggestions:
  - widen bounds
  - increase top fraction
  - relax alpha
  - shorten fragment size or lag range

Implementation touchpoints:

- `sdcpy_studio/static/app.js`
  - result summary / notes rendering

Acceptance criteria:

- Empty-result runs display clear next steps without hiding downloads or metadata.

## P2: Polish The Upload Experience

### P2.1 Show parse summary before the user runs

Change:

- For CSV:
  - show detected delimiter
  - show detected date column
  - show suggested series and any skipped columns
- For NetCDF:
  - show chosen variable
  - show original dims
  - show normalized dims
  - show time and spatial coverage

Reason:

- This reduces surprise and makes uploads feel inspectable rather than opaque.

### P2.2 Add “Why this failed” help text inline

Change:

- Add short inline guidance under each upload error.
- Examples:
  - CSV: `Check delimiter, date format, and whether at least one series column is numeric.`
  - NetCDF: `Expected a gridded variable with time, latitude, and longitude coordinates.`

### P2.3 Add sample-file guidance

Change:

- Add one or two known-good examples in the UI copy or docs.
- Include NOAA monthly Gaussian-grid surface files as known-good references after P1.1.

## Test Plan

### API tests

Add cases in `tests/test_api.py` for:

- semicolon-delimited driver CSV
- CSV with rejected primary numeric column and accepted fallback column
- singleton-dimension NetCDF like `time, level, lat, lon`
- NetCDF decode fallback path with `decode_times=False`
- inspect payload including incompatibility reasons and warnings

### E2E tests

Extend `tests/e2e/basic-flow.spec.ts` with:

- invalid CSV then valid CSV clears stale error
- invalid NetCDF then valid NetCDF clears stale error
- singleton-level NOAA-style NetCDF upload succeeds
- successful run with `valid_cells == 0` renders warning state and recovery suggestions

## Recommended Delivery Slices

### Slice 1

- P0.0 typed Pydantic upload-inspection payloads
- P0.1 delimiter detection
- P0.2 auto-selection warning
- P0.3 stale-error cleanup
- tests

### Slice 2

- P1.0 NetCDF runtime dependencies (`cftime`, `netCDF4`)
- P1.1 singleton-dimension NetCDF support
- P1.2 NetCDF open/decode fallback
- P1.3 richer incompatibility reasons
- tests

### Slice 3

- P1.4 empty-result warning state
- P1.5 recovery suggestions
- P2 polish
- tests

## Review Notes

The highest-value UX fix is not cosmetic. It is changing the app from “strict and opaque” to “strict where correctness matters, but explicit about what it inferred and why it rejected a file.”

The most important functional fixes are:

- shipping `cftime` and `netCDF4` as part of the supported map runtime
- normalizing singleton extra dimensions in NetCDF variables

Without the first, some valid files fail because the environment is incomplete.
Without the second, common NOAA monthly products still fail even in a complete environment.

The most important trust fix is exposing when the app auto-selected a different series than the user likely intended.
