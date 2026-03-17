# sdcpy-studio
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Web App](https://img.shields.io/badge/web-sdcpy--studio-0b7285.svg)](https://github.com/AlFontal/sdcpy-studio)
[![Engine](https://img.shields.io/badge/engine-sdcpy-111827.svg)](https://github.com/AlFontal/sdcpy)
[![Map Engine](https://img.shields.io/badge/map-sdcpy--map-1f7a8c.svg)](https://github.com/AlFontal/sdcpy-map)

`sdcpy-studio` is a web app for interactive scale-dependent correlation analysis of time series.

It offers two workflows:
- `2-Way SDC`: fast exploratory correlation analysis for paired time series.
- `SDC Map (beta)`: event-conditioned mapping over gridded fields using separate positive and negative driver-event classes.

## Visual Tour
### Home and main workflow
![sdcpy-studio home](docs/images/studio-home.png)

### 2-Way Explorer (ONI example after run)
![sdcpy-studio 2-way explorer ONI run](docs/images/studio-two-way-oni.png)

### SDC Map exploration
![sdcpy-studio map workflow](docs/images/studio-map-explore.png)

## Quick Start (Local)
### 1) Install dependencies
```bash
git clone https://github.com/AlFontal/sdcpy-studio.git
cd sdcpy-studio
uv sync --extra dev
```

Map support is bundled in the main install, including the NetCDF runtime dependencies used by custom field uploads.
If `../sdcpy-map` exists, `uv` is configured to use that sibling checkout in editable mode during local development.

### 2) Run the app
```bash
npm run dev:api
```

Open: `http://127.0.0.1:8000`

## Docker Deployment
The default deployment path is now image-first: pull the published container and run it with a persistent cache volume.

### 1) Pull the image
```bash
docker pull ghcr.io/alfontal/sdcpy-studio:latest
```

### 2) Download the deployment compose file
```bash
curl -O https://raw.githubusercontent.com/AlFontal/sdcpy-studio/main/docker-compose.yml
```

### 3) Start the app
```bash
docker compose up -d
```

Open: `http://127.0.0.1:8050`

### First-run cache warmup
On first boot the container starts immediately and warms the bundled SDC Map catalog cache in the background.

- `2-Way SDC` is available immediately.
- Custom uploaded SDC Map inputs (`CSV` driver + custom `NetCDF` field) are available immediately.
- Catalog-backed SDC Map datasets may be slower until the warmup completes.
- The cache lives in the named volume `sdcpy_map_cache`.

You can inspect the live warmup status at:
```bash
curl http://127.0.0.1:8050/health
```

The response keeps the normal liveness contract and adds `map_cache` metadata:
```json
{
  "status": "ok",
  "map_cache": {
    "status": "warming"
  }
}
```

### Warmup controls
Default Docker behavior is:
```bash
SDCPY_STUDIO_MAP_PREWARM_MODE=auto
```

Supported modes:
- `auto`: warm the bundled catalog cache on startup if the cache volume is missing or stale.
- `off`: disable startup warmup.
- `force`: rewarm the bundled catalog cache on every startup.

Example:
```bash
SDCPY_STUDIO_MAP_PREWARM_MODE=force docker compose up -d
```

### Optional manual prewarm
If you want a deterministic admin/init step before first use:
```bash
docker compose --profile tools run --rm cache-map
```

### Automatic image updates
If you want the server to keep tracking `ghcr.io/alfontal/sdcpy-studio:latest` automatically, enable the optional `watchtower` profile:
```bash
docker compose --profile ops up -d
```

That will:
- poll GHCR every 5 minutes
- pull a newer `latest` image when available
- recreate the `sdcpy-studio` container automatically
- remove old image layers after the update

If you want manual control instead, keep `watchtower` disabled and run:
```bash
docker compose pull
docker compose up -d
```

### Log UI
The same `ops` profile also starts `Dozzle`, a lightweight web UI for Docker logs.

Start it with:
```bash
docker compose --profile ops up -d
```

Then open:
```text
http://YOUR_SERVER:8051
```

That lets you:
- browse logs for `sdcpy-studio`, `watchtower`, and the rest of the stack
- stream logs live
- search and filter without shelling into the server

CLI fallback:
```bash
docker compose logs -f
docker compose logs -f sdcpy-studio
docker compose logs --since=30m sdcpy-studio
```

### Stop / reset
```bash
docker compose down
docker compose down -v
```

### Local source-built Docker workflow
For contributor workflows from a checkout:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

## What You Can Do
- Upload CSV data or paste two series directly.
- Run asynchronous analysis and monitor progress.
- Explore 2-way SDC outputs interactively.
- Use SDC Map exploration with a date slider, map bounds, and grid-cell comparison.
- Preview detected positive events (`N+`) and negative events (`N-`) before launching a map run.
- Run event-conditioned SDC maps with `Correlation width (r_w)`, `Base-state beta`, and separate positive/negative map outputs.
- Download outputs for reports and sharing.

## SDC Map Method
The studio’s SDC Map workflow now follows the methodology in [docs/sdc_explained.md](docs/sdc_explained.md):

- detect the strongest positive and negative driver events separately,
- define the base state from `beta * x0` rather than a single peak date,
- correlate centered driver event windows against lagged field windows instead of running one full-series map and filtering it afterward,
- compute and display separate positive-event and negative-event map products,
- keep `fragment size` terminology only for the 2-Way SDC workflow.

If you are co-developing `sdcpy-studio` with a sibling checkout of `sdcpy-map`, `uv` will prefer that local source automatically.

## Tech Stack
- FastAPI backend
- Vanilla JS + Plotly frontend
- `sdcpy` and `sdcpy-map` computational engines
- `uv` for dependency management

## Frontend Assets
Plotly is vendored into the image at [plotly-2.35.2.min.js](/Users/alejandro/projects/sdcpy-studio/sdcpy_studio/static/plotly-2.35.2.min.js) so deployments do not depend on external CDN access.

To upgrade Plotly:
```bash
curl -L https://cdn.plot.ly/plotly-NEW_VERSION.min.js -o sdcpy_studio/static/plotly-NEW_VERSION.min.js
```

Then:
- update the `<script>` tag in [index.html](/Users/alejandro/projects/sdcpy-studio/sdcpy_studio/templates/index.html)
- remove the old vendored file if it is no longer needed
- run the browser smoke tests again

## Development
```bash
uv sync --extra dev
ruff check .
uv run pytest -q
npm run test:e2e:with-api
```

For local Docker development from source:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml build
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## CI and Merge Protection
- CI runs on every PR to `main` and on pushes to `main` via `.github/workflows/ci.yml`.
- The workflow runs:
  - `uv run pytest -q tests/test_api.py`
  - `npm run test:e2e:with-api`

## License
MIT
