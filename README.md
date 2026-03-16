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

## Docker Compose Deployment
`docker-compose.yml` is the easiest way to deploy and keep a persistent dataset cache.

### Start / update
```bash
docker compose up -d --build
```

Open: `http://127.0.0.1:8050`

### Prewarm map datasets (recommended)
```bash
docker compose --profile tools run --rm cache-map
```

### Stop
```bash
docker compose down
```

### Stop + remove cache volume
```bash
docker compose down -v
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

## Development
```bash
uv sync --extra dev
ruff check .
uv run pytest -q
npm run test:e2e:with-api
```

## CI and Merge Protection
- CI runs on every PR to `main` and on pushes to `main` via `.github/workflows/ci.yml`.
- The workflow runs:
  - `uv run pytest -q tests/test_api.py`
  - `npm run test:e2e:with-api`

## License
MIT
