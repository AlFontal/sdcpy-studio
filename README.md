# sdcpy-studio
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Engine](https://img.shields.io/badge/engine-sdcpy-111827.svg)](https://github.com/AlFontal/sdcpy)
[![Framework](https://img.shields.io/badge/framework-FastAPI-059669.svg)](https://fastapi.tiangolo.com/)
[![Deploy](https://img.shields.io/badge/deploy-Render-46E3B7.svg)](https://render.com/)

<img src="https://raw.githubusercontent.com/AlFontal/sdcpy-app/master/static/sdcpy_logo_black.png" width="180" alt="sdcpy logo" />

Interactive web studio for Scale-Dependent Correlation (SDC) analysis powered by [`sdcpy`](https://github.com/AlFontal/sdcpy).

+ Free software: MIT license
+ Repository: https://github.com/AlFontal/sdcpy-studio

## Features

- Dataset-first workflow (CSV upload, date/numeric inference, TS1/TS2 mapping)
- Optional paste-values workflow with automatic validation
- Asynchronous SDC jobs with status polling
- Interactive 2-way SDC explorer with hover-linked fragment highlighting
- Significant-mask visualization + on-hover non-significant (`NS`) feedback
- Downloadable artifacts (`.xlsx`, `.png`, `.svg`) with contextual filenames
- Built-in ONI demo dataset bootstrap

## Installation

Clone and install with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/AlFontal/sdcpy-studio.git
cd sdcpy-studio
uv sync --extra dev
```

To enable the SDC Map workflow locally:

```bash
uv sync --extra dev --extra map
```

Prewarm local map datasets cache (recommended before testing map workflows):

```bash
npm run cache:map
```

If one large field file fails mid-download, retry only that dataset:

```bash
npm run cache:map -- --field-key ersstv5_sst
```

This downloads all offered map datasets into:
- `SDCPY_STUDIO_SDCPY_MAP_DATA_DIR` (if set), or
- `~/.cache/sdcpy-studio/sdcpy-map`

## Run Locally

```bash
npm run dev:api
```

Open `http://127.0.0.1:8000`.

## Docker

Default image build (works without map private dependency access):

```bash
docker build -t sdcpy-studio .
```

Run on a Linux server:

```bash
docker run -d \
  --name sdcpy-studio \
  --restart unless-stopped \
  -p 8000:8000 \
  -e SDCPY_STUDIO_MAX_WORKERS=2 \
  sdcpy-studio
```

Image includes a healthcheck on `/health`.

To enable SDC Map dependencies at build time:

```bash
docker build -t sdcpy-studio-map \
  --build-arg INSTALL_MAP_DEPS=1 \
  --build-arg GITHUB_TOKEN=<your_github_token> \
  .
```

Notes:
- `sdcpy-map` is installed from GitHub in the `map` extra; private-repo access requires `GITHUB_TOKEN` while the repo is private.
- Default build sets `INSTALL_MAP_DEPS=0` so deployment does not fail in environments without GitHub credentials.
- With `INSTALL_MAP_DEPS=0`, the app still runs but the SDC Map tab returns a dependency error until map deps are installed.
- You can prewarm the runtime dataset cache with `npm run cache:map` (or run it inside the container).

## API Endpoints

- `GET /health`
- `GET /api/v1/health`
- `GET /api/v1/examples/synthetic`
- `GET /api/v1/examples/oni-dataset`
- `GET /api/v1/examples/oni-dataset.csv`
- `POST /api/v1/datasets/inspect`
- `POST /api/v1/jobs/sdc`
- `POST /api/v1/jobs/sdc/csv`
- `POST /api/v1/jobs/sdc/dataset`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/result`
- `GET /api/v1/jobs/{job_id}/download/{fmt}` (`fmt` in `xlsx|png|svg`)

## Render Deployment

Render deployment is configured via `render.yaml` in this repository.

After creating a new **Web Service** on Render and connecting this repo:

1. Select `render.yaml` blueprint deploy (recommended), or create a Python Web Service manually.
2. Use start command:
   `uvicorn sdcpy_studio.main:create_app --factory --host 0.0.0.0 --port $PORT`
3. Set health check path to `/health`.

## Development

```bash
uv sync --extra dev
ruff check .
uv run pytest -q
npm run test:e2e:with-api
```
