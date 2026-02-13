# sdcpy-studio

Modern web studio for non-technical Scale-Dependent Correlation (SDC) workflows.

`sdcpy-studio` runs on top of [`sdcpy`](https://github.com/AlFontal/sdcpy), adds asynchronous execution for expensive jobs, and provides an interactive UI that includes:

- Job-based execution (`queued`/`running`/`succeeded`/`failed`)
- In-job progress updates (progress bar + status text)
- CSV upload and paste-in input modes
- Dataset workflow (single table upload, date/numeric inference, TS1/TS2 selection)
- Interactive **2-way explorer** with side time series and hover-linked fragment highlighting
- Significant-only matrix rendering with user-adjustable explorer alpha
- Downloadable PNG/SVG generated directly by `sdcpy` `combi_plot(...)`
- Result downloads (`.xlsx`, `.png`, `.svg`)
- Docker deployment

## Quick start

```bash
uv sync --extra dev
sdcpy-studio
```

Open `http://localhost:8000`.

## Docker

```bash
docker compose -f docker-compose.yml up --build
```

## API

- `GET /api/v1/health`
- `GET /api/v1/examples/synthetic`
- `POST /api/v1/jobs/sdc`
- `POST /api/v1/jobs/sdc/csv`
- `POST /api/v1/datasets/inspect`
- `POST /api/v1/jobs/sdc/dataset`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/result`
- `GET /api/v1/jobs/{job_id}/download/{fmt}` (`fmt` in `xlsx|png|svg`)

## Why jobs?

SDC can be expensive when fragment size is small, lag range is wide, and permutations are high. The app immediately returns a `job_id`, computes in background workers, and the frontend polls status to keep the UI responsive.

## Development

```bash
uv sync --extra dev
ruff check .
uv run pytest -q
```
