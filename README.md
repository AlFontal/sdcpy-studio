# sdcpy-studio

Modern web studio for non-technical Scale-Dependent Correlation (SDC) workflows.

`sdcpy-studio` runs on top of [`sdcpy`](https://github.com/AlFontal/sdcpy), adds asynchronous execution for expensive jobs, and provides an interactive UI that includes:

- Job-based execution (`queued`/`running`/`succeeded`/`failed`)
- CSV upload and paste-in input modes
- Interactive SDC heatmaps
- Interactive **2-way explorer** with side time series and hover-linked fragment highlighting
- Docker deployment

## Quick start

```bash
pip install -e .[dev]
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
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/result`

## Why jobs?

SDC can be expensive when fragment size is small, lag range is wide, and permutations are high. The app immediately returns a `job_id`, computes in worker processes, and the frontend polls status to keep the UI responsive.

## Development

```bash
pip install -e .[dev]
ruff check .
pytest -q
```
