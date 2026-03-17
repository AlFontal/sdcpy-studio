FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    SDCPY_STUDIO_MAP_PREWARM_MODE=auto \
    SDCPY_STUDIO_SDCPY_MAP_DATA_DIR=/root/.cache/sdcpy-studio/sdcpy-map

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md /app/
COPY sdcpy_studio /app/sdcpy_studio
COPY scripts /app/scripts

RUN uv sync --no-dev --no-sources --no-editable

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "sdcpy_studio.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
