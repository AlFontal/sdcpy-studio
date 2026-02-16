FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG INSTALL_MAP_DEPS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY sdcpy_studio /app/sdcpy_studio

RUN pip install --no-cache-dir --upgrade pip \
    && if [ "$INSTALL_MAP_DEPS" = "1" ]; then \
        pip install --no-cache-dir ".[map]"; \
      else \
        pip install --no-cache-dir .; \
      fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "sdcpy_studio.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
