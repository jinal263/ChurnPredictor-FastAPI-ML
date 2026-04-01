# ── Stage 1: builder ────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/          ./app/
COPY templates/    ./templates/
COPY run.py        .

# Create model store directory
RUN mkdir -p models_store

# Environment defaults (override with -e DATABASE_URL=... etc.)
ENV DATABASE_URL=sqlite+aiosqlite:///./cc_underwriting.db \
    MODELS_STORE=./models_store \
    PORT=8000

EXPOSE 8000

# Healthcheck — hits /health every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "run.py"]
