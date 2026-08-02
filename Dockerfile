FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# System deps needed by scikit-learn wheels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code + the trained model artifact + params.yaml (needed for
# the same text-cleaning config used at training time).
COPY app/ app/
COPY src/ src/
COPY params.yaml .
# The serving artifact is committed directly to git at models/production/
# (small file, no Git LFS needed) so a fresh `git clone` + Docker build is
# self-contained — no DVC remote credentials required at build time. DVC
# still owns models/model.pkl for experiment tracking/reproducibility; see
# scripts/promote_model.py for how a model moves from "trained" to "serving".
COPY models/production/model.pkl models/model.pkl

# Predictions get logged here for drift monitoring; mount a volume in
# production if you want the log to survive redeploys.
RUN mkdir -p logs

ENV MODEL_PATH=models/model.pkl \
    PARAMS_PATH=params.yaml \
    PREDICTION_LOG_PATH=logs/predictions.jsonl \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0) if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health').status==200 else sys.exit(1)"

# Render/Railway/Cloud Run all inject $PORT; fall back to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
