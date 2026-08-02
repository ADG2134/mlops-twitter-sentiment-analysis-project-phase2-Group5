"""FastAPI inference service for the Twitter sentiment model.

Loads the sklearn Pipeline produced by `src/train.py` (models/model.pkl) and
serves /predict, /predict/batch, /health, /metrics.

Every prediction is appended to `logs/predictions.jsonl` (text, cleaned text,
prediction, confidence, timestamp). That log is the "current data" window
that scripts/monitor_drift.py compares against the training reference sample
to detect data/prediction drift.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.schemas import (  # noqa: E402
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.prepare import clean_text  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", "params.yaml"))
LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "unversioned")

state: dict = {"model": None, "prepare_cfg": {}, "model_version": MODEL_VERSION}


def load_model() -> None:
    if not MODEL_PATH.exists():
        state["model"] = None
        return
    with open(MODEL_PATH, "rb") as f:
        state["model"] = pickle.load(f)
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, encoding="utf-8") as f:
            state["prepare_cfg"] = yaml.safe_load(f).get("prepare", {})
    if state["model_version"] == "unversioned" and MODEL_PATH.exists():
        state["model_version"] = f"local-{int(MODEL_PATH.stat().st_mtime)}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Serves the TF-IDF + Logistic Regression sentiment model "
    "trained in Phase 1 of the MLOps pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _log_prediction(raw_text: str, cleaned: str, label: str, positive_prob: float) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": raw_text,
        "cleaned_text": cleaned,
        "text_length": len(raw_text),
        "cleaned_length": len(cleaned),
        "label": label,
        "positive_probability": positive_prob,
        "model_version": state["model_version"],
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        # Logging must never break inference (e.g. read-only filesystem).
        pass


def _predict_one(text: str) -> PredictResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleaned = clean_text(text, state["prepare_cfg"])
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is empty after cleaning")

    model = state["model"]
    pred = model.predict([cleaned])[0]

    if hasattr(model, "predict_proba"):
        positive_prob = float(model.predict_proba([cleaned])[0][1])
    elif hasattr(model, "decision_function"):
        # e.g. LinearSVC has no predict_proba; squash the margin instead.
        margin = float(model.decision_function([cleaned])[0])
        positive_prob = 1 / (1 + pow(2.718281828, -margin))
    else:
        positive_prob = float(pred)

    label = "positive" if pred == 1 else "negative"
    score = positive_prob if pred == 1 else 1 - positive_prob

    _log_prediction(text, cleaned, label, positive_prob)

    return PredictResponse(
        text=text,
        label=label,
        score=round(score, 4),
        positive_probability=round(positive_prob, 4),
        model_version=state["model_version"],
    )


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "twitter-sentiment-api",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok" if state["model"] is not None else "degraded",
        model_loaded=state["model"] is not None,
        model_version=state["model_version"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    start = time.time()
    result = _predict_one(req.text)
    result_dict = result.model_dump()
    result_dict["_latency_ms"] = round((time.time() - start) * 1000, 2)
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest):
    return BatchPredictResponse(results=[_predict_one(t) for t in req.texts])


@app.get("/metrics", tags=["meta"])
def metrics():
    """Lightweight prediction-volume metric (not Prometheus format) used by
    the monitoring workflow to decide whether enough new traffic exists to
    justify a drift check."""
    count = 0
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            count = sum(1 for _ in f)
    return {"logged_predictions": count, "model_version": state["model_version"]}


@app.get("/predictions/export", tags=["monitoring"])
def export_predictions(limit: int = 5000):
    """Return recently logged predictions as JSON.

    scripts/monitor_drift.py calls this against the *deployed* API URL so the
    scheduled drift-check workflow can pull "current" production traffic
    without needing a database or persistent volume — good enough for a
    free-tier demo deployment. Note: Render's free plan has an ephemeral
    filesystem, so this log resets on redeploy/restart; see DEPLOY_GUIDE.md
    for how to upgrade to a persistent store in a real deployment.
    """
    if not LOG_PATH.exists():
        return {"count": 0, "predictions": []}
    records = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = records[-limit:]
    return {"count": len(records), "predictions": records}
