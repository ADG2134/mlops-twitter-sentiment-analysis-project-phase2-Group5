"""API tests. Run with a real trained model available at models/model.pkl
(CI trains a tiny throwaway model on a data fixture first — see ci.yml)."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PREDICTION_LOG_PATH", "logs/test_predictions.jsonl")

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # `with` triggers FastAPI's lifespan handler (startup/shutdown), which
    # is what actually loads the model into memory. A bare TestClient(app)
    # skips lifespan and every /predict call would 503.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200


@pytest.mark.skipif(
    not os.path.exists(os.environ.get("MODEL_PATH", "models/model.pkl")),
    reason="no trained model artifact available in this environment",
)
def test_predict_positive_text(client):
    r = client.post("/predict", json={"text": "I love this, it works great!"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("positive", "negative")
    assert 0.0 <= body["positive_probability"] <= 1.0


@pytest.mark.skipif(
    not os.path.exists(os.environ.get("MODEL_PATH", "models/model.pkl")),
    reason="no trained model artifact available in this environment",
)
def test_predict_rejects_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


@pytest.mark.skipif(
    not os.path.exists(os.environ.get("MODEL_PATH", "models/model.pkl")),
    reason="no trained model artifact available in this environment",
)
def test_predict_batch(client):
    r = client.post(
        "/predict/batch",
        json={"texts": ["great job team", "this is terrible"]},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "logged_predictions" in r.json()
