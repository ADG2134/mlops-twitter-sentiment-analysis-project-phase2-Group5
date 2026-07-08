"""Stage 3 — evaluate: score the trained model on the held-out test set.

Metrics go to BOTH:
  - metrics/scores.json  (tracked by DVC: `dvc metrics show`, `dvc metrics diff`)
  - the same MLflow run that train.py created (via models/run_id.txt)
Also saves a confusion-matrix artifact to MLflow.
"""
import json
import os
import pickle

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def main() -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    ml_cfg = params["mlflow"]

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment_name"])

    with open("models/model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    test_df = pd.read_csv("data/processed/test.csv")
    X, y = test_df["text"].astype(str), test_df["label"]
    preds = pipeline.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
    }

    # AUC needs scores/probabilities; LinearSVC exposes decision_function only.
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        scores = pipeline.predict_proba(X)[:, 1]
        metrics["auc_roc"] = roc_auc_score(y, scores)
    elif hasattr(clf, "decision_function"):
        metrics["auc_roc"] = roc_auc_score(y, pipeline.decision_function(X))

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y, preds).tolist()
    with open("metrics/confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"labels": ["negative", "positive"], "matrix": cm}, f, indent=2)

    # Attach test metrics to the SAME run train.py opened.
    with open("models/run_id.txt", encoding="utf-8") as f:
        run_id = f.read().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact("metrics/confusion_matrix.json")

    print("evaluate:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
