"""Stage 2 — train: vectorize text, train a classifier, log everything to MLflow.

Input : data/processed/train.csv
Output: models/model.pkl (sklearn Pipeline: vectorizer + classifier)
        models/run_id.txt (MLflow run id, consumed by evaluate stage)
"""
import os
import pickle

import mlflow
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_vectorizer(cfg: dict):
    common = dict(
        max_features=cfg["max_features"],
        ngram_range=(1, cfg["ngram_max"]),
    )
    if cfg["vectorizer"] == "tfidf":
        return TfidfVectorizer(**common)
    if cfg["vectorizer"] == "count":
        return CountVectorizer(**common)
    raise ValueError(f"Unknown vectorizer: {cfg['vectorizer']}")


def build_model(cfg: dict):
    class_weight = cfg.get("class_weight") or None
    if cfg["model"] == "logreg":
        return LogisticRegression(
            C=cfg["C"], max_iter=cfg["max_iter"], class_weight=class_weight)
    if cfg["model"] == "linearsvc":
        return LinearSVC(C=cfg["C"], max_iter=cfg["max_iter"],
                         class_weight=class_weight)
    if cfg["model"] == "nb":
        return MultinomialNB()
    raise ValueError(f"Unknown model: {cfg['model']}")


def main() -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    feat_cfg, train_cfg, ml_cfg = (
        params["featurize"], params["train"], params["mlflow"])

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment_name"])

    train_df = pd.read_csv("data/processed/train.csv")
    X, y = train_df["text"].astype(str), train_df["label"]

    pipeline = Pipeline([
        ("vectorizer", build_vectorizer(feat_cfg)),
        ("clf", build_model(train_cfg)),
    ])

    run_name = f"{train_cfg['model']}-{feat_cfg['vectorizer']}-ng{feat_cfg['ngram_max']}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "vectorizer": feat_cfg["vectorizer"],
            "max_features": feat_cfg["max_features"],
            "ngram_max": feat_cfg["ngram_max"],
            "model": train_cfg["model"],
            "C": train_cfg["C"],
            "class_weight": train_cfg.get("class_weight"),
            "train_rows": len(train_df),
        })

        pipeline.fit(X, y)
        train_acc = pipeline.score(X, y)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.sklearn.log_model(pipeline, name="model")

        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(pipeline, f)
        with open("models/run_id.txt", "w", encoding="utf-8") as f:
            f.write(run.info.run_id)

        print(f"train: run_id={run.info.run_id} train_accuracy={train_acc:.4f}")


if __name__ == "__main__":
    main()
