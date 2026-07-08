"""Stage 1 — prepare: load raw Sentiment140 CSV, clean text, split train/test.

Input : data/raw/sentiment140.csv  (Kaggle format, latin-1, no header)
Output: data/processed/train.csv, data/processed/test.csv
"""
import os
import re
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_COLUMNS = ["target", "id", "date", "flag", "user", "text"]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, cfg: dict) -> str:
    if cfg.get("lowercase", True):
        text = text.lower()
    if cfg.get("remove_urls", True):
        text = URL_RE.sub(" ", text)
    if cfg.get("remove_mentions", True):
        text = MENTION_RE.sub(" ", text)
    if cfg.get("remove_hashtag_symbol", True):
        text = text.replace("#", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def main() -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    cfg = params["prepare"]

    raw_path = cfg["raw_path"]
    if not os.path.exists(raw_path):
        sys.exit(
            f"Raw dataset not found at {raw_path}.\n"
            "Download Sentiment140 from Kaggle "
            "(https://www.kaggle.com/datasets/kazanova/sentiment140), "
            "then place the CSV at that path."
        )

    df = pd.read_csv(raw_path, encoding="latin-1", header=None, names=RAW_COLUMNS)

    # Sentiment140 labels: 0 = negative, 4 = positive (no neutral in the released set)
    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = (df["target"] == 4).astype(int)  # 1 = positive, 0 = negative

    sample_size = cfg.get("sample_size")
    if sample_size:
        n = min(sample_size // 2, df["label"].value_counts().min())
        df = (
            df.groupby("label")
            .sample(n=n, random_state=cfg["random_state"])
            .reset_index(drop=True)
        )

    df["text"] = df["text"].astype(str).map(lambda t: clean_text(t, cfg))
    df = df[df["text"].str.len() > 0][["text", "label"]]

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df["label"],
    )

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"prepare: {len(train_df)} train rows, {len(test_df)} test rows")
    print(f"prepare: class balance (train) = {train_df['label'].mean():.3f} positive")


if __name__ == "__main__":
    main()
