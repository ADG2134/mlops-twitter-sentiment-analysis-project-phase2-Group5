"""Build the drift-detection reference dataset.

Evidently compares "reference" (what the model was trained/expected to see)
against "current" (recent production traffic). The reference sample is a
small, git-committed CSV so the drift-monitor workflow doesn't need access
to the full 238MB raw dataset or a DVC remote at runtime.

Schema matches what app/main.py logs per prediction (text_length,
cleaned_length, positive_probability, label) so the two sides of the drift
comparison are apples-to-apples.

Usage: python scripts/generate_reference_sample.py [--n 2000]
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
import yaml

from src.prepare import clean_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="reference sample size")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--out", default="data/reference/reference_sample.csv")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        prepare_cfg = yaml.safe_load(f).get("prepare", {})

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(args.train_csv)
    n = min(args.n, len(df))
    sample = df.sample(n=n, random_state=42).reset_index(drop=True)

    # data/processed/train.csv is already cleaned by prepare.py, but we
    # re-run clean_text so the reference is produced the exact same way
    # /predict cleans raw text at inference time (defensive against any
    # future drift-in-preprocessing bugs).
    sample["cleaned_text"] = sample["text"].astype(str).map(lambda t: clean_text(t, prepare_cfg))
    sample["text_length"] = sample["text"].astype(str).str.len()
    sample["cleaned_length"] = sample["cleaned_text"].str.len()

    if hasattr(model, "predict_proba"):
        sample["positive_probability"] = model.predict_proba(sample["cleaned_text"])[:, 1]
    else:
        sample["positive_probability"] = model.predict(sample["cleaned_text"]).astype(float)

    sample["predicted_label"] = (sample["positive_probability"] >= 0.5).map(
        {True: "positive", False: "negative"}
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["text", "cleaned_text", "text_length", "cleaned_length",
            "positive_probability", "predicted_label", "label"]
    sample[cols].to_csv(out_path, index=False)
    print(f"generate_reference_sample: wrote {len(sample)} rows to {out_path}")


if __name__ == "__main__":
    main()
