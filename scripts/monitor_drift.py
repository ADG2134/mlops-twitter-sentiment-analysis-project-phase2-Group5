"""Drift detection: compare the committed reference sample against recent
production predictions using EvidentlyAI.

Two ways to get "current" data:
  --source local   read a local JSONL prediction log (used in CI / local dev)
  --source api      GET {API_URL}/predictions/export from the *deployed*
                    service (used by the scheduled drift-monitor workflow)

Outputs:
  metrics/drift/drift_report_<ts>.html   full interactive Evidently report
  metrics/drift/drift_summary.json       {"drift_detected": bool, ...}
  GITHUB_OUTPUT (if set)                 drift_detected=true|false, for the
                                          workflow to conditionally trigger
                                          scripts/retrain.py

Run: python -m scripts.monitor_drift --source api --api-url https://your-api.onrender.com
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Below this many current-traffic rows, a drift verdict is unreliable —
# report but don't fail/retrain on noise from a handful of requests.
MIN_CURRENT_ROWS = 30

COLUMN_MAPPING = ColumnMapping(
    numerical_features=["text_length", "cleaned_length", "positive_probability"],
    text_features=[],  # kept numeric-only for CI speed; see README for the
                        # TextOverviewPreset variant if you want NLP drift too
)


def load_reference(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_current_local(log_path: str) -> pd.DataFrame:
    records = []
    p = Path(log_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return pd.DataFrame.from_records(records)


def load_current_api(api_url: str, limit: int) -> pd.DataFrame:
    resp = requests.get(f"{api_url.rstrip('/')}/predictions/export", params={"limit": limit}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return pd.DataFrame.from_records(payload.get("predictions", []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "api"], default="local")
    parser.add_argument("--api-url", default=os.getenv("API_URL", ""))
    parser.add_argument("--local-log", default="logs/predictions.jsonl")
    parser.add_argument("--reference", default="data/reference/reference_sample.csv")
    parser.add_argument("--out-dir", default="metrics/drift")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--drift-share-threshold", type=float, default=0.5,
                         help="fraction of monitored columns that must drift to flag dataset drift")
    args = parser.parse_args()

    reference_df = load_reference(args.reference)

    if args.source == "api":
        if not args.api_url:
            print("::error::--source api requires --api-url or $API_URL", file=sys.stderr)
            return 2
        current_df = load_current_api(args.api_url, args.limit)
    else:
        current_df = load_current_local(args.local_log)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary = {
        "timestamp": timestamp,
        "source": args.source,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
    }

    if len(current_df) < MIN_CURRENT_ROWS:
        summary["drift_detected"] = False
        summary["skipped"] = True
        summary["reason"] = (
            f"only {len(current_df)} current rows (< {MIN_CURRENT_ROWS}); "
            "not enough traffic for a reliable drift verdict"
        )
        _write_summary(out_dir, summary)
        _set_github_output("drift_detected", "false")
        print(json.dumps(summary, indent=2))
        return 0

    needed_cols = ["text_length", "cleaned_length", "positive_probability"]
    for col in needed_cols:
        if col not in current_df.columns:
            current_df[col] = pd.NA

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[needed_cols],
        current_data=current_df[needed_cols],
        column_mapping=COLUMN_MAPPING,
    )

    html_path = out_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))

    result = report.as_dict()
    drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )
    dataset_drift = bool(drift_metric["result"]["dataset_drift"])
    share_drifted = drift_metric["result"]["share_of_drifted_columns"]

    # Also track class-balance drift directly: Sentiment140 reference is a
    # ~50/50 balanced sample, so a swing in predicted-positive rate is a
    # meaningful, easy-to-explain signal on top of Evidently's feature-drift
    # verdict.
    ref_positive_rate = (reference_df["positive_probability"] >= 0.5).mean()
    cur_positive_rate = (current_df["positive_probability"] >= 0.5).mean()
    positive_rate_delta = abs(cur_positive_rate - ref_positive_rate)

    summary.update({
        "drift_detected": dataset_drift or share_drifted >= args.drift_share_threshold,
        "dataset_drift_flag": dataset_drift,
        "share_of_drifted_columns": share_drifted,
        "reference_positive_rate": ref_positive_rate,
        "current_positive_rate": cur_positive_rate,
        "positive_rate_delta": positive_rate_delta,
        "report_html": str(html_path),
    })
    _write_summary(out_dir, summary)
    _set_github_output("drift_detected", "true" if summary["drift_detected"] else "false")
    print(json.dumps(summary, indent=2))
    return 0


def _write_summary(out_dir: Path, summary: dict) -> None:
    with open(out_dir / "drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def _set_github_output(key: str, value: str) -> None:
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    raise SystemExit(main())
