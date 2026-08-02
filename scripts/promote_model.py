"""Promote a freshly trained model (models/model.pkl, from src/train.py) to
the serving path (models/production/model.pkl) that Dockerfile copies into
the API image.

Guardrail: only promotes if the new model's test metrics (metrics/scores.json,
from src/evaluate.py) don't regress below the proposal thresholds and don't
regress vs the currently-served model's recorded metrics
(models/production/metrics.json), if one exists. This stops a bad retrain
(e.g. triggered by noisy drift on tiny traffic) from silently replacing a
good production model.

Usage: python -m scripts.promote_model [--force]
"""
import argparse
import json
import shutil
from pathlib import Path

THRESHOLDS = {"accuracy": 0.75, "f1": 0.70, "auc_roc": 0.78}
MAX_REGRESSION = 0.02  # allow up to 2 points of metric regression vs current prod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--scores", default="metrics/scores.json")
    parser.add_argument("--reference", default="data/reference/reference_sample.csv")
    parser.add_argument("--prod-dir", default="models/production")
    parser.add_argument("--force", action="store_true", help="skip the metric guardrail")
    args = parser.parse_args()

    with open(args.scores, encoding="utf-8") as f:
        new_scores = json.load(f)

    prod_dir = Path(args.prod_dir)
    prod_scores_path = prod_dir / "metrics.json"
    prod_scores = None
    if prod_scores_path.exists():
        with open(prod_scores_path, encoding="utf-8") as f:
            prod_scores = json.load(f)

    reasons = []
    for metric, floor in THRESHOLDS.items():
        if new_scores.get(metric, 0) < floor:
            reasons.append(f"{metric}={new_scores.get(metric):.4f} below floor {floor}")

    if prod_scores:
        for metric in THRESHOLDS:
            old = prod_scores.get(metric, 0)
            new = new_scores.get(metric, 0)
            if new < old - MAX_REGRESSION:
                reasons.append(
                    f"{metric} regressed: {old:.4f} -> {new:.4f} "
                    f"(more than {MAX_REGRESSION} drop)"
                )

    if reasons and not args.force:
        print("promote_model: REFUSING to promote —")
        for r in reasons:
            print(f"  - {r}")
        return 1

    prod_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.model, prod_dir / "model.pkl")
    with open(prod_scores_path, "w", encoding="utf-8") as f:
        json.dump(new_scores, f, indent=2)

    print(f"promote_model: promoted {args.model} -> {prod_dir / 'model.pkl'}")
    print(f"promote_model: new metrics {json.dumps(new_scores, indent=2)}")
    if reasons:
        print("promote_model: NOTE — promoted with --force despite:")
        for r in reasons:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
