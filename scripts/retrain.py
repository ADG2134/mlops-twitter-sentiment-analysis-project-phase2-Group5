"""End-to-end retrain: prepare -> train -> evaluate -> refresh reference
sample -> promote (if metrics clear the guardrail in promote_model.py).

Assumes data/raw/sentiment140.csv is already present — the retrain workflow
(.github/workflows/retrain.yml) fetches it first via `dvc pull` (if a DVC
remote is configured) or the Kaggle API (if KAGGLE_USERNAME/KAGGLE_KEY
secrets are set). See DEPLOY_GUIDE.md.

Usage: python -m scripts.retrain [--force-promote]
"""
import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-promote", action="store_true")
    args = parser.parse_args()

    run([sys.executable, "src/prepare.py"])
    run([sys.executable, "src/train.py"])
    run([sys.executable, "src/evaluate.py"])
    run([sys.executable, "-m", "scripts.generate_reference_sample"])

    promote_cmd = [sys.executable, "-m", "scripts.promote_model"]
    if args.force_promote:
        promote_cmd.append("--force")

    result = subprocess.run(promote_cmd)
    if result.returncode != 0:
        print(
            "retrain: model trained but NOT promoted (guardrail failed). "
            "Review metrics/scores.json before re-running with --force-promote.",
            file=sys.stderr,
        )
        return result.returncode

    print("retrain: complete, new model promoted to models/production/model.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
