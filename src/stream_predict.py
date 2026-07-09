"""stream_predict.py — real-time sentiment prediction with the trained model.

Two modes:
  1) replay (default, FREE): simulates a live stream by replaying tweets
     from the Sentiment140 test split through the model, one per second.

        python src/stream_predict.py
        python src/stream_predict.py --limit 50 --delay 0.5

  2) x-api (paid, ~$0.005/tweet): pulls recent live tweets from the X API
     v2 recent-search endpoint. Requires a bearer token from a developer
     account with pay-per-use credits (https://developer.x.com).

        export X_BEARER_TOKEN="your-token"
        python src/stream_predict.py --source x-api --query "iphone lang:en" --limit 20

Run from the project root. Uses models/model.pkl produced by `dvc repro`
and the same text cleaning as the prepare stage.
"""
import argparse
import os
import pickle
import re
import sys
import time

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    return WS_RE.sub(" ", t).strip()


def load_model():
    if not os.path.exists("models/model.pkl"):
        sys.exit("models/model.pkl not found — run `dvc repro` first.")
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)


def predict_one(model, raw_text: str) -> str:
    text = clean_text(str(raw_text))
    if not text:
        return "?"
    pred = model.predict([text])[0]
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba([text])[0].max()
        return f"{label} ({conf:.0%})"
    return label


def stream_replay(model, limit: int, delay: float):
    """Replay held-out test tweets as if they were arriving live."""
    import pandas as pd
    path = "data/processed/test.csv"
    if not os.path.exists(path):
        sys.exit(f"{path} not found — run `dvc repro` first.")
    df = pd.read_csv(path).sample(limit, random_state=None)
    print(f"Replaying {limit} tweets from the held-out test set "
          f"(simulated live stream, {delay}s interval)\n")
    correct = 0
    for _, row in df.iterrows():
        result = predict_one(model, row["text"])
        truth = "POSITIVE" if row["label"] == 1 else "NEGATIVE"
        hit = result.startswith(truth)
        correct += hit
        mark = "OK " if hit else "MISS"
        print(f"[{mark}] pred={result:<16} true={truth:<8} | {row['text'][:80]}")
        time.sleep(delay)
    print(f"\nStream accuracy: {correct}/{limit} = {correct/limit:.1%}")


def stream_bluesky(model, query: str, limit: int):
    """Fetch live posts from Bluesky's public search API (FREE, no auth).
    Note: works from residential/campus networks; some cloud IPs are blocked."""
    import requests
    resp = requests.get(
        "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
        params={"q": query, "lang": "en", "limit": min(max(limit, 1), 100)},
        headers={"User-Agent": "student-sentiment-project/1.0"},
        timeout=30,
    )
    if resp.status_code != 200:
        sys.exit(f"Bluesky API error {resp.status_code} — try from a home/campus "
                 f"network, or use --source replay. Body: {resp.text[:200]}")
    posts = resp.json().get("posts", [])
    print(f"Fetched {len(posts)} live Bluesky posts for query: {query!r}\n")
    for p in posts[:limit]:
        text = p.get("record", {}).get("text", "")
        print(f"{predict_one(model, text):<18} | {text[:90]}")


def stream_x_api(model, query: str, limit: int):
    """Fetch recent live tweets via X API v2 (pay-per-use, ~$0.005/read)."""
    import requests
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        sys.exit("Set X_BEARER_TOKEN env var (developer.x.com, pay-per-use credits needed).")
    resp = requests.get(
        "https://api.x.com/2/tweets/search/recent",
        headers={"Authorization": f"Bearer {token}"},
        params={"query": query, "max_results": min(max(limit, 10), 100),
                "tweet.fields": "lang"},
        timeout=30,
    )
    if resp.status_code != 200:
        sys.exit(f"X API error {resp.status_code}: {resp.text[:300]}")
    tweets = resp.json().get("data", [])
    print(f"Fetched {len(tweets)} live tweets for query: {query!r}\n")
    for t in tweets[:limit]:
        print(f"{predict_one(model, t['text']):<18} | {t['text'][:90]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["replay", "bluesky", "x-api"], default="replay")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between replayed tweets")
    ap.add_argument("--query", default="(happy OR sad) lang:en -is:retweet")
    args = ap.parse_args()

    model = load_model()
    if args.source == "replay":
        stream_replay(model, args.limit, args.delay)
    elif args.source == "bluesky":
        stream_bluesky(model, args.query.split(" lang:")[0], args.limit)
    else:
        stream_x_api(model, args.query, args.limit)
