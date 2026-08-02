# Phase 2 Setup & Deploy Guide

Everything in this folder was built and tested locally (FastAPI app trained
against a real sample of your dataset, all 13 tests + ruff passing, the
drift script verified against both drifted and non-drifted synthetic
traffic). It was **not** pushed to your repo or deployed — I don't have
write access to your GitHub account or a cloud account. Follow the steps
below to wire it up; each is copy-paste-able.

---

## 0. One-time repo tweak: `.gitignore`

Your current `.gitignore` blanket-ignores `models/` and `metrics/`. Phase 2
needs to commit a *serving* model artifact and its metrics directly to git
(so Docker builds and the auto-retrain PR don't require DVC-remote
credentials). Open `.gitignore` and replace the `models/` and `metrics/`
lines with:

```gitignore
# --- Phase 2 ---
models/*
!models/production/
!models/production/**

metrics/*
!metrics/scores.json
!metrics/confusion_matrix.json

logs/
```

This keeps `models/model.pkl` (the DVC-tracked experiment artifact) and
`metrics/drift/` (timestamped drift reports) out of git, while allowing the
committed serving copy at `models/production/model.pkl`.

---

## 1. Create the `phase2` branch and add these files

```bash
git clone https://github.com/ADG2134/mlops-twitter-sentiment-analysis-project-phase2-Group5.git
cd mlops-twitter-sentiment-analysis-project-phase2-Group5
git checkout -b phase2

# copy everything from this delivered folder into the repo root, e.g.:
cp -r /path/to/phase2-deliverables/app .
cp -r /path/to/phase2-deliverables/scripts .
cp -r /path/to/phase2-deliverables/tests .
cp -r /path/to/phase2-deliverables/.github .
cp -r /path/to/phase2-deliverables/data/fixtures data/
cp -r /path/to/phase2-deliverables/presentation .
cp /path/to/phase2-deliverables/Dockerfile .
cp /path/to/phase2-deliverables/.dockerignore .
cp /path/to/phase2-deliverables/docker-compose.yml .
cp /path/to/phase2-deliverables/render.yaml .
cp /path/to/phase2-deliverables/requirements-api.txt .
cp /path/to/phase2-deliverables/requirements-dev.txt .
cp /path/to/phase2-deliverables/pyproject.toml .
cp /path/to/phase2-deliverables/MODEL_CARD.md .
cp /path/to/phase2-deliverables/DEPLOY_GUIDE.md .
# then apply the .gitignore edit from Step 0
```

## 2. Train the model you'll actually deploy

The serving model needs to be committed at `models/production/model.pkl`
(see Step 0 for why). Locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-api.txt -r requirements-dev.txt

# place the Kaggle CSV at data/raw/sentiment140.csv (per the main README),
# then optionally bump params.yaml's featurize.ngram_max to 2 — that's the
# "Experiment 1" config from your Phase 1 report and scored best on every
# metric (see MODEL_CARD.md). Also try sample_size: null for the full 1.6M
# rows if you have time/compute for the final numbers.

python src/prepare.py
python src/train.py
python src/evaluate.py
python -m scripts.generate_reference_sample
python -m scripts.promote_model
```

`promote_model.py` refuses to promote if accuracy/F1/AUC fall below the
proposal thresholds — see `MODEL_CARD.md` for the exact numbers. If it
refuses and you're just testing plumbing, rerun with `--force`.

## 3. Test locally with Docker

```bash
docker compose up --build
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "I love this, it works great!"}'
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

## 4. Commit and push

```bash
git add app scripts tests .github data/fixtures data/reference \
        models/production presentation Dockerfile .dockerignore \
        docker-compose.yml render.yaml requirements-api.txt \
        requirements-dev.txt pyproject.toml MODEL_CARD.md DEPLOY_GUIDE.md \
        .gitignore metrics/scores.json metrics/confusion_matrix.json
git commit -m "Phase 2: FastAPI serving, Docker, CI/CD, drift monitoring, retraining"
git push -u origin phase2
```

Open a PR from `phase2` into `main` (or just work on `phase2` directly and
merge later — your call for the class deliverable).

## 5. Create the Render service (free tier, no card)

1. Sign up at [render.com](https://render.com) with your GitHub account.
2. **New > Web Service** → connect this repo → branch `phase2` (switch to
   `main` after you merge).
3. Runtime: **Docker**. Render auto-detects the `Dockerfile`.
4. Plan: **Free**.
5. Click **Create Web Service**. First build takes a few minutes; note the
   public URL, e.g. `https://twitter-sentiment-api.onrender.com` — that's
   your **public API URL** deliverable.
6. Once live: Settings → **Deploy Hook** → copy the URL. You'll need it next.

(Alternatively: `render.yaml` in this repo is a Blueprint — Render can spin
the service up automatically from **New > Blueprint** pointed at your repo,
skipping steps 2–4.)

## 6. Wire up GitHub Actions secrets/variables

Repo **Settings > Secrets and variables > Actions**:

| Type | Name | Value | Used by |
|---|---|---|---|
| Secret | `RENDER_DEPLOY_HOOK_URL` | the Deploy Hook URL from Step 5.6 | `ci.yml` deploy job |
| Variable | `API_URL` | your Render URL, e.g. `https://twitter-sentiment-api.onrender.com` | `drift-monitor.yml` |
| Secret (optional) | `KAGGLE_USERNAME` / `KAGGLE_KEY` | your Kaggle API credentials | `retrain.yml`, to re-download the full dataset |
| Secret (optional) | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | if you set up a DVC S3 remote per the main README | `retrain.yml`, alternative data source |

`GITHUB_TOKEN` (used to trigger `retrain.yml` from `drift-monitor.yml`,
and to open the retrain PR) is provided automatically — no setup needed,
just make sure **Settings > Actions > General > Workflow permissions** is
set to "Read and write permissions."

## 7. Verify the pipeline end to end

- Push a trivial commit to `phase2` (or open/merge a PR into `main`,
  depending on which branch you connected Render to) → watch the **Actions**
  tab: `lint-and-test` should pass, then `deploy` should fire the Render
  hook.
- `curl https://<your-render-url>/health` → `{"status":"ok", ...}`.
- Manually run **Drift Monitor** from the Actions tab (workflow_dispatch) →
  check the uploaded `drift-report-*` artifact HTML.
- Manually run **Retrain** from the Actions tab → it should open a PR named
  `Auto-retrain: refresh production model` (requires Kaggle or DVC-remote
  secrets from Step 6, or it'll fail at the "Fetch raw dataset" step with a
  clear error message).

## Known limitations to mention in your presentation

- **Render free tier has an ephemeral filesystem** and spins the container
  down after 15 minutes idle. `logs/predictions.jsonl` (and therefore
  `/predictions/export`, which the drift monitor reads) resets on every
  redeploy/restart. Fine for a live demo; a real deployment would ship
  predictions to a database or log store instead (call this out as a
  "next steps" slide — it's a legitimate, expected trade-off for a free
  tier, not an oversight).
- **First request after idle is slow** (cold start, ~30–50s) — mention this
  if you demo it live, or hit `/health` a minute before your demo starts.
- **Retraining requires a data source secret** (Kaggle or DVC remote) that
  isn't configured by default — the workflow fails loudly and explains why
  rather than silently no-op'ing.
