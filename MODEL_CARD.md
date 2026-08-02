# Model Card — Twitter Sentiment Classifier

## Model Details

- **Model type:** TF-IDF (unigrams + bigrams) → Logistic Regression, scikit-learn `Pipeline`.
- **Version served in Phase 2:** `logreg-tfidf-ng2` ("Experiment 1" from the Phase 1 report) — the best-performing of the three configurations tested, chosen for deployment over the originally-committed unigram baseline.
- **Framework:** scikit-learn ≥ 1.4.
- **Input:** a single raw tweet-length string (≤ 1000 characters).
- **Output:** binary label (`positive` / `negative`) plus `P(positive)`.
- **Owners:** Debolina Das (ML Lead), Monireh Eshghinezhad (Engineering Lead), Anish Dasgupta (Project Lead) — MAI201, Seneca Polytechnic.
- **License / use:** academic coursework project, not for commercial deployment as-is.

## Intended Use

- **Primary use case:** demonstrating an end-to-end MLOps pipeline (data → training → tracking → serving → CI/CD → monitoring → retraining) for a course assignment.
- **In scope:** short, informal, English-language, social-media-style text similar to 2009–2010 Twitter content.
- **Out of scope:** long-form text, non-English text, sarcasm-heavy or highly context-dependent statements, financial/medical/legal decision-making, moderation or safety-critical filtering of real user content. This model should not be used to make consequential decisions about real people.

## Training Data

- **Source:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140), 1.6M tweets, distant-supervision labels derived from emoticons (then stripped from the text).
- **Labels:** binary only — `0` = negative, `4`→`1` = positive. The released dataset contains **no neutral tweets**, and neither does this model's output space. Any operational use expecting neutral/mixed sentiment detection will misclassify.
- **Time period:** 2009–2010. Slang, topics, and even emoji/emoticon conventions have shifted materially since then — see "Known Limitations."
- **Preprocessing:** lowercased, URLs stripped, `@mentions` stripped, `#` symbol stripped (hashtag word kept). See `src/prepare.py::clean_text`, reused verbatim by the API at inference time so training/serving skew is minimized.
- **Class balance:** exactly 50/50 by construction (Sentiment140 itself, and the training sample drawn from it).

## Evaluation Results (Phase 1, 100K stratified sample, 80/20 split, seed 42)

| Configuration | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression · TF-IDF · unigrams (baseline) | 0.785 | 0.787 | 0.864 |
| **Logistic Regression · TF-IDF · bigrams (deployed)** | **0.792** | **0.793** | **0.872** |
| Multinomial Naive Bayes · bigrams | 0.784 | 0.782 | 0.864 |

Against the original proposal targets (accuracy ≥ 80%, F1 ≥ 0.73/class, AUC ≥ 0.80): F1 and AUC are met on the 100K sample; accuracy is within a point and expected to close on the full 1.6M-tweet training run (`sample_size: null` in `params.yaml`). Bigrams help most on negation ("not good"), which is consistent with the accuracy/F1/AUC gain over unigrams.

`scripts/promote_model.py` enforces a hard floor (accuracy ≥ 0.75, F1 ≥ 0.70, AUC ≥ 0.78) plus a max-2-point regression check against the currently-served model before any retrained model can replace it in production.

## Known Limitations

- **Temporal drift by construction.** The model is trained on 2009–2010 tweets. Vocabulary, topics, and platform norms (e.g., character limits, meme formats) have changed substantially; expect accuracy to be lower on current text than the reported metrics. This is exactly the kind of drift `scripts/monitor_drift.py` is built to catch in production.
- **Label noise.** Emoticon-based distant supervision is a noisy proxy for sentiment — a few percent of labels are expected to be wrong, capping achievable accuracy regardless of model choice.
- **No neutral class.** Confidently-neutral or mixed statements get forced into positive/negative; treat mid-range `positive_probability` (e.g. 0.4–0.6) as low-confidence, not "neutral."
- **English only,** and mostly informal register; performance on formal text, non-English text, or code-mixed text is unknown/untested.
- **Not robust to adversarial input** (e.g. deliberate obfuscation, leetspeak) — no adversarial evaluation was performed.
- **Demographic/topic bias untested.** No fairness or subgroup evaluation was performed across dialects, named entities, or topics. Sentiment140's collection method (keyword/emoticon search circa 2009) is not representative of any specific present-day population, and outputs should not be used to characterize opinions of any real demographic group.

## Ethical Considerations

- This is a coursework artifact, not a vetted production sentiment system. It should not be used for content moderation, hiring/HR signal, mental-health inference, or any decision with real consequences for a person.
- Predictions are logged (`logs/predictions.jsonl` / `/predictions/export`) for drift monitoring. In the demo deployment this may include raw user-submitted text; do not submit personal, sensitive, or identifying information to the public demo endpoint.

## How the Model Is Served and Kept Fresh

1. **Serving:** FastAPI (`app/main.py`) loads `models/production/model.pkl` at startup, containerized via `Dockerfile`, deployed to Render (`render.yaml`).
2. **CI/CD:** every push runs lint + tests (`.github/workflows/ci.yml`); pushes to `main` additionally trigger a Render deploy.
3. **Monitoring:** `.github/workflows/drift-monitor.yml` runs daily (and on demand), comparing recent production predictions against `data/reference/reference_sample.csv` using EvidentlyAI's `DataDriftPreset`, plus a class-balance check.
4. **Retraining:** if drift is detected, `.github/workflows/retrain.yml` is triggered automatically, retrains on the full dataset, and opens a PR (human review required before it reaches production) rather than auto-merging.

See `DEPLOY_GUIDE.md` for the exact setup steps and required secrets.
