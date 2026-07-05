# How This Project Works

A technical walkthrough of the SaaS Churn & Engagement Dashboard: what happens at each
stage, why the modelling choices were made, and how the pieces fit together. If you just
want to run it, see the [README](README.md); this document explains the internals.

## The big picture

The project answers one question end to end: **which customers are about to churn, and
why?** It does that in five stages, each a single CLI command, with data flowing one
direction through them:

```
 generate / scrape        model.py                server.py
 (raw records)     ┌───────────────────┐    ┌──────────────────┐
      │            │  train + compare  │    │  Flask REST API  │
      ▼            │  RF vs XGBoost    │    │  + dashboard     │
 customers.csv     │  tune threshold   │    │                  │
      │            └───────────────────┘    └──────────────────┘
      ▼                     ▲                        ▲
 data_pipeline.py           │                        │
 clean + engineer  ─────────┴── customers.db ────────┘
 13 features                    (SQLite)
```

Every stage reads the previous stage's output from disk and writes its own, so any stage
can be re-run independently and the whole thing is reproducible from scratch with
`python predict.py run`.

## Stage 1: Getting the data

Customer records come from one of two sources, chosen at runtime:

- **`generate`** builds a synthetic dataset locally (`generate_data.py`), no API key
  required. This is the default and what makes the project reproducible for anyone who
  clones it.
- **`scrape`** pulls real records from the **Stripe API** in test mode (`scraper.py`),
  reading the key from `.env`. This exists to prove the pipeline runs against a real
  billing system, not just a toy CSV.

Either way the output is identical in shape: `data/customers.csv`, one row per customer
with raw fields like plan, logins, support tickets, NPS, invoice history, and subscription
status. Downstream stages do not know or care which source produced it.

## Stage 2: Cleaning and feature engineering (`data_pipeline.py`)

This is where raw records become model-ready features. Three steps run in order.

### Cleaning (`clean`)

Every raw column is read as a string and then coerced safely: numeric fields are parsed
with `errors="coerce"` and imputed with sensible defaults, then **clamped** to realistic
ranges (NPS to 0-10, logins to 0-50, account age to 1-1095 days) so a garbage value cannot
distort the model. Plan names are normalised to the known set (anything unrecognised falls
back to `Free`), and subscription status is lowercased. Free-plan customers, who have no
Stripe subscription object, get an explicit `none` status.

### Feature engineering (`build_features`)

Thirteen features are derived. The non-obvious ones:

- **`engagement_score`** (0-100): a weighted composite, not a raw field. Logins contribute
  up to 40 points, features-used up to 40, and NPS up to 20:
  `login_norm + feat_norm + nps_norm`. This single number turns out to be the strongest
  churn predictor.
- **`failure_rate`**: `payment_failure_count / max(invoice_count, 1)`, so a customer with
  1 failure out of 2 invoices reads as riskier than 1 out of 20.
- **`avg_monthly_spend`**: `total_spend_usd / (account_age_days / 30)`, normalising
  lifetime spend into a monthly rate so new and old accounts are comparable.
- **`days_since_invoice`**: capped at 365; unknowns (negative values) are imputed to 180.
- **`risk_flags`** (0-5): a count of binary warning signals, each independently
  interpretable: very low logins (≤1/week), high support load (≥3 tickets), any payment
  failure, NPS detractor (≤4), and barely using the product (≤1 feature). This is the
  human-readable risk proxy shown in the dashboard's "Churn by Risk Flag Count" chart.

### The churn label, and a deliberate anti-leakage choice

The label is defined in exactly one place:

```python
churned = 1 if subscription_status == "canceled" else 0
```

An earlier version derived churn for Free-plan users from their engagement and support
activity. That was **data leakage**: the label became a deterministic function of features
the model also sees, which inflated AUC to an unrealistic level. The fix (documented in
`MODEL_ISSUES_AND_FIXES.md`) was to make churn an independent ground-truth outcome, a
canceled subscription, for every plan. The current AUC of ~0.87 is honest as a result.

### Persistence (`save_to_db`)

Cleaned rows are written to `data/customers.db` (SQLite) with an **incremental upsert**:
existing customer IDs are deleted and re-inserted, and new feature columns are migrated in
with `ALTER TABLE`. This means re-running the pipeline on fresh data updates the store in
place rather than duplicating rows, which is what a real daily-refresh job needs.

## Stage 3: Training the model (`model.py`)

The model reads the 13 feature columns from SQLite and trains a churn classifier. Several
choices here are deliberate and worth calling out.

**Split before fitting.** The data is split into train (75%) and test (25%) with
`stratify=y` *before* any model touches it, so the test set is a clean holdout and the
reported metrics are not optimistic.

**Two models, keep the winner.** A `RandomForestClassifier` (300 trees, depth 12,
`class_weight="balanced"`) and an `XGBClassifier` (with `scale_pos_weight` set to the
class ratio) are both trained, and whichever scores a higher test AUC-ROC is kept. Both
handle the class imbalance explicitly, since only ~32% of customers churn.

**Tuned decision threshold.** Churn is recall-sensitive: missing a customer who leaves
costs far more than a false alarm on one who stays. The default 0.5 cutoff under-predicts
churn, so the threshold that maximises F1 is chosen, and critically it is chosen on
**out-of-fold** predictions (`cross_val_predict` on the training set), never on the test
set, so threshold selection cannot leak test information. The result lands around 0.41.

**Cross-validated AUC.** Alongside the single test-set AUC, a 5-fold `cross_val_score`
gives a mean and standard deviation, a more honest estimate of generalisation than one
split.

Four artifacts are written to `models/`: the pickled winning model, an encoders file
holding the feature-column order and tuned threshold, `feature_importance.json`, and
`metrics.json`. The dashboard reads the last two directly.

## Stage 4: Serving (`server.py`)

A small Flask app exposes three endpoints and serves the dashboard:

- **`GET /api/data`** returns everything the dashboard renders: KPIs, churn-by-plan,
  churn-by-flags, the probability histogram, feature importance, model metrics, and the
  ranked at-risk table. Predictions are generated by loading the cached model and scoring
  active customers.
- **`POST /api/run`** kicks off the full pipeline (scrape/generate → pipeline → train) in
  the background and returns immediately.
- **`GET /api/status`** is polled by the dashboard to stream the run log and detect
  completion, at which point the charts refresh automatically.

The model is cached in memory keyed by file mtime (`load_model`), so repeated
`/api/data` calls do not re-read the pickle from disk unless it has actually changed.

## Stage 5: The dashboard (`templates/dashboard.html`)

A single self-contained HTML page using Plotly for charts. It shows five KPI cards, four
charts (churn by plan, churn by risk flags, probability distribution, feature importance),
a model-performance card, and a ranked table of the highest-risk active customers, the
intervention list. Every card carries a small "i" info icon that opens a plain-English
explanation of what it shows, so a non-technical stakeholder can read the dashboard
unaided. A "Run Pipeline" button triggers `/api/run` and live-streams the log.

## Design decisions, in one place

- **CLI-first, one command per stage.** Every stage is independently runnable and
  scriptable (`predict.py` via Typer), which keeps the pipeline debuggable and the
  Makefile trivial.
- **Synthetic data by default, Stripe optional.** Reproducible for anyone who clones it,
  while still proving it works against a real billing API.
- **Honest metrics over impressive ones.** The leakage fix lowered the headline AUC on
  purpose; a believable 0.87 is worth more than an inflated 0.99.
- **Recall-weighted threshold.** The whole point is catching churners early, so the
  operating point is tuned for recall rather than left at the default.
- **Everything persisted to disk between stages.** No hidden in-memory state; the DB and
  the model artifacts are the contract between stages.

## Where to look in the code

| You want to understand... | Read |
|---|---|
| How features are built | `data_pipeline.py` → `build_features` |
| The churn label and leakage fix | `data_pipeline.py` → `build_features` + `MODEL_ISSUES_AND_FIXES.md` |
| Model choice and threshold tuning | `model.py` → `train` |
| How predictions are served | `server.py` + `model.py` → `predict_churn_prob` |
| The CLI commands | `predict.py` |
| What each dashboard chart means | click the "i" icons, or `templates/dashboard.html` → `INFO` |
