# SaaS Customer Churn & Engagement Dashboard

Predict which SaaS customers are about to churn, and show *why*, through an end-to-end
pipeline that runs from raw billing data to a trained model to an interactive dashboard.

Built around a realistic Stripe-style billing dataset (520 customers across 4 plan tiers),
the project engineers 13 churn-risk features, trains and compares Random Forest and XGBoost
classifiers with 5-fold cross-validation, and serves live churn predictions plus feature
importance through a Flask dashboard. The whole pipeline is driven by a single CLI and can be
reproduced locally with no API key required.

For a stage-by-stage technical walkthrough of how it all fits together, see
**[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**.

## Key results

| Metric | Value |
|---|---|
| Model selected | Random Forest (by AUC-ROC vs XGBoost) |
| Test AUC-ROC | **0.87** |
| 5-fold CV AUC | 0.805 (std 0.023) |
| Recall (churners caught) | 0.88 |
| Precision | 0.58 |
| Dataset | 520 customers, 31.9% churn rate |
| Decision threshold | 0.41 (tuned, not default 0.5) |

The threshold is deliberately tuned below 0.5 to favour recall: for churn, missing a customer who
leaves costs more than a false alarm on one who stays.

### Top churn drivers (model feature importance)

1. **Engagement score** (0.20) — the single strongest signal
2. **Average monthly spend** (0.11)
3. **Days since last invoice** (0.11)
4. **Logins per week** (0.10)
5. **Plan tier** (0.09)

Product engagement dominates: how much customers *use* the product predicts churn better than
what they pay.

## How it works

```
Stripe API / synthetic generator
        │
        ▼
  scrape / generate        raw customer records → data/customers.csv
        │
        ▼
  pipeline                 clean + engineer 13 features → data/customers.db (SQLite, upsert)
        │
        ▼
  train                    Random Forest vs XGBoost, 5-fold CV → models/*.pkl + metrics.json
        │
        ▼
  serve                    Flask REST API + interactive dashboard
```

### Features engineered (13)

Composite `engagement_score`, `avg_monthly_spend`, `days_since_invoice`, `logins_per_week`,
`plan_tier`, `features_used`, `account_age_days`, `failure_rate`, `payment_failure_count`,
`nps_score`, `support_tickets`, `invoice_count`, and a `risk_flags` count.

## Tech stack

**Python**, **Pandas**, **scikit-learn**, **XGBoost**, **SQLite**, **Flask** (REST API +
dashboard), **Typer** + **Rich** (CLI), **Stripe API** (optional live data source), **pytest**.

## Quickstart

```bash
# 1. Install
make install                 # or: pip install -r requirements.txt

# 2. Run the whole pipeline locally (no Stripe key needed)
python predict.py run        # generate → pipeline → train

# 3. Launch the dashboard
python predict.py serve      # http://127.0.0.1:8080
```

Every stage is also available on its own:

```bash
python predict.py generate -n 520   # synthetic data, no API key
python predict.py pipeline          # clean + feature engineering
python predict.py train             # train + cross-validate
python predict.py serve -p 8080     # dashboard
python predict.py --help            # all commands
make test                           # run the test suite
```

### Using live Stripe data (optional)

```bash
cp .env.example .env                # then add your Stripe test key
python predict.py run --source stripe
```

Only a Stripe **test-mode** key is used, and it is read from `.env`, which is gitignored and
never committed.

## Project structure

```
predict.py            CLI entry point (Typer) — all commands route through here
generate_data.py      synthetic customer generator (no API key)
setup_stripe_data.py  seed synthetic customers into Stripe test mode
scraper.py            fetch customer data from the Stripe API
data_pipeline.py      cleaning + feature engineering → SQLite (incremental upsert)
model.py              train / cross-validate Random Forest & XGBoost, save artifacts
server.py             Flask REST API + dashboard
templates/            dashboard HTML
tests/                pytest suite for the pipeline
models/               metrics.json + feature_importance.json (trained artifacts)
```

## Notes

- The dataset is synthetic but modelled on real Stripe billing structure, so the pipeline runs
  identically against live Stripe test data or the local generator.
- `MODEL_ISSUES_AND_FIXES.md` documents modelling problems found and fixed during development
  (data leakage, threshold tuning, class imbalance), kept as an honest engineering log.
