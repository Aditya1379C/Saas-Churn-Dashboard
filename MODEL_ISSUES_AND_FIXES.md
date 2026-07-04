# Model Issues and Fixes

A log of data and modelling problems found in the churn pipeline, why each one
mattered, and how it was fixed. The goal is an honest model, not an impressive
but misleading metric.

---

## Issue 1: Label leakage on 39% of the data

**Symptom.** The model reported a strong AUC of 0.887, but the top feature
(`engagement_score`, 0.33 importance) looked suspiciously dominant.

**Root cause.** Free-plan customers have no Stripe subscription, so they can
never be `canceled`. To give them a churn label, the pipeline defined:

```python
# data_pipeline.py (old)
df.loc[free_mask, "churned"] = (
    (df.loc[free_mask, "engagement_score"] < 15)
    | (df.loc[free_mask, "support_tickets"] >= 3)
).astype(int)
```

`engagement_score` and `support_tickets` are also model **features**. For the
204 Free customers (39% of 520 rows), the label was a deterministic function of
inputs the model could see, so the model was partly memorising its own target.
That inflated AUC.

**Fix.** Use a single, independent source of truth for the label across every
plan: a canceled subscription. The Free-plan engagement override was removed
entirely.

```python
# data_pipeline.py (new)
df["churned"] = (df["subscription_status"] == "canceled").astype(int)
```

---

## Issue 2: Churn was independent of the engagement features

**Symptom.** Even setting leakage aside, within any single plan the engagement
features had no real relationship to churn.

**Root cause.** In the Stripe seeder, churn was assigned by a per-plan coin
flip, unrelated to a customer's behaviour:

```python
# setup_stripe_data.py
churned = random.random() < CHURN_RATES[plan]   # depends only on plan
```

So the only honest signal in the data was `plan` itself. A correctly-built
model would have learned almost nothing from engagement, and any apparent skill
came from the leakage in Issue 1.

**Fix.** A new local generator (`generate_data.py`) makes churn a realistic,
*stochastic* function of behaviour, plus irreducible noise:

```python
z = (base_logit[plan]
     - 0.13*logins - 0.10*features + 0.42*tickets - 0.28*(nps-5)
     + 0.70*payment_failures + 0.012*days_since_invoice
     - 0.0008*account_age + gauss(0, 0.8))
churned = random.random() < sigmoid(z)
```

The relationship is genuine but imperfect, so the model has something real to
learn and the metrics are trustworthy. The generator needs no API key, which
also makes the whole project reproducible from a clean clone.

---

## Issue 3: Two dead features

**Symptom.** `payment_failure_count` and its derivative `failure_rate` both had
**0.0** importance.

**Root cause.** `payment_failure_count` was `0` for all 520 rows. The Stripe
seeder created declined `PaymentIntent`s with `confirm=True`, which raises
`CardError` before the object persists, so the scraper never counted any. The
feature was constant, therefore useless, and `failure_rate` (= failures /
invoices) was always 0 too.

**Fix.** The local generator produces realistic, churn-correlated payment
failures (Poisson-style, higher on cheaper plans). Both features now carry real
signal (`failure_rate` 0.055, `payment_failure_count` 0.045 importance), and no
feature sits at 0.0.

---

## Issue 4: Near-constant invoice recency

**Symptom.** `last_invoice_days_ago` took only the values 0, 1, or -1, so the
derived `days_since_invoice` collapsed to roughly "Free vs paid."

**Root cause.** The seeded invoices were all finalised at the same time, so
recency never varied.

**Fix.** The generator draws invoice recency with realistic spread, widening for
at-risk customers (more support tickets, fewer logins). `days_since_invoice` is
now the #3 feature (0.107 importance).

---

## Issue 5: Default 0.5 threshold under-predicted churn

**Symptom.** Recall was only 0.556, so the model missed nearly half of all
churners.

**Why it matters.** Churn is recall-sensitive: a missed churner is a lost
customer, while a false alarm just triggers an unnecessary retention nudge. The
default 0.5 classification cutoff optimises accuracy, which is the wrong target.

**Fix.** `model.py` now tunes the decision threshold by maximising F1 on
**out-of-fold** training predictions (via `cross_val_predict`), never on the
test set, then applies that threshold to the held-out test set. The tuned
threshold (0.41) is saved in `metrics.json` and `encoders.pkl`, and a
`predict_churn_label()` helper applies it at inference.

---

## Before vs after

| Metric | Before (leaky) | After (honest) |
|--------|---------------:|---------------:|
| Test AUC-ROC      | 0.887 | 0.875 |
| CV AUC (mean ± sd)| 0.887 ± 0.029 | 0.805 ± 0.023 |
| Recall            | 0.556 | 0.881 |
| Precision         | 0.800 | 0.578 |
| F1                | 0.656 | 0.698 |
| Decision threshold| 0.50 (default) | 0.41 (tuned) |
| Features at 0.0 importance | 2 | 0 |
| Label source | mixed (status + engagement) | subscription status only |

The headline AUC barely moved, but it now reflects genuine predictive signal
rather than leakage, cross-validation is honest, every feature contributes, and
the model catches 88% of churners instead of 56%. Trading some precision for a
large recall gain is the correct call for a churn use case.
