#!/usr/bin/env python3
"""
generate_data.py
================
Local, no-Stripe synthetic data generator for the churn pipeline.

Why this exists
---------------
The original data came from Stripe test mode (setup_stripe_data.py), where
churn was assigned by a per-plan coin flip *independent* of the engagement
signals. That meant the engagement features carried no real predictive signal,
and the pipeline had to fabricate a churn label for Free users directly from
`engagement_score` — which leaked into the model (see MODEL_ISSUES_AND_FIXES.md).

This generator fixes the root cause: churn here is a realistic, *stochastic*
function of the customer's behaviour (engagement, tenure, support load, NPS,
payment failures, invoice recency) plus noise. The relationship is genuine but
imperfect, so the model has something real to learn and the reported metrics are
honest — no leakage, no dead features.

It writes the exact same schema as scraper.py, so the rest of the pipeline
(data_pipeline.py → model.py → report.py) is unchanged. Runs with no API key:

    python generate_data.py            # 520 rows (default)
    SEED_COUNT=2000 python generate_data.py
"""

import csv
import math
import os
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE     = Path(__file__).parent
CSV_PATH = HERE / "data" / "customers.csv"

SEED_COUNT = int(os.getenv("SEED_COUNT", "520"))
random.seed(42)

# ── Plan setup ────────────────────────────────────────────────────────────────

PLANS = {
    "Free":       {"amount": 0,   "weight": 0.35, "base_logit": -0.4},
    "Starter":    {"amount": 29,  "weight": 0.30, "base_logit": -0.9},
    "Pro":        {"amount": 99,  "weight": 0.25, "base_logit": -1.6},
    "Enterprise": {"amount": 299, "weight": 0.10, "base_logit": -2.3},
}
PLAN_NAMES   = list(PLANS.keys())
PLAN_WEIGHTS = [PLANS[p]["weight"] for p in PLAN_NAMES]

# Per-plan means for engagement proxies (drawn from gaussians).
ENGAGE_MEAN = {
    "Free":       {"logins": 2,  "features": 2,  "tickets": 1.5, "nps": 5},
    "Starter":    {"logins": 5,  "features": 5,  "tickets": 0.8, "nps": 6},
    "Pro":        {"logins": 10, "features": 9,  "tickets": 0.4, "nps": 7},
    "Enterprise": {"logins": 15, "features": 14, "tickets": 0.2, "nps": 8},
}

INDUSTRIES = ["E-commerce", "FinTech", "HealthTech", "EdTech", "MarTech",
              "HRTech", "LegalTech", "PropTech", "SaaS", "AgriTech"]
SOURCES    = ["organic", "referral", "paid_search", "social", "product_hunt", "content"]
SIZES      = ["1-10", "11-50", "51-200", "201-1000", "1000+"]

FIRST_NAMES = [
    "Liam", "Noah", "Oliver", "Elijah", "James", "Lucas", "Mason", "Ethan",
    "Emma", "Olivia", "Ava", "Sophia", "Isabella", "Mia", "Charlotte", "Amelia",
    "Rohan", "Priya", "Arjun", "Sneha", "Dev", "Ananya", "Vikram", "Meera",
    "Mohammed", "Fatima", "Omar", "Layla", "Yusuf", "Aisha", "Hassan", "Zara",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Garcia", "Miller", "Davis", "Wilson",
    "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Mehta", "Joshi", "Rao",
    "Ali", "Khan", "Ahmed", "Hassan", "Malik", "Sheikh", "Siddiqui", "Rahman",
]


def weighted_choice(choices, weights):
    r, cum = random.uniform(0, sum(weights)), 0.0
    for c, w in zip(choices, weights):
        cum += w
        if r <= cum:
            return c
    return choices[-1]


def rand_email(name: str) -> str:
    domain = random.choice(["gmail.com", "company.io", "startup.co", "corp.com"])
    suffix = "".join(random.choices(string.digits, k=3))
    return f"{name.lower().replace(' ', '.')}{suffix}@{domain}"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_customer(idx: int) -> dict:
    plan = weighted_choice(PLAN_NAMES, PLAN_WEIGHTS)
    em   = ENGAGE_MEAN[plan]

    account_age = random.randint(30, 730)
    logins   = max(0, min(50, int(random.gauss(em["logins"], 3))))
    features = max(0, min(30, int(random.gauss(em["features"], 3))))
    tickets  = max(0, min(20, int(random.gauss(em["tickets"], 1.2))))
    nps      = max(0, min(10, int(random.gauss(em["nps"], 2))))

    # Billing — only paid plans invoice.
    if plan == "Free":
        invoice_count = 0
        total_spend   = 0.0
        last_invoice  = random.randint(20, 120)          # last activity, not billing
        fail_count    = 0
    else:
        invoice_count = max(1, int(account_age / 30) + random.randint(-1, 1))
        total_spend   = round(invoice_count * PLANS[plan]["amount"], 2)
        # Recency of last successful invoice — wider gaps for at-risk customers.
        last_invoice  = max(0, int(random.gauss(18 + tickets * 3 - logins, 10)))
        # Payment failures — Poisson-ish, more likely on cheaper plans.
        lam        = {"Starter": 0.35, "Pro": 0.15, "Enterprise": 0.06}[plan]
        fail_count = min(5, sum(1 for _ in range(6) if random.random() < lam))

    # ── Churn propensity: a genuine but noisy function of behaviour ──────────
    # No hard threshold, no feature-equals-label. churn ~ Bernoulli(sigmoid(z)).
    z = (
        PLANS[plan]["base_logit"]
        - 0.13 * logins
        - 0.10 * features
        + 0.42 * tickets
        - 0.28 * (nps - 5)
        + 0.70 * fail_count
        + 0.012 * last_invoice
        - 0.0008 * account_age
        + random.gauss(0, 0.8)            # irreducible noise
    )
    churned = random.random() < sigmoid(z)

    # subscription_status is the single source of truth for the label.
    status = "canceled" if churned else "active"

    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    created = (datetime.now(timezone.utc) - timedelta(days=account_age)).strftime("%Y-%m-%d")

    return {
        "customer_id":          f"cus_syn{idx:05d}",
        "name":                 name,
        "email":                rand_email(name),
        "plan":                 plan,
        "industry":             random.choice(INDUSTRIES),
        "company_size":         random.choice(SIZES),
        "signup_source":        random.choice(SOURCES),
        "account_age_days":     account_age,
        "logins_per_week":      logins,
        "features_used":        features,
        "support_tickets":      tickets,
        "nps_score":            nps,
        "subscription_status":  status,
        "payment_failure_count": fail_count,
        "invoice_count":        invoice_count,
        "total_spend_usd":      total_spend,
        "last_invoice_days_ago": last_invoice,
        "created_at":           created,
    }


HEADERS = [
    "customer_id", "name", "email", "plan", "industry", "company_size",
    "signup_source", "account_age_days", "logins_per_week", "features_used",
    "support_tickets", "nps_score", "subscription_status", "payment_failure_count",
    "invoice_count", "total_spend_usd", "last_invoice_days_ago", "created_at",
]


def main():
    rows = [make_customer(i) for i in range(SEED_COUNT)]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    churn = sum(1 for r in rows if r["subscription_status"] == "canceled")
    print(f"Wrote {len(rows)} synthetic customers -> {CSV_PATH}")
    print(f"Churn rate: {churn / len(rows) * 100:.1f}%  ({churn}/{len(rows)})")
    print("Next step:  python predict.py pipeline")


if __name__ == "__main__":
    main()
