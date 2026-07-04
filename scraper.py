#!/usr/bin/env python3
"""
scraper.py
==========
Fetches all seeded customers and their Stripe data, then saves a flat
CSV to data/customers.csv.

Columns produced:
  customer_id, name, email, plan, industry, company_size, signup_source,
  account_age_days, logins_per_week, features_used, support_tickets,
  nps_score, subscription_status, payment_failure_count,
  invoice_count, total_spend_usd, last_invoice_days_ago, created_at
"""

import os
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import stripe
from rich.console import Console
from rich.progress import track

load_dotenv()

stripe.api_key = os.getenv("STRIPE_API_KEY")
if not stripe.api_key or not stripe.api_key.startswith("sk_test_"):
    raise SystemExit("❌  Set STRIPE_API_KEY=sk_test_... in .env")

HERE      = Path(__file__).parent
CSV_PATH  = HERE / "data" / "customers.csv"

console = Console()

CSV_HEADERS = [
    "customer_id", "name", "email", "plan",
    "industry", "company_size", "signup_source",
    "account_age_days", "logins_per_week", "features_used",
    "support_tickets", "nps_score",
    "subscription_status",   # active | canceled | none
    "payment_failure_count",
    "invoice_count",
    "total_spend_usd",
    "last_invoice_days_ago",
    "created_at",
]

NOW_TS = datetime.now(timezone.utc).timestamp()

def days_since(ts) -> int:
    """Convert a unix timestamp → days ago (int)."""
    if ts is None:
        return -1
    return max(0, int((NOW_TS - ts) / 86400))

def fetch_subscription_status(customer_id: str) -> str:
    """Return 'active', 'canceled', or 'none'."""
    subs = stripe.Subscription.list(customer=customer_id, limit=5, status="all")
    statuses = [s.status for s in subs.auto_paging_iter()]
    if not statuses:
        return "none"
    if "active" in statuses:
        return "active"
    if "canceled" in statuses:
        return "canceled"
    return statuses[0]

def fetch_invoice_stats(customer_id: str) -> tuple[int, float, int]:
    """
    Returns (invoice_count, total_spend_usd, last_invoice_days_ago).
    """
    invoices = stripe.Invoice.list(customer=customer_id, limit=100)
    count       = 0
    total_cents = 0
    latest_ts   = None
    for inv in invoices.auto_paging_iter():
        count += 1
        total_cents += inv.amount_paid or 0
        if latest_ts is None or (inv.status_transitions.finalized_at or 0) > latest_ts:
            latest_ts = inv.status_transitions.finalized_at
    return count, round(total_cents / 100, 2), days_since(latest_ts)

def fetch_failure_count(customer_id: str) -> int:
    """Count failed PaymentIntents for this customer."""
    pis = stripe.PaymentIntent.list(customer=customer_id, limit=100)
    return sum(1 for pi in pis.auto_paging_iter() if pi.status in ("requires_payment_method", "canceled"))

def scrape(max_customers: int = 5000) -> int:
    """
    Iterates all seeded Stripe customers, fetches their data, and writes CSV.
    Returns number of rows written.
    """
    console.rule("[bold]SaaS Churn Scraper — Stripe[/]")

    # Collect all seeded customers first (so we know total)
    console.print("Listing seeded customers…")
    customers = [
        c for c in stripe.Customer.list(limit=100).auto_paging_iter()
        if (lambda m: m["saas_churn_seed"] if "saas_churn_seed" in m else None)(c.metadata) == "true"
    ]
    console.print(f"  Found [bold]{len(customers)}[/] seeded customers\n")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()

        for i, c in enumerate(track(customers, description="Fetching customer data…")):
            meta = c.metadata or {}
            plan = (meta["plan"] if "plan" in meta else "Free")

            # Sub-resource fetches (with basic retry)
            for attempt in range(3):
                try:
                    sub_status        = fetch_subscription_status(c.id)
                    inv_count, spend, last_inv = fetch_invoice_stats(c.id)
                    fail_count        = fetch_failure_count(c.id)
                    break
                except stripe.error.RateLimitError:
                    time.sleep(2 ** attempt)
            else:
                sub_status, inv_count, spend, last_inv, fail_count = "unknown", 0, 0.0, -1, 0

            def mg(key, default=""):
                try:
                    return meta[key]
                except (KeyError, TypeError):
                    return default

            writer.writerow({
                "customer_id":          c.id,
                "name":                 c.name or "",
                "email":                c.email or "",
                "plan":                 plan,
                "industry":             mg("industry", "Unknown"),
                "company_size":         mg("company_size", "Unknown"),
                "signup_source":        mg("signup_source", "unknown"),
                "account_age_days":     mg("account_age_days", "0"),
                "logins_per_week":      mg("logins_per_week", "0"),
                "features_used":        mg("features_used", "0"),
                "support_tickets":      mg("support_tickets", "0"),
                "nps_score":            mg("nps_score", "5"),
                "subscription_status":  sub_status,
                "payment_failure_count": fail_count,
                "invoice_count":        inv_count,
                "total_spend_usd":      spend,
                "last_invoice_days_ago": last_inv,
                "created_at":           datetime.fromtimestamp(c.created).strftime("%Y-%m-%d"),
            })
            rows_written += 1

            # Polite rate limiting
            if i % 30 == 29:
                time.sleep(0.5)

    console.print(f"\n[green]✓[/] Saved {rows_written} rows → [bold]{CSV_PATH}[/]")
    console.print("[bold]Next step:[/]  python predict.py pipeline")
    return rows_written

if __name__ == "__main__":
    scrape()
