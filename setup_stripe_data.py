#!/usr/bin/env python3
"""
setup_stripe_data.py
====================
Seeds Stripe TEST MODE with synthetic SaaS customers, subscriptions,
and simulated churn signals (payment failures, cancellations).

Run ONCE before scraping:
    python setup_stripe_data.py

What it creates in Stripe test mode:
  - 4 Products  (Free, Starter, Pro, Enterprise)
  - 4 Prices    (one per product, monthly)
  - SEED_COUNT  Customers with varied metadata
  - 1 Subscription per customer
  - Cancelled subscriptions for churned customers
  - Failed PaymentIntents for at-risk customers

Prerequisites:
  - pip install stripe python-dotenv
  - .env with STRIPE_API_KEY=sk_test_...
"""

import os
import time
import random
import string
from datetime import datetime, timedelta
from dotenv import load_dotenv
import stripe
from rich.console import Console
from rich.progress import track

load_dotenv()

stripe.api_key = os.getenv("STRIPE_API_KEY")
if not stripe.api_key or not stripe.api_key.startswith("sk_test_"):
    raise SystemExit("❌  Set STRIPE_API_KEY=sk_test_... in .env")

SEED_COUNT = int(os.getenv("SEED_COUNT", "520"))
console    = Console()

random.seed(42)

# ── Plan definitions ──────────────────────────────────────────────────────────

PLANS = {
    "Free":       {"amount": 0,     "weight": 0.35},
    "Starter":    {"amount": 2900,  "weight": 0.30},  # $29/mo in cents
    "Pro":        {"amount": 9900,  "weight": 0.25},  # $99/mo
    "Enterprise": {"amount": 29900, "weight": 0.10},  # $299/mo
}

PLAN_NAMES  = list(PLANS.keys())
PLAN_WEIGHTS = [PLANS[p]["weight"] for p in PLAN_NAMES]

# Realistic churn rate per plan
CHURN_RATES = {"Free": 0.42, "Starter": 0.22, "Pro": 0.10, "Enterprise": 0.06}

# Payment failure probability (for non-churned at-risk users)
FAILURE_RATES = {"Free": 0.25, "Starter": 0.12, "Pro": 0.05, "Enterprise": 0.02}

INDUSTRIES = ["E-commerce", "FinTech", "HealthTech", "EdTech",
              "MarTech", "HRTech", "LegalTech", "PropTech", "SaaS", "AgriTech"]
SOURCES    = ["organic", "referral", "paid_search", "social", "product_hunt", "content"]
SIZES      = ["1-10", "11-50", "51-200", "201-1000", "1000+"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def meta_get(metadata, key: str, default=None):
    """Safe metadata access for both old (dict) and new (StripeObject) SDK."""
    try:
        return metadata[key]
    except (KeyError, TypeError):
        return default

def rand_email(name: str) -> str:
    domain = random.choice(["gmail.com", "company.io", "startup.co",
                             "corp.com", "tech.org", "venture.ai"])
    suffix = "".join(random.choices(string.digits, k=3))
    slug = name.lower().replace(" ", ".") + suffix
    return f"{slug}@{domain}"

def days_ago(n: int) -> int:
    """Return unix timestamp for n days ago."""
    return int((datetime.utcnow() - timedelta(days=n)).timestamp())

def weighted_choice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    cum = 0
    for c, w in zip(choices, weights):
        cum += w
        if r <= cum:
            return c
    return choices[-1]

# ── Step 1: Create Products & Prices ─────────────────────────────────────────

def ensure_products_and_prices() -> dict[str, str]:
    """
    Returns {plan_name: price_id}.
    Reuses existing products/prices if they already have matching metadata.
    """
    console.print("[bold cyan]Setting up Stripe Products & Prices…[/]")
    price_ids: dict[str, str] = {}

    # Check for existing seeded products
    existing = stripe.Product.list(limit=100, active=True)
    seeded   = {meta_get(p.metadata, "saas_plan"): p for p in existing.auto_paging_iter()
                if meta_get(p.metadata, "saas_churn_seed") == "true"}

    for plan_name, cfg in PLANS.items():
        if plan_name in seeded:
            product = seeded[plan_name]
            console.print(f"  ✓ Product '{plan_name}' already exists ({product.id})")
        else:
            product = stripe.Product.create(
                name=f"SaaS {plan_name}",
                metadata={"saas_churn_seed": "true", "saas_plan": plan_name},
            )
            console.print(f"  + Created product '{plan_name}' ({product.id})")

        # Find existing price for this product
        prices = stripe.Price.list(product=product.id, active=True, limit=5)
        price  = next(
            (p for p in prices.auto_paging_iter()
             if p.recurring and p.recurring.interval == "month"),
            None,
        )
        if price:
            price_ids[plan_name] = price.id
            console.print(f"    ✓ Price exists ({price.id})")
        else:
            price = stripe.Price.create(
                product=product.id,
                unit_amount=cfg["amount"],
                currency="usd",
                recurring={"interval": "month"},
            )
            price_ids[plan_name] = price.id
            console.print(f"    + Created price ({price.id})")

    return price_ids

# ── Step 2: Create Customers + Subscriptions ──────────────────────────────────

FIRST_NAMES = [
    "Liam","Noah","Oliver","Elijah","James","Aiden","Lucas","Mason","Ethan","Logan",
    "Emma","Olivia","Ava","Sophia","Isabella","Mia","Charlotte","Amelia","Harper","Evelyn",
    "Rohan","Priya","Arjun","Sneha","Dev","Ananya","Vikram","Meera","Rahul","Pooja",
    "Mohammed","Fatima","Omar","Layla","Yusuf","Aisha","Hassan","Zara","Bilal","Nour",
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Wilson","Moore",
    "Sharma","Patel","Singh","Kumar","Gupta","Mehta","Joshi","Rao","Nair","Iyer",
    "Ali","Khan","Ahmed","Hassan","Hussein","Malik","Sheikh","Siddiqui","Chaudhry","Rahman",
]

def random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def make_customer_and_subscription(price_ids: dict[str, str], idx: int):
    plan       = weighted_choice(PLAN_NAMES, PLAN_WEIGHTS)
    churned    = random.random() < CHURN_RATES[plan]
    account_age = random.randint(30, 730)  # 1 month – 2 years
    name       = random_name()

    # Engagement proxies stored as metadata (Stripe has no native login tracking)
    logins_pw   = max(0, int(random.gauss({"Free":2,"Starter":5,"Pro":10,"Enterprise":15}[plan], 3)))
    features    = max(0, int(random.gauss({"Free":2,"Starter":5,"Pro":9,"Enterprise":14}[plan], 3)))
    tickets     = max(0, int(random.gauss({"Free":1.5,"Starter":0.8,"Pro":0.4,"Enterprise":0.2}[plan], 1)))
    nps         = max(0, min(10, int(random.gauss({"Free":5,"Starter":6,"Pro":7,"Enterprise":8}[plan], 2))))

    customer = stripe.Customer.create(
        name=name,
        email=rand_email(name),
        metadata={
            "saas_churn_seed":   "true",
            "plan":              plan,
            "industry":          random.choice(INDUSTRIES),
            "company_size":      random.choice(SIZES),
            "signup_source":     random.choice(SOURCES),
            "account_age_days":  str(account_age),
            "logins_per_week":   str(logins_pw),
            "features_used":     str(features),
            "support_tickets":   str(tickets),
            "nps_score":         str(nps),
            "seed_index":        str(idx),
        },
    )

    # Free plan: no subscription object (common SaaS pattern)
    if plan == "Free":
        return customer, plan, churned, False

    # Attach a test payment method so Stripe accepts the subscription
    pm = stripe.PaymentMethod.attach("pm_card_visa", customer=customer.id)
    stripe.Customer.modify(
        customer.id,
        invoice_settings={"default_payment_method": pm.id},
    )

    # Create subscription
    sub_kwargs: dict = {
        "customer": customer.id,
        "items":    [{"price": price_ids[plan]}],
        "metadata": {"saas_churn_seed": "true"},
    }
    if churned:
        # Create then immediately cancel
        sub = stripe.Subscription.create(**sub_kwargs, cancel_at_period_end=False)
        stripe.Subscription.cancel(sub.id)
        return customer, plan, True, False
    else:
        sub = stripe.Subscription.create(**sub_kwargs)

    # Simulate payment failure for at-risk non-churned customers
    failed = False
    if random.random() < FAILURE_RATES[plan]:
        try:
            stripe.PaymentIntent.create(
                amount=PLANS[plan]["amount"],
                currency="usd",
                customer=customer.id,
                payment_method="pm_card_chargeDeclined",
                payment_method_types=["card"],
                confirm=True,
                metadata={"saas_churn_seed": "true"},
            )
        except stripe.error.CardError:
            failed = True  # expected — decline is the point

    return customer, plan, False, failed

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold]SaaS Churn Dashboard — Stripe Seeder[/]")
    console.print(f"Target: [bold]{SEED_COUNT}[/] customers in test mode\n")

    # Check existing seeded customers
    existing_count = sum(
        1 for c in stripe.Customer.list(limit=100).auto_paging_iter()
        if meta_get(c.metadata, "saas_churn_seed") == "true"
    )
    if existing_count >= SEED_COUNT:
        console.print(
            f"[yellow]⚠ {existing_count} seeded customers already exist. "
            "Delete them in the Stripe dashboard or run with a fresh test key.[/]"
        )
        return

    price_ids = ensure_products_and_prices()
    console.print()

    stats = {"total": 0, "churned": 0, "failed_payment": 0}
    plan_counts: dict[str, int] = {p: 0 for p in PLAN_NAMES}

    needed = SEED_COUNT - existing_count
    console.print(f"Creating [bold]{needed}[/] new customers…\n")

    for i in track(range(needed), description="Seeding Stripe…"):
        try:
            _, plan, churned, failed = make_customer_and_subscription(price_ids, existing_count + i)
            stats["total"]          += 1
            plan_counts[plan]       += 1
            if churned:
                stats["churned"]         += 1
            if failed:
                stats["failed_payment"]  += 1
            # Stripe rate-limit: ~100 req/s in test mode; small pause every 50
            if i % 50 == 49:
                time.sleep(0.5)
        except stripe.error.StripeError as e:
            console.print(f"[red]Stripe error on customer {i}: {e}[/]")

    console.print()
    console.rule("[bold green]Seeding Complete[/]")
    console.print(f"  Customers created : {stats['total']}")
    console.print(f"  Churned (cancelled): {stats['churned']} "
                  f"({stats['churned']*100//max(stats['total'],1)}%)")
    console.print(f"  Payment failures  : {stats['failed_payment']}")
    console.print(f"  Plan breakdown    : {plan_counts}")
    console.print("\n[bold]Next step:[/]  python predict.py scrape")

if __name__ == "__main__":
    main()
