#!/usr/bin/env python3
"""
data_pipeline.py
================
Reads data/customers.csv, cleans it, engineers churn features,
and saves to data/customers.db (table: customers).

Churn label definition:
  churned = 1  if subscription_status == 'canceled'
  churned = 0  otherwise

Engineered features:
  plan_tier          — ordinal: Free=0, Starter=1, Pro=2, Enterprise=3
  engagement_score   — composite of logins, features, NPS  (0-100)
  failure_rate       — payment_failure_count / max(invoice_count, 1)
  days_since_invoice — last_invoice_days_ago (capped at 365)
  risk_flags         — count of individual risk signals
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

HERE     = Path(__file__).parent
CSV_PATH = HERE / "data" / "customers.csv"
DB_PATH  = HERE / "data" / "customers.db"
LOG_PATH = HERE / "logs" / "pipeline.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH),
    ],
)
log = logging.getLogger(__name__)

PLAN_TIER = {"Free": 0, "Starter": 1, "Pro": 2, "Enterprise": 3}

# ── Cleaning ──────────────────────────────────────────────────────────────────

def _coerce_int(series: pd.Series, default: int = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(int)

def _coerce_float(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, normalise, and impute raw CSV columns."""
    log.info("Cleaning %d rows…", len(df))

    df = df.copy()

    # Numeric columns
    df["account_age_days"]      = _coerce_int(df["account_age_days"],   default=90)
    df["logins_per_week"]       = _coerce_int(df["logins_per_week"],     default=0)
    df["features_used"]         = _coerce_int(df["features_used"],       default=0)
    df["support_tickets"]       = _coerce_int(df["support_tickets"],     default=0)
    df["nps_score"]             = _coerce_int(df["nps_score"],           default=5)
    df["payment_failure_count"] = _coerce_int(df["payment_failure_count"], default=0)
    df["invoice_count"]         = _coerce_int(df["invoice_count"],       default=0)
    df["total_spend_usd"]       = _coerce_float(df["total_spend_usd"],   default=0.0)
    df["last_invoice_days_ago"] = _coerce_int(df["last_invoice_days_ago"], default=-1)

    # Clamp
    df["nps_score"]          = df["nps_score"].clip(0, 10)
    df["logins_per_week"]    = df["logins_per_week"].clip(0, 50)
    df["features_used"]      = df["features_used"].clip(0, 30)
    df["support_tickets"]    = df["support_tickets"].clip(0, 20)
    df["account_age_days"]   = df["account_age_days"].clip(1, 1095)

    # Normalise plan names
    df["plan"] = df["plan"].str.strip().str.title()
    df["plan"] = df["plan"].where(df["plan"].isin(PLAN_TIER), other="Free")

    # Normalise subscription_status
    df["subscription_status"] = (
        df["subscription_status"]
        .str.strip().str.lower()
        .fillna("none")
    )

    # Free plan customers have no subscription object → assign "none"
    mask_free = df["plan"] == "Free"
    df.loc[mask_free, "subscription_status"] = df.loc[mask_free, "subscription_status"].replace("", "none")

    # created_at
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d")

    log.info("  Cleaned — %d rows remain", len(df))
    return df

# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns used by the model."""
    log.info("Engineering features…")
    df = df.copy()

    # Ordinal plan tier
    df["plan_tier"] = df["plan"].map(PLAN_TIER).fillna(0).astype(int)

    # Engagement score: weighted composite (0–100)
    #   logins_per_week → normalise to 0-20 scale (cap 20 → 1.0)
    #   features_used   → normalise to 0-15 (cap 15)
    #   nps_score       → 0-10
    login_norm = (df["logins_per_week"].clip(0, 20) / 20 * 40)     # max 40 pts
    feat_norm  = (df["features_used"].clip(0, 15) / 15 * 40)       # max 40 pts
    nps_norm   = (df["nps_score"] / 10 * 20)                       # max 20 pts
    df["engagement_score"] = (login_norm + feat_norm + nps_norm).round(2)

    # Payment failure rate
    df["failure_rate"] = (
        df["payment_failure_count"] / df["invoice_count"].clip(lower=1)
    ).round(4)

    # Days since last invoice (cap negatives / unknowns at 365)
    df["days_since_invoice"] = df["last_invoice_days_ago"].clip(-1, 365)
    df.loc[df["days_since_invoice"] < 0, "days_since_invoice"] = 180  # impute unknown

    # Average monthly spend
    df["avg_monthly_spend"] = (
        df["total_spend_usd"] / (df["account_age_days"] / 30).clip(lower=1)
    ).round(2)

    # Risk flag count (binary signals)
    df["risk_flags"] = (
        (df["logins_per_week"] <= 1).astype(int)          # very low activity
        + (df["support_tickets"] >= 3).astype(int)        # high support load
        + (df["payment_failure_count"] >= 1).astype(int)  # any payment failure
        + (df["nps_score"] <= 4).astype(int)              # detractor
        + (df["features_used"] <= 1).astype(int)          # barely using product
    )

    # Churn label — single source of truth: a canceled subscription.
    # NOTE: We deliberately do NOT derive the label from engagement_score or
    # support_tickets here. Doing so for Free-plan users (the previous behaviour)
    # made the label a deterministic function of features the model also sees,
    # which leaked and inflated AUC. churn is now an independent ground-truth
    # outcome for every plan. See MODEL_ISSUES_AND_FIXES.md.
    df["churned"] = (df["subscription_status"] == "canceled").astype(int)

    log.info(
        "  Churn rate: %.1f%%  (%d/%d)",
        df["churned"].mean() * 100,
        df["churned"].sum(),
        len(df),
    )
    return df

# ── Validation ────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    issues = []
    if df["customer_id"].duplicated().any():
        issues.append("Duplicate customer_ids detected")
    if df["churned"].isna().any():
        issues.append("Null values in 'churned' column")
    if not set(df["plan"].unique()).issubset(set(PLAN_TIER.keys())):
        issues.append(f"Unexpected plan values: {df['plan'].unique()}")
    if issues:
        for msg in issues:
            log.warning("VALIDATION: %s", msg)
    else:
        log.info("  Validation passed ✓")

# ── Save to SQLite ────────────────────────────────────────────────────────────

def save_to_db(df: pd.DataFrame) -> None:
    """
    Incremental upsert: delete matching customer_ids then insert.
    Handles fresh DB, migration, and incremental runs.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    existing_cols: list[str] = []
    try:
        existing_cols = [
            row[1] for row in conn.execute("PRAGMA table_info(customers)").fetchall()
        ]
    except Exception:
        pass

    if not existing_cols:
        # Fresh DB
        df.to_sql("customers", conn, if_exists="replace", index=False)
        log.info("  Created new table 'customers' (%d rows)", len(df))
    else:
        new_cols = [c for c in df.columns if c not in existing_cols]
        for col in new_cols:
            conn.execute(f'ALTER TABLE customers ADD COLUMN "{col}" TEXT')
            log.info("  Migrated: added column '%s'", col)
        # Delete then re-insert changed rows
        ids = df["customer_id"].tolist()
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM customers WHERE customer_id IN ({placeholders})", ids)
        df.to_sql("customers", conn, if_exists="append", index=False)
        log.info("  Upserted %d rows into existing table", len(df))

    conn.commit()
    conn.close()
    log.info("  Saved → %s", DB_PATH)

# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found — run 'python predict.py scrape' first."
        )

    df_raw = pd.read_csv(CSV_PATH, dtype=str)
    log.info("Loaded %d rows from %s", len(df_raw), CSV_PATH)

    df = clean(df_raw)
    df = build_features(df)
    validate(df)
    save_to_db(df)

    log.info("Pipeline complete — %d customers ready for modelling", len(df))
    print(f"\n✓  Pipeline complete: {len(df)} customers → {DB_PATH}")
    print("Next step:  python predict.py train")
    return df

if __name__ == "__main__":
    run()
