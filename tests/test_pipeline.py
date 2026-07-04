"""
tests/test_pipeline.py
======================
Unit tests for data_pipeline.py and model.py.
Run with:  pytest tests/ -v   or   make test
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_pipeline as dp
import model as mdl

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_raw_df(n: int = 30, plan: str | None = None) -> pd.DataFrame:
    """Minimal raw DataFrame mimicking scraper.py CSV output."""
    plans = ["Free", "Starter", "Pro", "Enterprise"]
    return pd.DataFrame({
        "customer_id":          [f"cus_{i:04d}" for i in range(n)],
        "name":                 [f"User {i}" for i in range(n)],
        "email":                [f"user{i}@test.com" for i in range(n)],
        "plan":                 [plan or plans[i % 4] for i in range(n)],
        "industry":             ["SaaS"] * n,
        "company_size":         ["1-10"] * n,
        "signup_source":        ["organic"] * n,
        "account_age_days":     [str(90 + i * 10) for i in range(n)],
        "logins_per_week":      [str(i % 10) for i in range(n)],
        "features_used":        [str(i % 8) for i in range(n)],
        "support_tickets":      [str(i % 5) for i in range(n)],
        "nps_score":            [str(5 + i % 6) for i in range(n)],
        "subscription_status":  ["active" if i % 4 != 0 else "canceled" for i in range(n)],
        "payment_failure_count":[str(i % 3) for i in range(n)],
        "invoice_count":        [str(i % 12 + 1) for i in range(n)],
        "total_spend_usd":      [str(float(i * 29)) for i in range(n)],
        "last_invoice_days_ago":[str(i * 5) for i in range(n)],
        "created_at":           ["2024-01-15"] * n,
    })

# ── clean() ───────────────────────────────────────────────────────────────────

class TestClean:
    def test_returns_dataframe(self):
        df = dp.clean(make_raw_df())
        assert isinstance(df, pd.DataFrame)

    def test_row_count_preserved(self):
        raw = make_raw_df(20)
        assert len(dp.clean(raw)) == 20

    def test_numeric_columns_are_int(self):
        df = dp.clean(make_raw_df())
        for col in ["logins_per_week", "features_used", "support_tickets",
                    "nps_score", "invoice_count", "account_age_days"]:
            assert pd.api.types.is_integer_dtype(df[col]), f"{col} should be int"

    def test_nps_clamped_0_10(self):
        raw = make_raw_df()
        raw["nps_score"] = "999"
        df = dp.clean(raw)
        assert df["nps_score"].max() <= 10

    def test_logins_clamped_0_50(self):
        raw = make_raw_df()
        raw["logins_per_week"] = "-5"
        df = dp.clean(raw)
        assert df["logins_per_week"].min() >= 0

    def test_plan_unknown_defaults_to_free(self):
        raw = make_raw_df(plan="GoldPremiumUltra")
        df  = dp.clean(raw)
        assert (df["plan"] == "Free").all()

    def test_handles_empty_strings(self):
        raw = make_raw_df(5)
        raw.loc[0, "logins_per_week"] = ""
        raw.loc[1, "nps_score"]       = ""
        df = dp.clean(raw)
        assert not df["logins_per_week"].isna().any()
        assert not df["nps_score"].isna().any()

    def test_handles_float_strings(self):
        raw = make_raw_df(5)
        raw["logins_per_week"] = "3.7"
        df = dp.clean(raw)
        assert df["logins_per_week"].iloc[0] == 3

    def test_subscription_status_lowercased(self):
        raw = make_raw_df(5)
        raw["subscription_status"] = "CANCELED"
        df = dp.clean(raw)
        assert (df["subscription_status"] == "canceled").all()

# ── build_features() ─────────────────────────────────────────────────────────

class TestBuildFeatures:
    @pytest.fixture
    def cleaned(self):
        return dp.clean(make_raw_df())

    def test_plan_tier_range(self, cleaned):
        df = dp.build_features(cleaned)
        assert df["plan_tier"].between(0, 3).all()

    def test_engagement_score_range(self, cleaned):
        df = dp.build_features(cleaned)
        assert df["engagement_score"].between(0, 100).all()

    def test_failure_rate_non_negative(self, cleaned):
        df = dp.build_features(cleaned)
        assert (df["failure_rate"] >= 0).all()

    def test_risk_flags_range(self, cleaned):
        df = dp.build_features(cleaned)
        assert df["risk_flags"].between(0, 5).all()

    def test_churned_column_binary(self, cleaned):
        df = dp.build_features(cleaned)
        assert set(df["churned"].unique()).issubset({0, 1})

    def test_canceled_is_churned(self):
        raw = make_raw_df(10)
        raw["subscription_status"] = "canceled"
        df  = dp.build_features(dp.clean(raw))
        assert (df["churned"] == 1).all()

    def test_active_not_churned(self):
        raw = make_raw_df(10)
        raw["subscription_status"] = "active"
        raw["plan"] = "Pro"   # non-free so engagement churn rule doesn't apply
        raw["logins_per_week"] = "10"
        raw["support_tickets"] = "0"
        df  = dp.build_features(dp.clean(raw))
        assert (df["churned"] == 0).all()

    def test_avg_monthly_spend_non_negative(self, cleaned):
        df = dp.build_features(cleaned)
        assert (df["avg_monthly_spend"] >= 0).all()

    def test_days_since_invoice_capped(self, cleaned):
        df = dp.build_features(cleaned)
        assert (df["days_since_invoice"] <= 365).all()

# ── model.build_X() ───────────────────────────────────────────────────────────

class TestBuildX:
    def test_shape(self):
        df = dp.build_features(dp.clean(make_raw_df(20)))
        X  = mdl.build_X(df, mdl.FEATURE_COLS)
        assert X.shape == (20, len(mdl.FEATURE_COLS))

    def test_dtype_float(self):
        df = dp.build_features(dp.clean(make_raw_df(10)))
        X  = mdl.build_X(df, mdl.FEATURE_COLS)
        assert X.dtype == float

    def test_no_nans(self):
        df = dp.build_features(dp.clean(make_raw_df(10)))
        X  = mdl.build_X(df, mdl.FEATURE_COLS)
        assert not np.isnan(X).any()

    def test_missing_col_raises(self):
        df = dp.build_features(dp.clean(make_raw_df(5)))
        with pytest.raises(ValueError):
            mdl.build_X(df, mdl.FEATURE_COLS + ["nonexistent_col"])

# ── validate() ────────────────────────────────────────────────────────────────

class TestValidate:
    def test_no_exception_on_clean_data(self):
        df = dp.build_features(dp.clean(make_raw_df()))
        dp.validate(df)  # should not raise

    def test_warns_on_duplicate_ids(self, capsys):
        df = dp.build_features(dp.clean(make_raw_df(4)))
        df2 = pd.concat([df, df], ignore_index=True)
        dp.validate(df2)  # logs a warning — should not raise
