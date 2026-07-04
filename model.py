#!/usr/bin/env python3
"""
model.py
========
Trains a Random Forest churn classifier on the cleaned customer DB.
Compares against XGBoost and keeps the winner.

Outputs:
  models/churn_model.pkl         — winning model
  models/encoders.pkl            — feature column list + label mappings
  models/feature_importance.json — {feature: importance} sorted desc
  models/metrics.json            — accuracy, AUC, precision, recall, F1
"""

import json
import logging
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HERE            = Path(__file__).parent
DB_PATH         = HERE / "data" / "customers.db"
MODEL_PATH      = HERE / "models" / "churn_model.pkl"
ENCODERS_PATH   = HERE / "models" / "encoders.pkl"
IMPORTANCE_PATH = HERE / "models" / "feature_importance.json"
METRICS_PATH    = HERE / "models" / "metrics.json"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

LOG_PATH = HERE / "logs" / "model.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
)
log = logging.getLogger(__name__)

# Features the model uses
FEATURE_COLS = [
    "plan_tier",
    "account_age_days",
    "logins_per_week",
    "features_used",
    "support_tickets",
    "nps_score",
    "engagement_score",
    "failure_rate",
    "days_since_invoice",
    "avg_monthly_spend",
    "risk_flags",
    "invoice_count",
    "payment_failure_count",
]

MIN_ROWS = 50  # minimum to attempt training

# ── Model cache (avoid re-loading on every predict call) ─────────────────────

_CACHE: dict = {}

def load_model(force_reload: bool = False):
    """
    Returns (model, feature_cols, metrics) from disk, using mtime-based cache.
    """
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
    if not force_reload and _CACHE.get("mtime") == mtime:
        return _CACHE["model"], _CACHE["feature_cols"], _CACHE["metrics"]

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} — run 'python predict.py train' first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    _CACHE.update({"mtime": mtime, "model": model,
                   "feature_cols": encoders["feature_cols"],
                   "threshold": encoders.get("threshold", 0.5), "metrics": metrics})
    return model, encoders["feature_cols"], metrics

# ── Feature matrix ────────────────────────────────────────────────────────────

def build_X(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df[feature_cols].fillna(0).to_numpy(dtype=float)


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability cutoff in [0.1, 0.9] that maximises F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2)

# ── Training ──────────────────────────────────────────────────────────────────

def train() -> dict:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} — run 'python predict.py pipeline' first.")

    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    log.info("Loaded %d rows from DB", len(df))

    if len(df) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows; got {len(df)}")

    # Drop rows where churn label is missing
    df = df.dropna(subset=["churned"])
    df["churned"] = df["churned"].astype(int)

    X = build_X(df, FEATURE_COLS)
    y = df["churned"].values

    # Split BEFORE any fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    log.info("Train: %d  Test: %d  Churn rate: %.1f%%",
             len(y_train), len(y_test), y.mean() * 100)

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    log.info("Random Forest AUC: %.4f", rf_auc)

    # ── XGBoost (if available) ──
    best_model = rf
    best_auc   = rf_auc
    best_name  = "RandomForest"

    if HAS_XGB:
        xg = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            eval_metric="auc", n_jobs=1, random_state=42,
            verbosity=0,
        )
        xg.fit(X_train, y_train)
        xg_auc = roc_auc_score(y_test, xg.predict_proba(X_test)[:, 1])
        log.info("XGBoost AUC: %.4f", xg_auc)
        if xg_auc > rf_auc:
            best_model = xg
            best_auc   = xg_auc
            best_name  = "XGBoost"

    log.info("Winner: %s (AUC %.4f)", best_name, best_auc)

    # ── Tune the decision threshold for churn recall ──
    # Churn is recall-sensitive: a missed churner is a lost customer, while a
    # false alarm just means an unnecessary retention nudge. The default 0.5
    # cutoff under-predicts churn. We pick the threshold that maximises F1 on
    # out-of-fold training predictions (never the test set, to avoid optimism).
    oof_prob = cross_val_predict(
        best_model, X_train, y_train, cv=5,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    threshold = _best_f1_threshold(y_train, oof_prob)
    log.info("Tuned decision threshold: %.2f (default 0.50)", threshold)

    # ── Final metrics on test set, at the tuned threshold ──
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cv_scores = cross_val_score(best_model, X_train, y_train,
                                cv=5, scoring="roc_auc", n_jobs=1)

    metrics = {
        "model":     best_name,
        "n_samples": len(df),
        "n_train":   len(y_train),
        "n_test":    len(y_test),
        "churn_rate_pct": round(float(y.mean()) * 100, 2),
        "decision_threshold": threshold,
        "auc_roc":   round(float(best_auc), 4),
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std":  round(float(cv_scores.std()), 4),
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    # ── Feature importance ──
    importance = dict(zip(
        FEATURE_COLS,
        [round(float(v), 6) for v in best_model.feature_importances_],
    ))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ── Save artefacts ──
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump({"feature_cols": FEATURE_COLS, "threshold": threshold}, f)
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    log.info("Saved model → %s", MODEL_PATH)

    # Console summary
    print(f"\n{'─'*50}")
    print(f"  Winner      : {best_name}")
    print(f"  AUC-ROC     : {metrics['auc_roc']}  (CV {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']})")
    print(f"  Accuracy    : {metrics['accuracy']}")
    print(f"  Precision   : {metrics['precision']}   Recall: {metrics['recall']}   F1: {metrics['f1']}")
    print(f"  Churn rate  : {metrics['churn_rate_pct']}%  ({int(y.sum())}/{len(y)} customers)")
    print(f"\n  Top features:")
    for feat, imp in list(importance.items())[:5]:
        bar = "█" * int(imp * 100)
        print(f"    {feat:<25} {imp:.4f}  {bar}")
    print(f"{'─'*50}")
    print(f"\n✓  Model saved → {MODEL_PATH}")
    print("Next step:  python predict.py serve")

    return metrics

# ── Inference ─────────────────────────────────────────────────────────────────

def predict_churn_prob(df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame with the required feature columns,
    returns an array of churn probabilities (0–1).
    """
    model, feature_cols, _ = load_model()
    X = build_X(df, feature_cols)
    return model.predict_proba(X)[:, 1]


def predict_churn_label(df: pd.DataFrame) -> np.ndarray:
    """Churn predictions (0/1) using the tuned decision threshold."""
    prob = predict_churn_prob(df)
    threshold = _CACHE.get("threshold", 0.5)
    return (prob >= threshold).astype(int)

if __name__ == "__main__":
    train()
