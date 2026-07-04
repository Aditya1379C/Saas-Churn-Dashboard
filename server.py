#!/usr/bin/env python3
"""
server.py
=========
Flask local server for the SaaS Churn Dashboard.

Routes:
  GET  /              — dashboard HTML
  GET  /api/data      — JSON payload for all charts
  POST /api/run       — trigger full pipeline subprocess
  GET  /api/status    — live pipeline status

Usage:
  python server.py
  python predict.py serve --port 8080
"""

import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask import Response as FlaskResponse

import model as mdl
import data_pipeline as dp

HERE        = Path(__file__).parent
DB_PATH     = HERE / "data" / "customers.db"
METRICS_PATH = HERE / "models" / "metrics.json"
IMPORTANCE_PATH = HERE / "models" / "feature_importance.json"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0   # no caching during dev

# ── Pipeline state ────────────────────────────────────────────────────────────

_pipeline_state: dict = {"running": False, "log": [], "exit_code": None}
_state_lock = threading.Lock()

def _run_pipeline_bg():
    with _state_lock:
        _pipeline_state.update({"running": True, "log": [], "exit_code": None})

    steps = [
        [sys.executable, "-u", str(HERE / "scraper.py")],
        [sys.executable, "-u", str(HERE / "data_pipeline.py")],
        [sys.executable, "-u", str(HERE / "model.py")],
    ]
    for cmd in steps:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            with _state_lock:
                _pipeline_state["log"].append(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            with _state_lock:
                _pipeline_state.update({"running": False, "exit_code": proc.returncode})
            return

    with _state_lock:
        _pipeline_state.update({"running": False, "exit_code": 0})

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_db() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM customers").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def _attach_churn_prob(rows: list[dict]) -> list[dict]:
    """Add model churn_prob to each row if model exists."""
    if not rows or not (HERE / "models" / "churn_model.pkl").exists():
        for r in rows:
            r["churn_prob"] = None
        return rows
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        probs = mdl.predict_churn_prob(df)
        for r, p in zip(rows, probs):
            r["churn_prob"] = round(float(p), 4)
    except Exception:
        for r in rows:
            r["churn_prob"] = None
    return rows

# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/data")
def api_data():
    rows = _load_db()
    rows = _attach_churn_prob(rows)

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())

    importance = {}
    if IMPORTANCE_PATH.exists():
        importance = json.loads(IMPORTANCE_PATH.read_text())

    # Summary KPIs
    total   = len(rows)
    churned = sum(1 for r in rows if r.get("churned") == 1)
    at_risk = sum(1 for r in rows
                  if r.get("churn_prob") is not None and r["churn_prob"] > 0.6
                  and r.get("churned") == 0)
    avg_eng = round(sum(r.get("engagement_score", 0) for r in rows) / max(total, 1), 1)

    # Churn by plan
    from collections import defaultdict
    plan_data: dict = defaultdict(lambda: {"total": 0, "churned": 0, "eng_sum": 0})
    for r in rows:
        p = r.get("plan", "Unknown")
        plan_data[p]["total"]   += 1
        plan_data[p]["churned"] += int(r.get("churned") == 1)
        plan_data[p]["eng_sum"] += float(r.get("engagement_score") or 0)

    churn_by_plan = [
        {
            "plan":      p,
            "total":     v["total"],
            "churned":   v["churned"],
            "churn_pct": round(v["churned"] / max(v["total"], 1) * 100, 1),
            "avg_eng":   round(v["eng_sum"] / max(v["total"], 1), 1),
        }
        for p, v in sorted(plan_data.items(),
                            key=lambda x: ["Free","Starter","Pro","Enterprise"].index(x[0])
                            if x[0] in ["Free","Starter","Pro","Enterprise"] else 99)
    ]

    # Churn by risk_flags count
    flag_data: dict = defaultdict(lambda: {"total": 0, "churned": 0})
    for r in rows:
        f = min(int(r.get("risk_flags") or 0), 5)
        flag_data[f]["total"]   += 1
        flag_data[f]["churned"] += int(r.get("churned") == 1)
    churn_by_flags = [
        {"flags": k, "total": v["total"],
         "churn_pct": round(v["churned"] / max(v["total"], 1) * 100, 1)}
        for k, v in sorted(flag_data.items())
    ]

    # Prob distribution histogram (20 bins)
    probs = [r["churn_prob"] for r in rows if r.get("churn_prob") is not None]
    hist  = [0] * 20
    for p in probs:
        idx = min(int(p * 20), 19)
        hist[idx] += 1

    # At-risk customers table (top 20, not yet churned)
    at_risk_table = sorted(
        [r for r in rows if r.get("churn_prob", 0) is not None
         and r.get("churn_prob", 0) > 0.5 and r.get("churned") == 0],
        key=lambda x: x.get("churn_prob", 0), reverse=True,
    )[:20]

    return jsonify({
        "kpis": {
            "total": total, "churned": churned, "at_risk": at_risk,
            "churn_pct": round(churned / max(total, 1) * 100, 1),
            "avg_engagement": avg_eng,
        },
        "metrics":        metrics,
        "importance":     importance,
        "churn_by_plan":  churn_by_plan,
        "churn_by_flags": churn_by_flags,
        "prob_histogram": hist,
        "at_risk_table":  at_risk_table,
        "has_model":      (HERE / "models" / "churn_model.pkl").exists(),
    })

@app.route("/api/run", methods=["POST"])
def api_run():
    with _state_lock:
        if _pipeline_state["running"]:
            return jsonify({"status": "already_running"}), 409
    t = threading.Thread(target=_run_pipeline_bg, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/status")
def api_status():
    with _state_lock:
        return jsonify({
            "running":   _pipeline_state["running"],
            "log":       _pipeline_state["log"][-50:],  # last 50 lines
            "exit_code": _pipeline_state["exit_code"],
        })

@app.route("/")
def index():
    resp = FlaskResponse(render_template("dashboard.html"))
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp

# ── Main ──────────────────────────────────────────────────────────────────────

def serve(host: str = "127.0.0.1", port: int = 8080, debug: bool = False):
    print(f"\n  SaaS Churn Dashboard → http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    serve()
