#!/usr/bin/env python3
"""
predict.py — SaaS Churn Dashboard CLI
======================================
Commands:
  generate  Generate synthetic data locally (no Stripe) → data/customers.csv
  seed      Create synthetic customers in Stripe test mode
  scrape    Fetch customer data from Stripe API → data/customers.csv
  pipeline  Clean + engineer features → data/customers.db
  train     Train churn model → models/churn_model.pkl
  serve     Start Flask dashboard server
  run       End-to-end (local): generate → pipeline → train

Usage:
  python predict.py --help
  python predict.py generate -n 520
  python predict.py pipeline
  python predict.py train
  python predict.py serve --port 8080
  python predict.py run                 # no API key needed
  python predict.py run --source stripe # uses Stripe seed + scrape instead
"""

import typer
from rich.console import Console

app     = typer.Typer(add_completion=False, help=__doc__)
console = Console()

@app.command()
def generate(
    count: int = typer.Option(520, "--count", "-n", help="Number of customers to generate"),
):
    """Generate synthetic customer data locally, no Stripe key required."""
    import os
    os.environ["SEED_COUNT"] = str(count)
    from generate_data import main
    main()

@app.command()
def seed(
    count: int = typer.Option(520, "--count", "-n", help="Number of customers to seed"),
):
    """Create synthetic customers in Stripe test mode."""
    import os
    os.environ.setdefault("SEED_COUNT", str(count))
    from setup_stripe_data import main
    main()

@app.command()
def scrape():
    """Fetch customer data from Stripe API and save to data/customers.csv."""
    from scraper import scrape as _scrape
    n = _scrape()
    console.print(f"[green]✓[/] {n} rows saved")

@app.command()
def pipeline():
    """Clean raw CSV and engineer churn features → data/customers.db."""
    from data_pipeline import run
    run()

@app.command()
def train():
    """Train Random Forest / XGBoost churn classifier and save model artifacts."""
    from model import train as _train
    _train()

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8080, "--port", "-p", help="Port"),
    debug: bool = typer.Option(False, "--debug", help="Flask debug mode"),
):
    """Start the Flask dashboard server."""
    from server import serve as _serve
    _serve(host=host, port=port, debug=debug)

@app.command()
def run(
    source: str = typer.Option(
        "local", "--source",
        help="Data source: 'local' (no Stripe) or 'stripe' (seed + scrape)",
    ),
):
    """Run full pipeline end-to-end: generate/scrape → pipeline → train."""
    if source == "stripe":
        console.rule("[bold cyan]Step 1/3 — Scrape (Stripe)[/]")
        from scraper import scrape as _scrape
        _scrape()
    else:
        console.rule("[bold cyan]Step 1/3 — Generate (local)[/]")
        from generate_data import main as _generate
        _generate()

    console.rule("[bold cyan]Step 2/3 — Pipeline[/]")
    from data_pipeline import run as _run
    _run()

    console.rule("[bold cyan]Step 3/3 — Train[/]")
    from model import train as _train
    _train()

    console.rule("[bold green]Done[/]")
    console.print("Start the dashboard:  [bold]python predict.py serve[/]")

if __name__ == "__main__":
    app()
