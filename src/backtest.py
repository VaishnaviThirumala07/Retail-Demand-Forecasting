"""
Rolling evaluation over earlier 28-day windows, which do not overlap the final test window.

Window 0 is the final test window from the main pipeline. Windows 1 and 2 end 28 and 56 days
earlier. Each is run in its own process (--window) so memory is returned between runs, and each
is retrained using only data before it. --summarize then combines all three.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from baselines import run_all_baselines
from method_names import DISPLAY_METHODS
from metrics import bottom_level, evaluate_window, open_warehouse
from reconcile import format_lgbm_forecasts, reconcile_window
from train_lgbm import fit_window, generate_forecasts

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FORECASTS_DIR = ROOT / "data" / "forecasts"
REPORTS = ROOT / "reports"
HORIZON = 28

BIAS_VARIANTS = ["Bias-corrected MinTrace (global)", "Bias-corrected MinTrace (per dept)"]

app = typer.Typer()


def daily_totals(recon: pd.DataFrame, actuals: pd.Series) -> pd.DataFrame:
    """State-level daily forecast per method and actual sales, for sizing safety stock."""
    methods = [m for m in DISPLAY_METHODS if m in recon.columns]
    totals = bottom_level(recon).groupby("ds")[methods].sum().rename(columns=DISPLAY_METHODS)
    totals["actual"] = actuals
    return totals.reset_index()


def run_window(window: int, state: str, history_days: int) -> None:
    con = open_warehouse(WAREHOUSE)
    latest = pd.Timestamp(con.execute("SELECT MAX(sale_date) FROM raw_sales_enriched").fetchone()[0])
    con.close()
    test_end = latest - pd.Timedelta(days=HORIZON * window)
    test_start = test_end - pd.Timedelta(days=HORIZON - 1)
    console.print(f"[bold cyan]Window {window}: {test_start.date()} → {test_end.date()}[/bold cyan]\n")

    model, _, test, _, factors = fit_window(history_days, test_end)
    lgbm = format_lgbm_forecasts(generate_forecasts(model, test, factors))
    del model, test

    con = open_warehouse(WAREHOUSE)
    _, baseline_rows = run_all_baselines(con, state, test_end)
    recon, _, _ = reconcile_window(con, state, lgbm, test_end)
    results, wrmsse = evaluate_window(con, state, recon, baseline_rows, test_end)
    actuals = con.execute(
        f"""
        SELECT sale_date AS ds, CAST(SUM(units) AS DOUBLE) AS y FROM raw_sales_enriched
        WHERE state_id = '{state}'
          AND sale_date BETWEEN DATE '{test_start.date()}' AND DATE '{test_end.date()}'
        GROUP BY ds
        """
    ).df()
    con.close()

    actuals["ds"] = pd.to_datetime(actuals["ds"])
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(FORECASTS_DIR / f"backtest_w{window}_results.csv", index=False)
    wrmsse.to_csv(FORECASTS_DIR / f"backtest_w{window}_wrmsse.csv", index=False)
    daily_totals(recon, actuals.set_index("ds")["y"]).to_csv(
        FORECASTS_DIR / f"backtest_w{window}_daily_totals.csv", index=False
    )
    console.print(f"\n[bold green]✓ Window {window} complete.[/bold green]")


def summarize() -> None:
    results = pd.concat([
        pd.read_csv(REPORTS / "results.csv").assign(window=0),
        pd.read_csv(FORECASTS_DIR / "backtest_w1_results.csv").assign(window=1),
        pd.read_csv(FORECASTS_DIR / "backtest_w2_results.csv").assign(window=2),
    ], ignore_index=True)
    wrmsse = pd.concat([
        pd.read_csv(REPORTS / "wrmsse.csv").assign(window=0),
        pd.read_csv(FORECASTS_DIR / "backtest_w1_wrmsse.csv").assign(window=1),
        pd.read_csv(FORECASTS_DIR / "backtest_w2_wrmsse.csv").assign(window=2),
    ], ignore_index=True)

    results.to_csv(REPORTS / "backtest_results.csv", index=False)
    wrmsse.to_csv(REPORTS / "backtest_wrmsse.csv", index=False)

    summary = (
        wrmsse.groupby("method")["wrmsse_9_aggregate_levels"]
        .agg(["mean", "std", "min", "max"]).sort_values("mean")
    )
    tbl = Table(title="WRMSSE over 9 aggregate levels, across all 3 windows")
    tbl.add_column("Method", style="cyan")
    for col in summary.columns:
        tbl.add_column(col, style="green")
    for method, row in summary.iterrows():
        tbl.add_row(method, *[f"{v:.3f}" for v in row])
    console.print(tbl)

    # Chosen on the earlier windows only, so the final window plays no part in the choice.
    earlier = wrmsse[wrmsse["window"] > 0]
    scores = earlier[earlier["method"].isin(BIAS_VARIANTS)].groupby("method")["wrmsse_12_levels"].mean()
    console.print(f"Bias correction chosen on windows 1-2 (12-level WRMSSE): [bold]{scores.idxmin()}[/bold]")
    console.print(f"Backtest tables saved → [cyan]{REPORTS}[/cyan]")


@app.command()
def main(
    window: int = typer.Option(None, help="Run this earlier window (1 or 2)"),
    do_summarize: bool = typer.Option(False, "--summarize", help="Combine all windows into reports/"),
    state: str = typer.Option("CA", help="State scope"),
    history_days: int = typer.Option(365, help="Days of history to train on (limits memory)"),
) -> None:
    if do_summarize:
        summarize()
    else:
        run_window(window, state, history_days)


if __name__ == "__main__":
    app()
