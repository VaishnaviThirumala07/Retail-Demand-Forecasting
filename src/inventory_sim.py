"""
DS31 Phase 9: Inventory Cost Simulation
Compares stockout and holding cost for each forecasting method on the same dates,
under the same policy. All cost assumptions are stated explicitly.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from method_names import DISPLAY_METHODS
from metrics import bottom_level

console = Console()
ROOT = Path(__file__).resolve().parent.parent
FORECASTS_DIR = ROOT / "data" / "forecasts"
REPORTS = ROOT / "reports"
HORIZON = 28

# ── Cost assumptions (stated explicitly) ────────────────────────────────────
# Stockout cost: lost margin per unit not sold (assumed 30% of average price)
# Holding cost: carrying cost per unit per day (assumed 0.5% of average price/day)
# Average unit price: $3.50 (approximate M5 Walmart grocery item)
AVG_PRICE = 3.50
STOCKOUT_COST_PER_UNIT = AVG_PRICE * 0.30   # = $1.05
HOLDING_COST_PER_UNIT_DAY = AVG_PRICE * 0.005  # = $0.0175/unit/day
SAFETY_STOCK_Z = 1.65  # 95th percentile coverage

app = typer.Typer()


def order_up_to_policy(
    forecast: np.ndarray,
    forecast_std: float,
    actuals: np.ndarray,
    z: float = SAFETY_STOCK_Z,
) -> dict:
    """
    Order-up-to inventory policy.
    Order quantity = forecast + z * forecast_std (safety stock).
    Returns total stockout and holding cost over the horizon.

    forecast     : point forecasts (units), length H
    forecast_std : std dev of forecast errors, from a window before the test window
    actuals      : actual demand, length H
    z            : safety stock multiplier (z-score for service level)
    """
    order_qty = np.maximum(forecast + z * forecast_std, 0)

    stockout_units = np.maximum(actuals - order_qty, 0)
    overstock_units = np.maximum(order_qty - actuals, 0)

    stockout_cost = np.sum(stockout_units) * STOCKOUT_COST_PER_UNIT
    holding_cost = np.sum(overstock_units) * HOLDING_COST_PER_UNIT_DAY * HORIZON / 2

    return {
        "stockout_units": float(np.sum(stockout_units)),
        "overstock_units": float(np.sum(overstock_units)),
        "stockout_cost": float(stockout_cost),
        "holding_cost": float(holding_cost),
        "total_cost": float(stockout_cost + holding_cost),
        "fill_rate": float(1 - np.sum(stockout_units) / max(np.sum(actuals), 1)),
    }


def prior_error_std(method: str) -> float:
    """
    Forecast error std dev from the preceding window (backtest window 1).
    Using the test window's own errors would let each method size its safety
    stock with knowledge of the demand it is being scored on.
    """
    prior = pd.read_csv(FORECASTS_DIR / "backtest_w1_daily_totals.csv")
    return float(np.std(prior[method] - prior["actual"]))


@app.command()
def main(
    warehouse: Path = typer.Option(ROOT / "warehouse" / "m5.duckdb", help="DuckDB path"),
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    console.print("[bold cyan]DS31 — Phase 9: Inventory Cost Simulation[/bold cyan]\n")

    con = duckdb.connect(str(warehouse), read_only=True, config={"memory_limit": "1GB", "threads": 2})
    actuals_df = con.execute(
        """
        SELECT sale_date AS ds, CAST(SUM(units) AS DOUBLE) AS y
        FROM raw_sales_enriched
        WHERE sale_date > (SELECT MAX(sale_date) - INTERVAL '28 days' FROM raw_sales_enriched)
          AND state_id = 'CA'
        GROUP BY ds
        ORDER BY ds
        """
    ).df()
    con.close()
    actuals_df["ds"] = pd.to_datetime(actuals_df["ds"])
    actuals = actuals_df.set_index("ds")["y"]

    recon = bottom_level(pd.read_parquet(FORECASTS_DIR / "reconciled_forecasts.parquet"))
    methods = [m for m in DISPLAY_METHODS if m in recon.columns]
    totals = recon[recon["state_id"] == "CA"].groupby("ds")[methods].sum().loc[actuals.index]

    results = []
    for column in methods:
        name = DISPLAY_METHODS[column]
        results.append({
            "method": name,
            **order_up_to_policy(totals[column].to_numpy(), prior_error_std(name), actuals.to_numpy()),
        })

    tbl = Table(title="Inventory Cost Simulation (CA, 28-day test period)")
    tbl.add_column("Method", style="cyan")
    tbl.add_column("Stockout Cost ($)", style="red")
    tbl.add_column("Holding Cost ($)", style="yellow")
    tbl.add_column("Total Cost ($)", style="bold")
    tbl.add_column("Fill Rate", style="green")
    for r in results:
        tbl.add_row(
            r["method"],
            f"${r['stockout_cost']:,.2f}",
            f"${r['holding_cost']:,.2f}",
            f"${r['total_cost']:,.2f}",
            f"{r['fill_rate']:.1%}",
        )
    console.print(tbl)

    pd.DataFrame(results).to_csv(REPORTS / "inventory_simulation.csv", index=False)
    console.print(f"\nSimulation saved → [cyan]{REPORTS / 'inventory_simulation.csv'}[/cyan]")

    console.print("\n[dim]Cost assumptions:[/dim]")
    console.print(f"  Avg unit price: ${AVG_PRICE}")
    console.print(f"  Stockout cost/unit: ${STOCKOUT_COST_PER_UNIT:.2f} (30% margin)")
    console.print(f"  Holding cost/unit/day: ${HOLDING_COST_PER_UNIT_DAY:.4f} (0.5%/day)")
    console.print(f"  Safety stock z-score: {SAFETY_STOCK_Z} (95% service level)")

    console.print("\n[bold green]✓ Inventory simulation complete.[/bold green]")


if __name__ == "__main__":
    app()
