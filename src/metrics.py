"""
DS31 Phase 8: Evaluation Metrics
Per-level RMSSE, MAE and bias, plus M5-style WRMSSE, for any 28-day test window.
"""
from __future__ import annotations

import re
from functools import reduce
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from method_names import DISPLAY_METHODS

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FORECASTS_DIR = ROOT / "data" / "forecasts"
REPORTS = ROOT / "reports"
TEMP_DIR = ROOT / "data" / "duckdb_tmp"
HORIZON = 28

# The 12 M5 aggregation levels. With one state in scope, the state levels duplicate
# their all-state counterparts; they are kept so each level still weighs 1/12.
WRMSSE_LEVELS = {
    "L1_total":       [],
    "L2_state":       ["state_id"],
    "L3_store":       ["store_id"],
    "L4_cat":         ["cat_id"],
    "L5_dept":        ["dept_id"],
    "L6_state_cat":   ["state_id", "cat_id"],
    "L7_state_dept":  ["state_id", "dept_id"],
    "L8_store_cat":   ["store_id", "cat_id"],
    "L9_store_dept":  ["store_id", "dept_id"],
    "L10_item":       ["item_id"],
    "L11_item_state": ["item_id", "state_id"],
    "L12_item_store": ["item_id", "store_id"],
}
AGGREGATE_LEVELS = list(WRMSSE_LEVELS)[:9]

# Levels shown in the accuracy table, named as in reports/results.csv.
REPORT_LEVELS = {
    "total":    "L1_total",
    "state":    "L2_state",
    "store":    "L3_store",
    "category": "L4_cat",
    "dept":     "L5_dept",
}


def rmsse(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Scaled Error.
    Denominator: mean squared one-step naive error on training series.
    """
    h = len(y_true)
    mse_pred = np.mean((y_pred[:h] - y_true[:h]) ** 2)
    naive_sq_errors = np.diff(y_train) ** 2
    denom = np.mean(naive_sq_errors) if len(naive_sq_errors) > 0 else 1.0
    denom = max(denom, 1e-8)
    return float(np.sqrt(mse_pred / denom))


# ---------------------------------------------------------------------------
# Scaling denominators and weights
# ---------------------------------------------------------------------------

def load_level_scales(
    con: duckdb.DuckDBPyConnection,
    state: str,
    keys: list[str],
    train_end: pd.Timestamp,
    extra_filter: str = "",
) -> pd.DataFrame:
    """
    For one level, per series:
    scale  = mean squared one-step difference over training, from the first non-zero sale
    revenue = dollar sales over the last 28 training days (the M5 weight, before normalising)
    """
    key_sql = "concat_ws('/', " + ", ".join(keys) + ")" if keys else "'total'"
    return con.execute(
        f"""
        WITH daily AS (
            SELECT {key_sql} AS key, sale_date,
                   SUM(units) AS y,
                   SUM(units * COALESCE(sell_price, 0)) AS revenue
            FROM raw_sales_enriched
            WHERE state_id = '{state}' AND sale_date <= DATE '{train_end.date()}' {extra_filter}
            GROUP BY 1, 2
        ), diffs AS (
            SELECT key, sale_date, revenue,
                   y - LAG(y) OVER w AS dy,
                   SUM(y) OVER (PARTITION BY key ORDER BY sale_date
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS sold_before
            FROM daily
            WINDOW w AS (PARTITION BY key ORDER BY sale_date)
        )
        SELECT key,
               AVG(dy * dy) FILTER (WHERE sold_before > 0) AS scale,
               COALESCE(SUM(revenue) FILTER (
                   WHERE sale_date > DATE '{train_end.date()}' - INTERVAL '{HORIZON} days'), 0) AS revenue
        FROM diffs
        GROUP BY key
        """
    ).df()


def load_scales(con: duckdb.DuckDBPyConnection, state: str, train_end: pd.Timestamp) -> dict:
    """Scale and normalised weight for every series at all 12 levels, indexed by series key."""
    stores = [r[0] for r in con.execute(
        f"SELECT DISTINCT store_id FROM raw_sales_enriched WHERE state_id = '{state}'"
    ).fetchall()]

    scales = {}
    for level, keys in WRMSSE_LEVELS.items():
        if level == "L12_item_store":
            # One store at a time keeps the window sort over ~24M rows small.
            table = pd.concat([
                load_level_scales(con, state, keys, train_end, f"AND store_id = '{store}'")
                for store in stores
            ], ignore_index=True)
        else:
            table = load_level_scales(con, state, keys, train_end)
        table["weight"] = table["revenue"] / table["revenue"].sum()
        scales[level] = table.set_index("key")[["scale", "weight"]]
    return scales


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def level_frame(bottom: pd.DataFrame, keys: list[str], methods: list[str]) -> pd.DataFrame:
    """Sum SKU-store rows up to one level, with one `key` column matching load_level_scales."""
    if keys:
        grouped = bottom.groupby([*keys, "ds"], observed=True)[["y", *methods]].sum().reset_index()
        grouped["key"] = reduce(lambda a, b: a + "/" + b, (grouped[k].astype(str) for k in keys))
    else:
        grouped = bottom.groupby("ds")[["y", *methods]].sum().reset_index()
        grouped["key"] = "total"
    return grouped


def score_series(frame: pd.DataFrame, method: str, scale: pd.DataFrame) -> pd.DataFrame:
    """RMSSE, MAE and bias for each series. Series that never sold in training are dropped."""
    err = frame[method] - frame["y"]
    stats = pd.DataFrame({
        "mse": (err ** 2).groupby(frame["key"]).mean(),
        "mae": err.abs().groupby(frame["key"]).mean(),
        "bias": err.groupby(frame["key"]).mean(),
    }).join(scale, how="inner")
    stats = stats[stats["scale"] > 0]
    stats["rmsse"] = np.sqrt(stats["mse"] / stats["scale"])
    return stats


def wrmsse_by_level(frames: dict, method: str, scales: dict) -> dict:
    """Weighted RMSSE per level: the sum of (dollar weight x RMSSE) across that level's series."""
    out = {}
    for level, frame in frames.items():
        stats = score_series(frame, method, scales[level])
        out[level] = float((stats["weight"] * stats["rmsse"]).sum())
    return out


def bottom_level(recon: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only SKU-store rows of the reconciled forecasts and split their
    "CA/CA_1/FOODS/FOODS_1/FOODS_1_001" id. Upper levels are sums of these
    (reconciled forecasts are coherent).
    """
    recon = recon[recon["unique_id"].str.count("/") == 4].copy()
    parts = recon["unique_id"].str.split("/", expand=True)
    parts.columns = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
    recon = pd.concat([recon, parts], axis=1)
    recon["ds"] = pd.to_datetime(recon["ds"])
    return recon


def load_actuals(
    con: duckdb.DuckDBPyConnection, state: str, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> pd.DataFrame:
    actuals = con.execute(
        f"""
        SELECT store_id, item_id, sale_date AS ds, CAST(units AS DOUBLE) AS y
        FROM raw_sales_enriched
        WHERE state_id = '{state}'
          AND sale_date BETWEEN DATE '{test_start.date()}' AND DATE '{test_end.date()}'
        """
    ).df()
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    return actuals


def evaluate_window(
    con: duckdb.DuckDBPyConnection,
    state: str,
    recon: pd.DataFrame,
    baseline_rows: pd.DataFrame | None,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score one 28-day window. Returns (per-level results, WRMSSE table), with methods
    named as in method_names.DISPLAY_METHODS. Baseline models other than seasonal naive
    only exist at the aggregate levels, so their 12-level WRMSSE is NaN.
    """
    test_start = test_end - pd.Timedelta(days=HORIZON - 1)
    train_end = test_start - pd.Timedelta(days=1)

    methods = [m for m in DISPLAY_METHODS if m in recon.columns]
    bottom = bottom_level(recon)[["store_id", "item_id", "state_id", "cat_id", "dept_id", "ds", *methods]]
    bottom = bottom.merge(load_actuals(con, state, test_start, test_end), on=["store_id", "item_id", "ds"])

    console.print("Computing scaling denominators and weights in DuckDB…")
    scales = load_scales(con, state, train_end)
    frames = {lvl: level_frame(bottom, keys, methods) for lvl, keys in WRMSSE_LEVELS.items()}

    result_rows, wrmsse_rows = [], []
    for method in methods:
        by_level = wrmsse_by_level(frames, method, scales)
        wrmsse_rows.append({
            "method": DISPLAY_METHODS[method],
            "wrmsse_12_levels": np.mean(list(by_level.values())),
            "wrmsse_9_aggregate_levels": np.mean([by_level[lvl] for lvl in AGGREGATE_LEVELS]),
        })
        for name, lvl in REPORT_LEVELS.items():
            stats = score_series(frames[lvl], method, scales[lvl])
            result_rows.append({
                "method": DISPLAY_METHODS[method], "level": name,
                "mean_rmsse": stats["rmsse"].mean(), "n_series": len(stats),
                "mean_mae": stats["mae"].mean(), "mean_bias": stats["bias"].mean(),
            })

    if baseline_rows is not None:
        models = [c for c in baseline_rows.columns
                  if c not in ("level", "unique_id", "ds", "SeasonalNaive")]
        level_tables = {
            lvl: baseline_rows[baseline_rows["level"] == lvl].rename(columns={"unique_id": "key"})
            .merge(frames[lvl][["key", "ds", "y"]], on=["key", "ds"])
            for lvl in AGGREGATE_LEVELS
        }
        for model in models:
            by_level = wrmsse_by_level(level_tables, model, scales)
            wrmsse_rows.append({
                "method": model, "wrmsse_12_levels": np.nan,
                "wrmsse_9_aggregate_levels": np.mean(list(by_level.values())),
            })
            for name, lvl in REPORT_LEVELS.items():
                stats = score_series(level_tables[lvl], model, scales[lvl])
                result_rows.append({
                    "method": model, "level": name,
                    "mean_rmsse": stats["rmsse"].mean(), "n_series": len(stats),
                    "mean_mae": stats["mae"].mean(), "mean_bias": stats["bias"].mean(),
                })

    return pd.DataFrame(result_rows), pd.DataFrame(wrmsse_rows)


def open_warehouse(path: Path) -> duckdb.DuckDBPyConnection:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=True, config={
        "memory_limit": "1500MB", "threads": 2, "temp_directory": str(TEMP_DIR),
    })


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB path"),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging"),
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    console.print("[bold cyan]DS31 — Phase 8: Evaluation[/bold cyan]\n")

    recon = pd.read_parquet(FORECASTS_DIR / "reconciled_forecasts.parquet")
    baselines_path = FORECASTS_DIR / "baseline_forecasts.parquet"
    baseline_rows = pd.read_parquet(baselines_path) if baselines_path.exists() else None

    con = open_warehouse(warehouse)
    test_end = con.execute("SELECT MAX(sale_date) FROM raw_sales_enriched").fetchone()[0]
    results, wrmsse = evaluate_window(con, state, recon, baseline_rows, pd.Timestamp(test_end))
    con.close()

    pivot = results.pivot(index="method", columns="level", values="mean_rmsse")[list(REPORT_LEVELS)]
    tbl = Table(title="Mean RMSSE per level")
    tbl.add_column("Method", style="cyan")
    for lvl in REPORT_LEVELS:
        tbl.add_column(lvl, style="green")
    for method, row in pivot.iterrows():
        tbl.add_row(method, *[f"{v:.3f}" for v in row])
    console.print(tbl)

    tbl = Table(title="M5-style WRMSSE (one state in scope)")
    tbl.add_column("Method", style="cyan")
    tbl.add_column("12 levels", style="green")
    tbl.add_column("9 aggregate levels", style="green")
    for _, row in wrmsse.iterrows():
        tbl.add_row(row["method"], f"{row['wrmsse_12_levels']:.3f}", f"{row['wrmsse_9_aggregate_levels']:.3f}")
    console.print(tbl)

    results.to_csv(REPORTS / "results.csv", index=False)
    wrmsse.to_csv(REPORTS / "wrmsse.csv", index=False)
    console.print(f"\nResults saved → [cyan]{REPORTS}[/cyan]")

    if not no_mlflow:
        try:
            mlflow.set_experiment("DS31-Evaluation")
            with mlflow.start_run(run_name="per_level_rmsse"):
                for _, row in results.iterrows():
                    method = re.sub(r"[^\w]+", "_", row["method"])
                    mlflow.log_metric(f"rmsse_{method}_{row['level']}", row["mean_rmsse"])
            console.print("[dim]MLflow evaluation run logged.[/dim]")
        except Exception as e:
            console.print(f"[yellow]MLflow logging skipped: {e}[/yellow]")

    console.print("\n[bold green]✓ Evaluation complete.[/bold green]")


if __name__ == "__main__":
    app()
