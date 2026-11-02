"""
DS31 Phase 5: Baseline Forecasts
Naive, seasonal-naive, and statsforecast AutoETS/AutoARIMA at aggregate levels.
All baselines are evaluated on the last 28 days (test set).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    HistoricAverage,
    Naive,
    SeasonalNaive,
)

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"
FORECASTS_DIR = ROOT / "data" / "forecasts"

HORIZON = 28
FREQ = "D"
SEASON_LENGTH = 7

# The 9 M5 aggregate levels. SKU levels are left out: AutoETS on ~12k series is impractical here.
BASELINE_LEVELS = {
    "L1_total":      "'total'",
    "L2_state":      "state_id",
    "L3_store":      "store_id",
    "L4_cat":        "cat_id",
    "L5_dept":       "dept_id",
    "L6_state_cat":  "state_id || '/' || cat_id",
    "L7_state_dept": "state_id || '/' || dept_id",
    "L8_store_cat":  "store_id || '/' || cat_id",
    "L9_store_dept": "store_id || '/' || dept_id",
}

app = typer.Typer()


def load_aggregated_series(
    con: duckdb.DuckDBPyConnection,
    level: str,
    scope_state: str = "CA",
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load time series aggregated at a given hierarchy level.
    Series ids match the keys in metrics.load_level_scales. `level` is an M5 level name.
    """
    state_filter = f"AND state_id = '{scope_state}'" if scope_state else ""
    end_filter = f"AND sale_date <= DATE '{end_date.date()}'" if end_date is not None else ""

    df = con.execute(
        f"""
        SELECT
            {BASELINE_LEVELS[level]} AS unique_id,
            sale_date AS ds,
            SUM(units) AS y
        FROM raw_sales_enriched
        WHERE 1=1 {state_filter} {end_filter}
        GROUP BY unique_id, ds
        ORDER BY unique_id, ds
        """
    ).df()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.rename(columns={"ds": "ds"})
    return df


def split_train_test(df: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split — last `horizon` days as test."""
    cutoff = df["ds"].max() - pd.Timedelta(days=horizon)
    train = df[df["ds"] <= cutoff].copy()
    test = df[df["ds"] > cutoff].copy()
    return train, test


def run_baselines(train: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Fit Naive, SeasonalNaive, HistoricAverage, AutoETS on train data."""
    models = [
        Naive(),
        SeasonalNaive(season_length=SEASON_LENGTH),
        HistoricAverage(),
        AutoETS(season_length=SEASON_LENGTH),
    ]

    sf = StatsForecast(
        models=models,
        freq=FREQ,
        n_jobs=1,
        verbose=False,
    )

    sf.fit(train)
    forecasts = sf.predict(h=horizon)
    return forecasts


def compute_rmsse(train: pd.DataFrame, test: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    RMSSE per series per model.
    RMSSE = RMSE / sqrt(mean(squared one-step naive errors on train))
    """
    # Naive benchmark error (denominator)
    naive_errors = {}
    for uid, g in train.groupby("unique_id"):
        y = g.sort_values("ds")["y"].values
        diffs = np.diff(y)
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) > 0:
            naive_errors[uid] = np.mean(diffs**2)
        else:
            naive_errors[uid] = 1.0

    results = []
    model_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]

    for uid, grp_f in forecasts.groupby("unique_id"):
        grp_t = test[test["unique_id"] == uid].sort_values("ds")
        grp_f = grp_f.sort_values("ds")
        denom = max(naive_errors.get(uid, 1.0), 1e-8)

        for model in model_cols:
            y_hat = grp_f[model].values
            y_true = grp_t["y"].values[: len(y_hat)]
            if len(y_true) == 0:
                continue
            mse = np.mean((y_hat - y_true) ** 2)
            rmsse = np.sqrt(mse / denom)
            results.append({"unique_id": uid, "model": model, "rmsse": rmsse})

    return pd.DataFrame(results)


def run_all_baselines(
    con: duckdb.DuckDBPyConnection,
    scope_state: str = "CA",
    end_date: pd.Timestamp | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Run baselines at each level; return RMSSE summaries and the forecast rows."""
    all_results = {}
    all_forecasts = []

    for level in BASELINE_LEVELS:
        console.print(f"\nLevel: [cyan]{level}[/cyan]")
        df = load_aggregated_series(con, level, scope_state, end_date)
        n_series = df["unique_id"].nunique()
        console.print(f"  {n_series} series, {len(df):,} rows")

        train, test = split_train_test(df)
        t0 = time.time()
        forecasts = run_baselines(train)
        elapsed = time.time() - t0
        all_forecasts.append(forecasts.assign(level=level))

        rmsse_df = compute_rmsse(train, test, forecasts)
        summary = rmsse_df.groupby("model")["rmsse"].mean().to_dict()
        all_results[level] = summary

        tbl = Table(title=f"RMSSE @ {level}")
        tbl.add_column("Model")
        tbl.add_column("Mean RMSSE", style="green")
        for m, v in sorted(summary.items(), key=lambda x: x[1]):
            tbl.add_row(m, f"{v:.4f}")
        console.print(tbl)
        console.print(f"  Computed in {elapsed:.1f}s")

    return all_results, pd.concat(all_forecasts, ignore_index=True)


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB path"),
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(warehouse), read_only=True, config={"memory_limit": "1500MB", "threads": 2})
    results, forecasts = run_all_baselines(con, scope_state=state)
    con.close()

    out = REPORTS / "baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(FORECASTS_DIR / "baseline_forecasts.parquet", index=False)
    console.print(f"\n[bold green]✓ Baselines saved → {out}[/bold green]")


if __name__ == "__main__":
    app()
