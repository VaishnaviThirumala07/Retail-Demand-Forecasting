"""
DS31 Phase 4 (Hierarchy) + Phase 7: Hierarchical Reconciliation
Uses Nixtla hierarchicalforecast to reconcile bottom-up LightGBM forecasts.
Reconciliation methods: Bottom-Up, Top-Down, MinTrace(mint_shrink).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
import typer
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUpSparse, MinTraceSparse, TopDownSparse
from hierarchicalforecast.utils import aggregate
from rich.console import Console
from rich.table import Table
from scipy.sparse import save_npz

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FORECASTS_DIR = ROOT / "data" / "forecasts"
REPORTS = ROOT / "reports"

HORIZON = 28
# SeasonalNaive only needs the last week before the cutoff; loading all ~1,941
# days for ~12k series (23M+ rows) exhausts RAM on a 16 GB laptop.
HISTORY_DAYS = HORIZON + 56

LGBM_MODELS = {
    "forecast": "LightGBM",
    "forecast_bc_global": "LightGBM_BC_Global",
    "forecast_bc_dept": "LightGBM_BC_Dept",
}

app = typer.Typer()


# ---------------------------------------------------------------------------
# Hierarchy definition
# ---------------------------------------------------------------------------

def build_hierarchy_spec() -> list[list[str]]:
    """
    M5 hierarchy mapped to brief's wording:
    Total → State (Regional) → Store → Category → Department → Item_Store (SKU)
    Returns a list of column paths, from coarsest to finest.
    """
    return [
        ["state_id"],
        ["state_id", "store_id"],
        ["state_id", "store_id", "cat_id"],
        ["state_id", "store_id", "cat_id", "dept_id"],
        ["state_id", "store_id", "cat_id", "dept_id", "item_id"],
    ]


def load_bottom_series(
    con: duckdb.DuckDBPyConnection,
    scope_state: str,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load HISTORY_DAYS of bottom-level (item × store) daily sales up to `end_date`."""
    state_filter = f"AND state_id = '{scope_state}'" if scope_state else ""
    end_sql = (
        f"DATE '{end_date.date()}'" if end_date is not None
        else "(SELECT MAX(sale_date) FROM raw_sales_enriched)"
    )
    df = con.execute(
        f"""
        SELECT
            state_id,
            store_id,
            cat_id,
            dept_id,
            item_id,
            sale_date AS ds,
            CAST(SUM(units) AS DOUBLE) AS y
        FROM raw_sales_enriched
        WHERE sale_date <= {end_sql}
          AND sale_date > {end_sql} - INTERVAL '{HISTORY_DAYS} days'
          {state_filter}
        GROUP BY state_id, store_id, cat_id, dept_id, item_id, ds
        ORDER BY store_id, item_id, ds
        """
    ).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def build_summing_matrix(df: pd.DataFrame) -> tuple:
    """
    Build summing matrix S and tags using hierarchicalforecast.utils.aggregate.
    The spec defines the column paths at each level.
    """
    hierarchy_spec = build_hierarchy_spec()
    # aggregate builds its own unique_id ("CA/CA_1/FOODS/FOODS_1/FOODS_1_001"),
    # so ours must not be passed in. Sparse S keeps memory flat for ~12k series.
    Y_df, S_df, tags = aggregate(
        df=df[["ds", "y", "state_id", "store_id", "cat_id", "dept_id", "item_id"]],
        spec=hierarchy_spec,
        sparse_s=True,
    )
    return Y_df, S_df, tags


def bottom_id(item_id: pd.Series, store_id: pd.Series) -> pd.Series:
    """Bottom-level unique_id in the format produced by `aggregate`."""
    dept_id = item_id.str.rsplit("_", n=1).str[0]
    cat_id = item_id.str.split("_").str[0]
    state_id = store_id.str.split("_").str[0]
    return state_id + "/" + store_id + "/" + cat_id + "/" + dept_id + "/" + item_id


def format_lgbm_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Rename LightGBM forecast columns and key them by bottom-level hierarchy id."""
    df = df.rename(columns={"sale_date": "ds", **LGBM_MODELS})
    df["unique_id"] = bottom_id(df["item_id"], df["store_id"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df[["unique_id", "ds", *LGBM_MODELS.values()]]


def load_lgbm_forecasts() -> pd.DataFrame | None:
    """Load LightGBM base forecasts if available."""
    path = FORECASTS_DIR / "lgbm_forecasts.parquet"
    if path.exists():
        return format_lgbm_forecasts(pd.read_parquet(path))
    return None


def prepare_base_forecasts(
    Y_df: pd.DataFrame,
    lgbm_df: pd.DataFrame | None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Build Y_hat_df: base forecasts for all series at all levels.
    - Bottom level: use LightGBM if available, else SeasonalNaive
    - Upper levels: use seasonal naive (they get reconciled anyway)
    """
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    # Train cutoff = all data minus last horizon days
    max_date = Y_df["ds"].max()
    cutoff = max_date - pd.Timedelta(days=horizon)
    train = Y_df[Y_df["ds"] <= cutoff].copy()

    # Seasonal naive for all series (upper levels and fallback)
    # n_jobs=1: on Windows each worker re-imports this module; SeasonalNaive is cheap anyway.
    sf = StatsForecast(models=[SeasonalNaive(season_length=7)], freq="D", n_jobs=1)
    sf.fit(train[["unique_id", "ds", "y"]])
    sn_forecasts = sf.predict(h=horizon)

    # Merge LightGBM bottom-level forecasts
    if lgbm_df is not None:
        Y_hat = sn_forecasts.merge(lgbm_df, on=["unique_id", "ds"], how="left")
        # Upper levels and any missing SKU rows use seasonal naive
        for model in LGBM_MODELS.values():
            Y_hat[model] = Y_hat[model].fillna(Y_hat["SeasonalNaive"])
    else:
        Y_hat = sn_forecasts.copy()
        for model in LGBM_MODELS.values():
            Y_hat[model] = Y_hat["SeasonalNaive"]

    return Y_hat


def reconcile_forecasts(
    Y_df: pd.DataFrame,
    Y_hat_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict,
) -> pd.DataFrame:
    """Apply Bottom-Up, Top-Down, and MinTrace reconciliation."""
    # wls_struct needs no in-sample residuals (mint_shrink would) and
    # scales to ~12k series with a sparse S.
    reconcilers = [
        BottomUpSparse(),
        TopDownSparse(method="forecast_proportions"),
        MinTraceSparse(method="wls_struct"),
    ]

    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    console.print("  Running reconciliation (this may take a minute)…")
    t0 = time.time()

    reconciled = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_df,
        S_df=S_df,
        tags=tags,
    )
    console.print(f"  Done in {time.time()-t0:.1f}s")
    return reconciled


def verify_coherence(reconciled: pd.DataFrame) -> bool:
    """Assert that every reconciled method's stores and SKUs sum to the state total each day."""
    console.print("\n[bold]Coherence verification…[/bold]")

    depth = reconciled["unique_id"].str.count("/")
    rec_cols = [c for c in reconciled.columns if "/" in c]

    all_pass = True
    for col in rec_cols:
        top = reconciled[depth == 0].groupby("ds")[col].sum()
        for level in (1, 4):
            children = reconciled[depth == level].groupby("ds")[col].sum()
            if not np.allclose(top, children, rtol=1e-6):
                console.print(f"  [red]✗ {col}: level {level} does not sum to the total[/red]")
                all_pass = False
    return all_pass


def reconcile_window(
    con: duckdb.DuckDBPyConnection,
    state: str,
    lgbm_df: pd.DataFrame | None,
    end_date: pd.Timestamp | None = None,
) -> tuple:
    """Build the hierarchy up to `end_date`, then reconcile base forecasts for its last 28 days."""
    console.print(f"Loading bottom-level series (state={state})…")
    df = load_bottom_series(con, scope_state=state, end_date=end_date)
    n_series = len(df[["store_id", "item_id"]].drop_duplicates())
    console.print(f"  {n_series:,} bottom series, {len(df):,} rows")

    console.print("\nBuilding summing matrix S…")
    Y_df, S_df, tags = build_summing_matrix(df)
    console.print(f"  {Y_df['unique_id'].nunique():,} total series across all levels")
    for level, uids in tags.items():
        console.print(f"    {level}: {len(uids)} series")

    console.print("Preparing base forecasts for all levels…")
    Y_hat_df = prepare_base_forecasts(Y_df, lgbm_df)

    console.print("\nReconciling forecasts…")
    reconciled = reconcile_forecasts(Y_df, Y_hat_df, S_df, tags)
    return reconciled, S_df, tags


def log_reconciliation_metrics(results: dict) -> None:
    """Log per-level RMSSE for each reconciliation method to MLflow."""
    try:
        mlflow.set_experiment("DS31-Reconciliation")
        with mlflow.start_run(run_name="reconciliation_comparison"):
            for method, levels in results.items():
                for level, rmsse in levels.items():
                    mlflow.log_metric(f"rmsse_{method}_{level}", rmsse)
        console.print("[dim]MLflow reconciliation run logged.[/dim]")
    except Exception as e:
        console.print(f"[yellow]MLflow logging skipped: {e}[/yellow]")


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB path"),
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]DS31 — Phase 7: Hierarchical Reconciliation[/bold cyan]\n")

    con = duckdb.connect(str(warehouse), read_only=True, config={"memory_limit": "3GB"})

    # Load LightGBM base forecasts
    lgbm_df = load_lgbm_forecasts()
    if lgbm_df is not None:
        console.print(f"Loaded LightGBM forecasts ({len(lgbm_df):,} rows)\n")
    else:
        console.print("[yellow]LightGBM forecasts not found — using SeasonalNaive as base[/yellow]\n")

    reconciled, S_df, tags = reconcile_window(con, state, lgbm_df)
    con.close()

    # Save
    out = FORECASTS_DIR / "reconciled_forecasts.parquet"
    reconciled.to_parquet(out, index=False)
    console.print(f"\nReconciled forecasts saved → [cyan]{out}[/cyan]")

    # Verify coherence
    coherent = verify_coherence(reconciled)

    # Save tags for later use
    tags_out = REPORTS / "hierarchy_tags.json"
    with open(tags_out, "w") as f:
        json.dump({k: list(v) for k, v in tags.items()}, f, indent=2)
    console.print(f"Hierarchy tags saved → [cyan]{tags_out}[/cyan]")

    s_out = REPORTS / "summing_matrix.npz"
    save_npz(s_out, S_df.to_csr())
    console.print(f"Summing matrix saved → [cyan]{s_out}[/cyan]")

    if not coherent:
        raise SystemExit("Coherence check failed: reconciled forecasts do not add up.")
    console.print("\n[bold green]✓ Coherence verified[/bold green]")


if __name__ == "__main__":
    app()
