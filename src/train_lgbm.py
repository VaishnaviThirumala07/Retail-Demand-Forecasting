"""
DS31 Phase 6: LightGBM Global Forecasting Model
Tweedie objective, leak-free features, time-based CV.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import LabelEncoder

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FEATURES_PARQUET = ROOT / "data" / "features" / "features.parquet"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"

HORIZON = 28

FEATURE_COLS = [
    "wday", "month", "year", "week_of_year",
    "snap_flag", "has_event",
    "lag_28", "lag_35", "lag_42", "lag_56",
    "roll_mean_28", "roll_mean_7", "roll_mean_14",
    "roll_std_28", "roll_max_28",
    "sell_price", "price_lag_7", "price_rel_mean",
    "price_change_7", "price_change_count",
    # Categoricals (label encoded)
    "item_id_enc", "dept_id_enc", "cat_id_enc",
    "store_id_enc", "state_id_enc", "event_type_enc",
]

CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type"]

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 512,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
}

app = typer.Typer()


def load_features(history_days: int, end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load `history_days` of features up to `end_date`, with label-encoded categoricals."""
    console.print(f"Loading features from [cyan]{FEATURES_PARQUET}[/cyan]…")
    numeric_cols = [c for c in FEATURE_COLS if not c.endswith("_enc")]
    cols = ", ".join(
        ["series_key", "sale_date", "CAST(units AS FLOAT) AS units", *CAT_COLS]
        + [f"CAST({c} AS FLOAT) AS {c}" for c in numeric_cols]
    )
    src = f"read_parquet('{FEATURES_PARQUET.as_posix()}')"
    # DuckDB defaults to 80% of RAM and one buffer per core; cap it for laptops.
    con = duckdb.connect(config={"memory_limit": "1GB", "threads": 2})
    end_sql = f"DATE '{end_date.date()}'" if end_date is not None else f"(SELECT MAX(sale_date) FROM {src})"
    df = con.execute(
        f"""
        SELECT {cols} FROM {src}
        WHERE sale_date <= {end_sql}
          AND sale_date > {end_sql} - INTERVAL '{history_days} days'
        """
    ).df()
    con.close()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    console.print(f"  {len(df):,} rows (last {history_days} days), {df.shape[1]} columns")

    # Label encode categoricals
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def time_based_split(df: pd.DataFrame, n_folds: int = 2) -> list[tuple]:
    """
    Rolling-origin cross-validation.
    Each fold has a train set and a 28-day validation window.
    Never shuffles — always respects temporal order.
    """
    max_date = df["sale_date"].max()
    folds = []
    for i in range(n_folds, 0, -1):
        val_end = max_date - pd.Timedelta(days=(i - 1) * HORIZON)
        val_start = val_end - pd.Timedelta(days=HORIZON - 1)
        train_end = val_start - pd.Timedelta(days=1)

        train = df[df["sale_date"] <= train_end]
        val = df[(df["sale_date"] >= val_start) & (df["sale_date"] <= val_end)]
        folds.append((train, val))
    return folds


def bias_factors(val_df: pd.DataFrame, val_pred: np.ndarray) -> dict:
    """Actual / predicted ratios on the validation window: one global, one per department."""
    frame = pd.DataFrame({
        "dept_id": val_df["dept_id"].to_numpy(),
        "actual": val_df["units"].to_numpy(dtype=float),
        "pred": val_pred.astype(float),
    })
    by_dept = frame.groupby("dept_id")[["actual", "pred"]].sum()
    return {
        "global": float(frame["actual"].sum() / frame["pred"].sum()),
        "dept": (by_dept["actual"] / by_dept["pred"]).to_dict(),
    }


def train_model(
    df: pd.DataFrame,
    params: dict,
    folds: list[tuple],
) -> tuple[lgb.Booster, list[float], dict]:
    """Train with early stopping on the last validation fold."""
    val_scores = []

    # Use last fold for early stopping
    train_df, val_df = folds[-1]
    folds.clear()

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["units"].to_numpy(dtype=np.float32)
    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df["units"].to_numpy(dtype=np.float32)
    del train_df

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURE_COLS)

    p = params.copy()
    n_est = p.pop("n_estimators", 2000)
    early_stop = p.pop("early_stopping_rounds", 50)

    callbacks = [
        lgb.early_stopping(early_stop, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    console.print(f"\nTraining LightGBM (n_estimators≤{n_est}, early_stop={early_stop})…")
    t0 = time.time()

    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_scores.append(val_rmse)
    factors = bias_factors(val_df, val_pred)

    console.print(
        f"  Best iteration: {model.best_iteration}  |  Val RMSE: {val_rmse:.4f}  |  "
        f"Global bias factor: {factors['global']:.4f}  |  {time.time()-t0:.1f}s"
    )
    return model, val_scores, factors


def generate_forecasts(model: lgb.Booster, test: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """
    Direct forecast of the held-out test window. Every lag/rolling feature is
    at least 28 days old, so each test row's features were already known at
    the forecast origin (the day before the window starts).
    Bias factors come from the validation window, never from the test window.
    """
    console.print(f"\nForecasting held-out test window ({test['sale_date'].nunique()} days)…")
    forecasts = test[["series_key", "item_id", "dept_id", "store_id", "state_id", "sale_date"]].copy()
    preds = model.predict(test[FEATURE_COLS].to_numpy(dtype=np.float32))
    forecasts["forecast"] = np.clip(preds, 0, None)
    forecasts["forecast_bc_global"] = forecasts["forecast"] * factors["global"]
    dept_factor = forecasts["dept_id"].map(factors["dept"]).fillna(factors["global"]).astype(float)
    forecasts["forecast_bc_dept"] = forecasts["forecast"] * dept_factor
    forecasts["horizon_day"] = (forecasts["sale_date"] - forecasts["sale_date"].min()).dt.days + 1
    console.print(f"  {len(forecasts):,} forecast rows")
    return forecasts


def log_to_mlflow(params: dict, metrics: dict, model: lgb.Booster) -> None:
    """Lightweight MLflow logging — per-level RMSSE and model artifact."""
    mlflow.set_experiment("DS31-LightGBM")
    with mlflow.start_run(run_name="lgbm_tweedie"):
        mlflow.log_params(
            {
                "objective": params["objective"],
                "tweedie_power": params["tweedie_variance_power"],
                "num_leaves": params["num_leaves"],
                "learning_rate": params["learning_rate"],
                "best_iteration": model.best_iteration,
            }
        )
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "lgbm_model.txt"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))
    console.print("[dim]MLflow run logged.[/dim]")


def fit_window(history_days: int, end_date: pd.Timestamp | None = None) -> tuple:
    """Train on data before the 28-day test window that ends at `end_date` (default: latest date)."""
    df, encoders = load_features(history_days, end_date)

    test_start = df["sale_date"].max() - pd.Timedelta(days=HORIZON - 1)
    is_test = df["sale_date"] >= test_start
    test = df[is_test].copy()
    df = df[~is_test]
    console.print(f"  Held out {len(test):,} test rows from {test_start.date()}")

    # Only the final fold is used (early stopping); building more just copies data.
    model, val_scores, factors = train_model(df, LGBM_PARAMS, time_based_split(df, n_folds=1))
    return model, encoders, test, val_scores, factors


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging"),
    history_days: int = typer.Option(365, help="Days of history to train on (limits memory)"),
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    model, encoders, test, val_scores, factors = fit_window(history_days)

    with open(REPORTS / "bias_correction.json", "w") as f:
        json.dump(factors, f, indent=2)

    # Feature importance
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    console.print("\n[bold]Top 10 features by gain:[/bold]")
    console.print(imp.head(10).to_string(index=False))

    # Save model and encoders
    model_path = MODELS_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    with open(MODELS_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)

    # Generate forecasts
    forecasts = generate_forecasts(model, test, factors)
    out = ROOT / "data" / "forecasts" / "lgbm_forecasts.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out, index=False)
    console.print(f"\nForecasts saved → [cyan]{out}[/cyan]")

    metrics = {"val_rmse_fold1": float(val_scores[0])}

    if not no_mlflow:
        try:
            log_to_mlflow(LGBM_PARAMS, metrics, model)
        except Exception as e:
            console.print(f"[yellow]MLflow logging skipped: {e}[/yellow]")

    console.print("\n[bold green]✓ LightGBM training complete.[/bold green]")


if __name__ == "__main__":
    app()
