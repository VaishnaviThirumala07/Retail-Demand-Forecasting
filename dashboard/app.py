"""
Streamlit dashboard over the pipeline's saved outputs.
Tabs: Forecast Explorer | Reconciliation Comparison | Accuracy Report | Inventory Simulation
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from method_names import DISPLAY_METHODS, MIN_TRACE, TOP_DOWN  # noqa: E402

WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
REPORTS = ROOT / "reports"
FORECASTS_PATH = ROOT / "data" / "forecasts" / "reconciled_forecasts.parquet"

HISTORY_DAYS = 120
RECONCILIATION_COLUMNS = {
    "LightGBM/BottomUpSparse": "Bottom-up",
    f"LightGBM/{TOP_DOWN}": "Top-down",
    f"LightGBM/{MIN_TRACE}": "MinTrace",
}

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")


@st.cache_resource
def warehouse_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(WAREHOUSE), read_only=True, config={"memory_limit": "1GB", "threads": 2})


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    df = pd.read_parquet(FORECASTS_PATH)
    df["ds"] = pd.to_datetime(df["ds"])
    df["depth"] = df["unique_id"].str.count("/")
    return df


@st.cache_data
def load_report(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_history(unique_id: str) -> pd.DataFrame:
    """Daily actual sales for one hierarchy id such as CA/CA_1/FOODS/FOODS_3."""
    columns = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
    parts = unique_id.split("/")
    where = " AND ".join(f"{col} = ?" for col in columns[: len(parts)])
    df = warehouse_connection().execute(
        f"""
        SELECT sale_date AS ds, CAST(SUM(units) AS DOUBLE) AS y
        FROM raw_sales_enriched
        WHERE {where}
          AND sale_date > (SELECT MAX(sale_date) FROM raw_sales_enriched) - INTERVAL '{HISTORY_DAYS + 28} days'
        GROUP BY ds
        ORDER BY ds
        """,
        parts,
    ).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def select_series(forecasts: pd.DataFrame, key: str) -> str:
    """Hierarchy selector: returns the chosen unique_id."""
    level = st.radio(
        "Level", ["State", "Store", "Department", "SKU"], horizontal=True, key=f"{key}_level"
    )
    if level == "State":
        return forecasts.loc[forecasts["depth"] == 0, "unique_id"].iloc[0]

    stores = sorted(forecasts.loc[forecasts["depth"] == 1, "unique_id"])
    store = st.selectbox("Store", stores, format_func=lambda s: s.split("/")[-1], key=f"{key}_store")
    if level == "Store":
        return store

    series = forecasts.loc[forecasts["depth"] == 3, "unique_id"]
    departments = sorted(series[series.str.startswith(store + "/")].unique())
    department = st.selectbox(
        "Department", departments, format_func=lambda s: s.split("/")[-1], key=f"{key}_dept"
    )
    if level == "Department":
        return department

    series = forecasts.loc[forecasts["depth"] == 4, "unique_id"]
    items = sorted(series[series.str.startswith(department + "/")].unique())
    return st.selectbox("SKU", items, format_func=lambda s: s.split("/")[-1], key=f"{key}_sku")


def forecast_chart(forecasts: pd.DataFrame, unique_id: str, columns: dict, height: int = 430) -> None:
    """Actual sales with one line per forecast column, over the 28-day test window."""
    history = load_history(unique_id)
    rows = forecasts[forecasts["unique_id"] == unique_id].sort_values("ds")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], name="Actual sales", line=dict(color="#94a3b8")))
    for column, label in columns.items():
        fig.add_trace(go.Scatter(x=rows["ds"], y=rows[column], name=label, line=dict(dash="dash", width=2)))
    fig.add_vline(x=rows["ds"].min(), line=dict(color="#94a3b8", dash="dot"))
    fig.update_layout(
        height=height, hovermode="x unified", yaxis_title="Units per day",
        legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


st.title("Retail Demand Forecasting")
st.caption(
    "Walmart M5 data, California. Every forecast covers a 28-day test window that was held out "
    "from training. The dotted line marks where the test window starts."
)

if not FORECASTS_PATH.exists() or not WAREHOUSE.exists():
    st.warning("No forecasts found. Run the pipeline first.")
    st.stop()

forecasts = load_forecasts()
explorer, comparison, accuracy, inventory = st.tabs([
    "Forecast Explorer", "Reconciliation Comparison", "Accuracy Report", "Inventory Simulation",
])

with explorer:
    unique_id = select_series(forecasts, "explorer")
    available = [c for c in DISPLAY_METHODS if c in forecasts.columns]
    labels = {DISPLAY_METHODS[c]: c for c in available}
    chosen = st.multiselect(
        "Forecasts to show",
        list(labels),
        default=[DISPLAY_METHODS[f"LightGBM/{MIN_TRACE}"], DISPLAY_METHODS["SeasonalNaive"]],
    )
    forecast_chart(forecasts, unique_id, {labels[name]: name for name in chosen})

with comparison:
    st.markdown("All three methods start from the same LightGBM forecasts, and all three add up across levels.")
    unique_id = select_series(forecasts, "comparison")
    forecast_chart(forecasts, unique_id, RECONCILIATION_COLUMNS)

    rows = forecasts[forecasts["unique_id"] == unique_id]
    totals = pd.Series({label: rows[col].sum() for col, label in RECONCILIATION_COLUMNS.items()})
    actual_total = load_history(unique_id).merge(rows[["ds"]].drop_duplicates(), on="ds")["y"].sum()
    table = pd.DataFrame({"28-day forecast (units)": totals.round(0)})
    table["Difference from actual (units)"] = (totals - actual_total).round(0)
    st.markdown(f"Actual sales over the 28 days: **{actual_total:,.0f} units**")
    st.dataframe(table, use_container_width=True)

with accuracy:
    results = load_report("results.csv")
    wrmsse = load_report("wrmsse.csv")
    backtest = load_report("backtest_wrmsse.csv")
    if results is None or wrmsse is None:
        st.info("Run the evaluation step to see accuracy results.")
    else:
        st.subheader("M5-style WRMSSE (lower is better)")
        st.caption(
            "California only, so this isn't the official all-state M5 score. AutoETS and the other "
            "classical baselines were not fitted at SKU level, so they have no 12-level score."
        )
        table = wrmsse.sort_values("wrmsse_9_aggregate_levels").set_index("method")
        table = table.map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
        table.index.name = "Method"
        st.dataframe(
            table.rename(columns={
                "wrmsse_12_levels": "12 levels", "wrmsse_9_aggregate_levels": "9 aggregate levels",
            }),
            use_container_width=True,
        )

        st.subheader("By hierarchy level")
        metric_names = {"mean_rmsse": "Mean RMSSE", "mean_mae": "Mean MAE", "mean_bias": "Mean bias"}
        metric = st.radio("Metric", list(metric_names), format_func=metric_names.get, horizontal=True)
        order = ["total", "state", "store", "category", "dept"]
        pivot = results.pivot(index="method", columns="level", values=metric)[order]
        st.dataframe(pivot.style.format("{:.3f}" if metric == "mean_rmsse" else "{:,.1f}"), use_container_width=True)

        if backtest is not None:
            st.subheader("Across evaluation windows")
            st.caption("Window 0 is the final test window; windows 1 and 2 are earlier, non-overlapping periods.")
            shown = backtest[~backtest["method"].isin(["Naive", "HistoricAverage"])]
            fig = px.line(
                shown.sort_values("window", ascending=False).assign(window=lambda d: d["window"].astype(str)),
                x="window", y="wrmsse_9_aggregate_levels", color="method", markers=True,
                category_orders={"window": ["2", "1", "0"]},
                labels={"window": "Window (earliest to latest)", "wrmsse_9_aggregate_levels": "WRMSSE, 9 aggregate levels"},
            )
            fig.update_layout(height=420, margin=dict(t=20, b=30))
            st.plotly_chart(fig, use_container_width=True)

with inventory:
    sim = load_report("inventory_simulation.csv")
    if sim is None:
        st.info("Run the evaluation step to see the inventory simulation.")
    else:
        st.caption(
            "Simulated on California daily totals with assumed costs: \\$3.50 unit price, \\$1.05 per unit "
            "of lost margin, 0.5% of price per unit per day to hold. Safety stock uses z = 1.65 and "
            "error sizes from the window before the test window."
        )
        sim = sim.sort_values("total_cost")
        fig = go.Figure([
            go.Bar(y=sim["method"], x=sim["stockout_cost"], name="Stockout cost", orientation="h"),
            go.Bar(y=sim["method"], x=sim["holding_cost"], name="Holding cost", orientation="h"),
        ])
        fig.update_layout(
            barmode="stack", height=400, margin=dict(t=20, b=30),
            xaxis_title="Simulated cost ($)", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sim[["method", "stockout_cost", "holding_cost", "total_cost", "fill_rate"]]
            .rename(columns={
                "method": "Method", "stockout_cost": "Stockout cost", "holding_cost": "Holding cost",
                "total_cost": "Total cost", "fill_rate": "Fill rate",
            }).set_index("Method").style.format({
                "Stockout cost": "${:,.0f}", "Holding cost": "${:,.0f}",
                "Total cost": "${:,.0f}", "Fill rate": "{:.1%}",
            }),
            use_container_width=True,
        )
