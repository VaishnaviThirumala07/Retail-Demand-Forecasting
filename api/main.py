"""Serves the pipeline's saved 28-day forecasts. Nothing is trained or predicted live."""
import sys
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method_names import DISPLAY_METHODS, MIN_TRACE  # noqa: E402

FORECASTS_PATH = ROOT / "data" / "forecasts" / "reconciled_forecasts.parquet"
METHODS_BY_LABEL = {label: column for column, label in DISPLAY_METHODS.items()}
DEFAULT_METHOD = DISPLAY_METHODS[f"LightGBM/{MIN_TRACE}"]

app = FastAPI(title="Retail Demand Forecasting API")


@app.get("/health")
async def health_check():
    return {"status": "ok", "forecasts_available": FORECASTS_PATH.exists()}


@app.get("/forecast/{item_id}")
def get_forecast(
    item_id: str,
    store_id: str | None = Query(None, description="Limit to one store, e.g. CA_1"),
    method: str = Query(DEFAULT_METHOD, description="Forecasting method"),
):
    """Return an item's saved 28-day daily forecast for each store (or one store)."""
    if method not in METHODS_BY_LABEL:
        raise HTTPException(status_code=422, detail=f"Unknown method. Choose from: {list(METHODS_BY_LABEL)}")
    if not FORECASTS_PATH.exists():
        raise HTTPException(status_code=503, detail="Forecasts not available. Run the pipeline first.")

    # The column name comes from the allow-list above; ids are bound as parameters.
    column = METHODS_BY_LABEL[method]
    con = duckdb.connect(config={"memory_limit": "256MB", "threads": 1})
    rows = con.execute(
        f"""
        SELECT split_part(unique_id, '/', 2) AS store_id, CAST(ds AS DATE) AS date, "{column}" AS forecast
        FROM read_parquet(?)
        WHERE len(string_split(unique_id, '/')) = 5
          AND split_part(unique_id, '/', 5) = ?
          AND (? IS NULL OR split_part(unique_id, '/', 2) = ?)
        ORDER BY store_id, date
        """,
        [FORECASTS_PATH.as_posix(), item_id, store_id, store_id],
    ).fetchall()
    con.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No forecast for item '{item_id}'")

    stores: dict[str, list] = {}
    for store, date, forecast in rows:
        stores.setdefault(store, []).append({"date": date.isoformat(), "forecast": round(forecast, 3)})
    return {"item_id": item_id, "method": method, "stores": stores}
