import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import bottom_level, level_frame, score_series, wrmsse_by_level  # noqa: E402
from train_lgbm import bias_factors  # noqa: E402


def test_bias_factors_global_and_per_department():
    val = pd.DataFrame({"dept_id": ["A", "A", "B"], "units": [10.0, 10.0, 30.0]})
    factors = bias_factors(val, np.array([8.0, 8.0, 34.0]))
    assert factors["global"] == pytest.approx(50 / 50)
    assert factors["dept"]["A"] == pytest.approx(20 / 16)
    assert factors["dept"]["B"] == pytest.approx(30 / 34)


def test_wrmsse_hand_computed():
    """Two series: errors of 2 and 0, scales 4 and 1, weights 0.25 and 0.75."""
    frame = pd.DataFrame({
        "key": ["a", "a", "b", "b"],
        "y": [10.0, 10.0, 5.0, 5.0],
        "m": [12.0, 12.0, 5.0, 5.0],
    })
    scale = pd.DataFrame({"scale": [4.0, 1.0], "weight": [0.25, 0.75]}, index=["a", "b"])
    # RMSSE a = sqrt(4 / 4) = 1, RMSSE b = 0, so the level score is 0.25 * 1 + 0.75 * 0.
    assert wrmsse_by_level({"L": frame}, "m", {"L": scale})["L"] == pytest.approx(0.25)


def test_series_that_never_sold_are_excluded():
    frame = pd.DataFrame({"key": ["a", "b"], "y": [1.0, 1.0], "m": [2.0, 2.0]})
    scale = pd.DataFrame({"scale": [1.0, np.nan], "weight": [1.0, 0.0]}, index=["a", "b"])
    assert list(score_series(frame, "m", scale).index) == ["a"]


def test_level_frame_keys_match_scale_keys():
    bottom = pd.DataFrame({
        "store_id": ["CA_1", "CA_1"], "cat_id": ["FOODS", "HOBBIES"],
        "ds": pd.to_datetime(["2016-01-01", "2016-01-01"]),
        "y": [1.0, 2.0], "m": [3.0, 4.0],
    })
    assert sorted(level_frame(bottom, ["store_id", "cat_id"], ["m"])["key"]) == ["CA_1/FOODS", "CA_1/HOBBIES"]
    total = level_frame(bottom, [], ["m"])
    assert total["key"].tolist() == ["total"] and total["m"].tolist() == [7.0]


def test_bottom_level_keeps_only_sku_rows():
    recon = pd.DataFrame({
        "unique_id": ["CA", "CA/CA_1/FOODS/FOODS_1/FOODS_1_001"],
        "ds": ["2016-01-01", "2016-01-01"],
    })
    out = bottom_level(recon)
    assert len(out) == 1 and out.iloc[0]["item_id"] == "FOODS_1_001" and out.iloc[0]["store_id"] == "CA_1"


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import api.main as api

    column = f"LightGBM/{api.MIN_TRACE}"
    path = tmp_path / "reconciled.parquet"
    pd.DataFrame({
        "unique_id": ["CA", "CA/CA_1/FOODS/FOODS_1/FOODS_1_001", "CA/CA_2/FOODS/FOODS_1/FOODS_1_001"],
        "ds": pd.to_datetime(["2016-05-01"] * 3),
        column: [100.0, 1.5, 2.5],
    }).to_parquet(path)
    monkeypatch.setattr(api, "FORECASTS_PATH", path)
    return TestClient(api.app)


def test_forecast_endpoint_returns_each_store(api_client):
    body = api_client.get("/forecast/FOODS_1_001").json()
    assert body["stores"]["CA_1"] == [{"date": "2016-05-01", "forecast": 1.5}]
    assert set(body["stores"]) == {"CA_1", "CA_2"}


def test_forecast_endpoint_filters_by_store(api_client):
    assert set(api_client.get("/forecast/FOODS_1_001?store_id=CA_2").json()["stores"]) == {"CA_2"}


def test_forecast_endpoint_errors(api_client):
    assert api_client.get("/forecast/NOPE").status_code == 404
    assert api_client.get("/forecast/FOODS_1_001?method=bad").status_code == 422
