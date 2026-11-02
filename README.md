# Retail Demand Forecasting with Hierarchical Time Series

[![CI](https://github.com/VaishnaviThirumala07/Retail-Demand-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/VaishnaviThirumala07/Retail-Demand-Forecasting/actions/workflows/ci.yml)

28-day demand forecasts for Walmart's M5 sales data, with forecasts made coherent across the
store → category → department → SKU hierarchy (each level sums exactly to the level above).

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| Ingest | `src/ingest.py` | Loads the M5 CSVs into DuckDB and unpivots them to 59,181,090 item-store-day rows |
| Model | `dbt_m5/` | dbt staging + mart models with 23 data tests |
| Features | `src/features.py` | Lags ≥ 28 days, rolling stats, price and calendar features (California scope) |
| Baselines | `src/baselines.py` | Naive, SeasonalNaive, HistoricAverage, AutoETS at 9 aggregate levels |
| Train | `src/train_lgbm.py` | One global LightGBM model (Tweedie loss), plus bias factors from its validation window |
| Reconcile | `src/reconcile.py` | Bottom-Up, Top-Down, MinTrace (`wls_struct`) over 12,241 series; fails if forecasts don't add up |
| Backtest | `src/backtest.py` | Repeats training and evaluation on two earlier 28-day windows |
| Evaluate | `src/metrics.py`, `src/inventory_sim.py` | RMSSE, MAE, bias, M5-style WRMSSE, and an inventory cost simulation |
| Serve | `dashboard/app.py`, `api/main.py` | Streamlit dashboard and a FastAPI endpoint over the saved forecasts |

## Results (real M5 data, California)

- **Windows:** the final test window is 2016-04-25 → 2016-05-22. Two earlier, non-overlapping windows are 02-29 → 03-27 and 03-28 → 04-24. Each is held out from training.
- **Training:** LightGBM uses the 180 days before the window, with early stopping on the final 28 of them.
- **Hierarchy:** 1 state, 4 stores, 12 store × category series, 28 store × department series and 12,196 SKU-store series.

**M5-style WRMSSE** (lower is better; dollar-weighted, each level counts equally). This covers California only, so it isn't comparable to official all-state M5 scores. AutoETS wasn't fitted at SKU level, so it has only the 9-level score.

| Method | 12 levels: final window | 12 levels: mean of 3 windows | 9 aggregate levels: final window | 9 aggregate levels: mean of 3 windows |
|--------|------|------|------|------|
| Seasonal naive | 0.683 | 0.752 | 0.560 | 0.643 |
| AutoETS | n/a | n/a | **0.514** | 0.549 |
| LightGBM bottom-up | 0.688 | 0.639 | 0.636 | 0.565 |
| LightGBM top-down | 0.631 | 0.699 | 0.560 | 0.643 |
| LightGBM + MinTrace | 0.620 | 0.656 | 0.546 | 0.588 |
| Bias-corrected bottom-up (global) | 0.623 | **0.623** | 0.550 | **0.544** |
| Bias-corrected MinTrace (global) | **0.609** | 0.650 | 0.531 | 0.579 |

- **Best on the final window:** bias-corrected MinTrace, **10.9% lower 12-level WRMSSE** than seasonal naive (0.609 vs 0.683).
- **Every window:** each LightGBM + MinTrace variant beats seasonal naive on 12-level WRMSSE in all three windows.
- **Bottom-up is the most variable:** it's far ahead in one window (0.502) and behind MinTrace in the other two, so its good average hides large swings.
- **AutoETS is the steadiest at aggregate levels:** across the three windows its 9-level WRMSSE varies by a standard deviation of 0.030, vs 0.051 for LightGBM + MinTrace and 0.157 for bias-corrected bottom-up.

**RMSSE by level, final window** (lower is better; mean over the series at each level):

| Method | Total | Store (4) | Category (3) | Dept (7) |
|--------|-------|-----------|--------------|----------|
| Seasonal naive | 0.495 | 0.562 | 0.545 | 0.829 |
| AutoETS | **0.458** | **0.551** | **0.508** | 0.717 |
| LightGBM bottom-up | 0.596 | 0.697 | 0.644 | 0.765 |
| LightGBM top-down | 0.495 | 0.562 | 0.545 | 0.829 |
| LightGBM + MinTrace | 0.497 | 0.571 | 0.537 | 0.709 |
| Bias-corrected bottom-up (global) | 0.480 | 0.611 | 0.549 | **0.676** |
| Bias-corrected MinTrace (global) | 0.476 | 0.555 | 0.523 | 0.709 |

**Bias correction** multiplies forecasts by actual ÷ predicted sales from the validation window. Global and per-department factors were compared on the two earlier windows, and global was chosen.

- **On the final window:** the global factor is 1.026. It cuts LightGBM bottom-up's mean bias at total level from −1,165 to −715 units/day, and lowers its total-level RMSSE from 0.596 to 0.480.
- **It isn't always a gain:** in the earliest window it over-corrects bottom-up, from −311 to +530 units/day. Most of the final window's under-forecast also remains; seasonal naive and AutoETS under-forecast that window too (−815 and −650).
- **Per-department factors did worse on the final window:** department RMSSE is 0.809 for MinTrace vs 0.709 with the global factor.

**Inventory cost simulation** (final window, CA daily totals, same dates and policy for each method; assumed costs: $3.50 unit price, $1.05 stockout cost per unit, 0.5% per unit per day holding cost). Safety stock is sized from each method's errors in the preceding window.

| Forecast | Stockout cost | Holding cost | Total cost | Fill rate |
|----------|---------------|--------------|------------|-----------|
| **LightGBM + MinTrace** | $8 | $8,415 | **$8,423** | 100.0% |
| Bias-corrected MinTrace (global) | $0 | $9,003 | $9,003 | 100.0% |
| Seasonal naive | $0 | $10,922 | $10,922 | 100.0% |
| LightGBM top-down | $0 | $10,922 | $10,922 | 100.0% |
| Bias-corrected bottom-up (global) | $6,808 | $4,839 | $11,647 | 98.8% |
| LightGBM bottom-up | $13,000 | $3,193 | $16,193 | 97.6% |

- **Reconciliation helps here:** LightGBM + MinTrace costs **22.9% less** than seasonal naive and 48% less than bottom-up.
- **Bias correction helps bottom-up only:** it cuts bottom-up's cost by 28%. MinTrace has almost no stockouts to remove, so correcting it only adds holding cost.
- **Treat these as indicative:** they come from one window and assumed costs, and the near-100% fill rates show safety stock is generous.

Source files are in [`reports/`](reports/): `wrmsse.csv`, `results.csv`, `backtest_wrmsse.csv`, `backtest_results.csv`, `inventory_simulation.csv`, `bias_correction.json`.

## Data insights

From [`sql/analysis/insights_summary.py`](sql/analysis/insights_summary.py), run on all 3 states over 2011–2016:

| Finding | Value |
|---------|-------|
| CA food sales on SNAP days vs other days | **+10.2%** |
| Busiest days (index, average day = 100) | Saturday **121**, Sunday **120**; Wednesday lowest at **87** |
| Revenue share of the top 20% of items | **59%** |
| Item-store series with more than 50% zero-sales days | **78%** |
| Super Bowl Sunday, FOODS_3 vs nearby Sundays | **+6.9%** |
| Christmas Day sales | **0%** (stores closed) |

## Checks

- The dbt build passes all 23 data tests on the real data.
- The reconcile step checks that store and SKU forecasts sum to the total for every method, and fails if they don't.
- 19 unit tests (`tests/`) cover the metric maths, WRMSSE, bias factors, the API, lag/rolling-window logic and coherence on toy data.
- CI runs on every push: the full pipeline on synthetic M5-format data, the tests, and a Docker build with an API `/health` check.

## How to run

Put the [M5 files](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) in `data/raw/`:
`sales_train_evaluation.csv` (116 MB), `calendar.csv`, `sell_prices.csv` (194 MB).

```bash
make setup
make all
make test
```

**On Windows, without `make`:** `scripts/run_pipeline.ps1` runs each step as a separate process and stops a step
if it exceeds its memory limit. The settings below fit a 16 GB laptop; the largest step peaked at 2.2 GB.

```bash
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -HistoryDays 180 -MaxStepGB 2.5 -MinFreeGB 1.5
```

**Without the M5 files:** `make sample-data` writes a small synthetic dataset in the same format. This is what CI uses.

**Dashboard** (after running the pipeline): forecast explorer, reconciliation comparison, accuracy report and inventory simulation.

```bash
python -m streamlit run dashboard/app.py
```

**API** (serves saved forecasts; nothing is trained live):

```bash
python -m uvicorn api.main:app --port 8000
```

`GET /forecast/FOODS_3_090` returns that item's 28-day forecast for each store. Optional parameters: `store_id`, and `method` (default `LightGBM + MinTrace`).

## Limitations

- Only California is modelled, so WRMSSE is M5-style rather than the official all-state score. Running all states needs more memory than a 16 GB laptop has.
- LightGBM was trained on 180 days of history to fit in memory, and still under-forecasts in the final window.
- The final window had already been inspected in earlier versions of this project, so it isn't fully untouched. The bias-correction variant was chosen using only the earlier windows.
- Bias factors come from the same window used for LightGBM's early stopping, so they may be slightly optimistic.
- Three windows from early 2016 can't show how the methods behave in other seasons, such as December.
- MinTrace uses `wls_struct`. `mint_shrink` needs more memory than is available here.
- Inventory costs are assumptions, not real Walmart figures, and the simulation runs on state daily totals rather than per SKU.
- 20.8% of item-days have no sell price in the data.
- The dashboard and API run locally and aren't deployed. `docker compose` hasn't been tested with the current outputs.
