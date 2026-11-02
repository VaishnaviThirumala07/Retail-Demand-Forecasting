# DS31 — Retail Demand Forecasting with Hierarchical Time Series

.PHONY: all setup sample-data ingest dbt-run features baselines train reconcile backtest evaluate app clean test help

PYTHON := python
STREAMLIT := streamlit
DBT := dbt

help:
	@echo "DS31 Retail Demand Forecasting — Available targets:"
	@echo ""
	@echo "  make setup      Install dependencies"
	@echo "  make sample-data Generate small synthetic M5 CSVs (used by CI)"
	@echo "  make ingest     Load M5 CSVs into DuckDB warehouse"
	@echo "  make dbt-run    Build dbt staging + mart models with tests"
	@echo "  make features   Engineer lag/rolling/price features"
	@echo "  make baselines  Compute naive + statsforecast baselines"
	@echo "  make train      Train LightGBM global model"
	@echo "  make reconcile  Run Bottom-Up, Top-Down and MinTrace"
	@echo "  make backtest   Evaluate on two earlier 28-day windows"
	@echo "  make evaluate   Compute WRMSSE + per-level metrics + cost sim"
	@echo "  make app        Launch Streamlit dashboard"
	@echo "  make all        Run full pipeline end-to-end"
	@echo "  make test       Run pytest test suite"
	@echo "  make clean      Remove generated artifacts"
	@echo ""

setup:
	pip install -r requirements.txt
	$(DBT) deps --project-dir dbt_m5

sample-data:
	$(PYTHON) scripts/make_sample_data.py

ingest:
	$(PYTHON) src/ingest.py

dbt-run:
	$(DBT) build --project-dir dbt_m5 --profiles-dir dbt_m5

features:
	$(PYTHON) src/features.py

baselines:
	$(PYTHON) src/baselines.py

train:
	$(PYTHON) src/train_lgbm.py

reconcile:
	$(PYTHON) src/reconcile.py

backtest:
	$(PYTHON) src/backtest.py --window 2
	$(PYTHON) src/backtest.py --window 1

evaluate:
	$(PYTHON) src/metrics.py
	$(PYTHON) src/inventory_sim.py
	$(PYTHON) src/backtest.py --summarize

app:
	$(STREAMLIT) run dashboard/app.py

test:
	$(PYTHON) -m pytest tests/ -v

all: ingest dbt-run features baselines train reconcile backtest evaluate
	@echo ""
	@echo "========================================="
	@echo " DS31 Pipeline Complete!"
	@echo " Next steps:"
	@echo "   make app    → Launch Streamlit dashboard"
	@echo "   make test   → Run test suite"
	@echo "========================================="

clean:
ifeq ($(OS),Windows_NT)
	@if exist warehouse\m5.duckdb del /q warehouse\m5.duckdb
	@if exist data\features rmdir /s /q data\features
	@if exist data\forecasts rmdir /s /q data\forecasts
	@if exist models rmdir /s /q models
	@if exist reports\results.csv del /q reports\results.csv
	@if exist reports\inventory_simulation.csv del /q reports\inventory_simulation.csv
	@if exist reports\feature_importance.csv del /q reports\feature_importance.csv
	@echo Cleaned generated artifacts.
else
	rm -f warehouse/m5.duckdb
	rm -rf data/features data/forecasts models
	rm -f reports/results.csv reports/inventory_simulation.csv reports/feature_importance.csv
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleaned generated artifacts."
endif
