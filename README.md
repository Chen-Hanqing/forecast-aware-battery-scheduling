# Forecast-aware battery scheduling

An end-to-end, repeatable extension of Gurobi's battery scheduling example. Instead of assuming load and electricity prices are known, it:

1. selects a forecasting model using rolling-origin validation (`seasonal_naive` or gradient boosting with calendar and lag features);
2. bootstraps joint forecast residuals into load/price scenarios;
3. solves a scenario-based linear program that minimizes expected grid cost plus CVaR risk; and
4. evaluates the resulting dispatch using realised data in a rolling backtest.

The default solver is SciPy/HiGHS, so no commercial solver is required. The formulation preserves the original example's behind-the-meter balance and battery dynamics. It supports an optional PV column; omit it to assume zero PV.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
battery-schedule make-demo-data --output data/raw/history.csv
battery-schedule run --config configs/default.yaml
pytest
```

Outputs are written under `artifacts/`: selected model report, next-day forecast, scenario schedules, and a rolling backtest report. To run daily in production, schedule `battery-schedule run --config ...` after ingesting the latest validated CSV. A container recipe and GitHub Actions checks are included.

## Input contract

CSV must contain `timestamp` (timezone-aware or UTC), `load_kw`, and `import_price_eur_kwh`. It may contain `pv_kw` and `export_price_eur_kwh`. The rows must be hourly, regular, and have no duplicate timestamps.

## Design choices

Gradient boosting is suitable for load/price profiles with nonlinear hour-of-week patterns and lagged autocorrelation; seasonal naive is retained as a strong, transparent baseline. Selection uses only past observations at each validation origin, avoiding leakage. Uncertainty is represented by **joint** residual bootstrapping, retaining load-price co-movement. Optimization uses a single here-and-now battery schedule across scenarios and penalizes high-cost outcomes through CVaR.
