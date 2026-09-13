"""Find the backtest day where a given candidate's Kendall's tau is worst, and
save per-day tau for a sample of origins to results/tau_scan_<candidate>.json.

This only fits the ONE candidate requested (default: lear) at each sampled
origin, not all three — much cheaper than the full per-candidate backtest,
for exploring "which day is most diagnostic" before committing to a full plot.

Writes results/tau_scan_<candidate>.json after EVERY origin, not just at the
end — results/ is git-tracked (unlike artifacts_*/), so this and other
expensive-to-recompute outputs (run_metrics.json copies, etc.) survive an
interrupted run and don't need re-running to inspect or reference later.

Usage:
    python -m scripts.diagnose_tau --config configs/real_de.yaml --candidate lear
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from battery_schedule.config import load_config
from battery_schedule.data import read_history
from battery_schedule.forecast import select_and_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_de.yaml")
    parser.add_argument("--candidate", default="lear")
    parser.add_argument("--stride", type=int, default=1, help="only test every Nth backtest origin")
    args = parser.parse_args()

    config = load_config(args.config)
    history = read_history(config["data"]["history_csv"], config["data"]["frequency"])
    horizon = config["data"]["horizon_steps"]
    days = config["backtest"]["days"]
    start = len(history) - days * horizon
    holiday_country = config["forecast"].get("holiday_country")
    lookback_days = config["forecast"].get("lookback_days")

    out_path = Path("results") / f"tau_scan_{args.candidate}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    origins = list(range(start, len(history) - horizon + 1, horizon))[:: args.stride]
    results = []
    for i, origin in enumerate(origins):
        train, actual = history.iloc[:origin], history.iloc[origin:origin + horizon]
        validation_days = min(config["forecast"]["validation_days"], max(2, (len(train) - 192) // 24))
        fc = select_and_forecast(train, horizon, validation_days, [args.candidate], holiday_country, lookback_days)
        actual_price = actual["import_price_eur_kwh"].to_numpy()
        forecast_price = fc.mean["import_price_eur_kwh"].to_numpy()
        tau, _ = kendalltau(forecast_price, actual_price)
        tau = None if tau is None or math.isnan(tau) else float(tau)
        mae = float(np.mean(np.abs(forecast_price - actual_price)))
        results.append({"origin": origin, "date": str(actual.index[0].date()), "tau": tau, "mae": mae})
        print(f"{i + 1}/{len(origins)}  {actual.index[0].date()}  tau={tau}  mae={mae:.4f}", flush=True)
        out_path.write_text(json.dumps(results, indent=2))  # after every origin, not just at the end

    worst = min((r for r in results if r["tau"] is not None), key=lambda r: r["tau"])
    print(f"\nWorst tau: {worst}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
