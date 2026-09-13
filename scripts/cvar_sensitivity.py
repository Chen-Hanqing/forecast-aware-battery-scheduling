"""CVaR risk-aversion sensitivity sweep on a 30-day backtest, for one named
forecast candidate (no auto-selection: mixing whichever model MAE happened to
pick on a given day would confound "risk-aversion effect" with "which model
ran that day" — see README).

At each backtest origin, the forecast and scenarios for that one candidate
are computed once and reused across every cvar_weight value (only the LP's
risk term changes) — this avoids re-running rolling-origin validation once
per sweep point, so the whole sweep costs about the same as a single backtest.

Usage:
    python -m scripts.cvar_sensitivity --config configs/real_de.yaml --candidate gradient_boosting
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from battery_schedule.config import load_config
from battery_schedule.data import read_history
from battery_schedule.forecast import make_scenarios, select_and_forecast
from battery_schedule.optimise import solve_schedule

CVAR_WEIGHTS = [0.0, 0.2, 0.5, 0.8, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_de.yaml")
    parser.add_argument("--candidate", default="gradient_boosting")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "artifacts_real"))
    history = read_history(config["data"]["history_csv"], config["data"]["frequency"])
    horizon = config["data"]["horizon_steps"]
    days = config["backtest"]["days"]
    start = len(history) - days * horizon
    cfg = config["optimization"]

    costs: dict[float, list[float]] = {w: [] for w in CVAR_WEIGHTS}
    baseline_costs: list[float] = []

    origins = range(start, len(history) - horizon + 1, horizon)
    for i, origin in enumerate(origins):
        train, actual = history.iloc[:origin], history.iloc[origin:origin + horizon]
        validation_days = min(config["forecast"]["validation_days"], max(2, (len(train) - 192) // 24))
        fc = select_and_forecast(train, horizon, validation_days, [args.candidate], config["forecast"].get("holiday_country"), config["forecast"].get("lookback_days"))
        scenarios = make_scenarios(
            fc, min(30, config["scenarios"]["count"]), config["scenarios"]["residual_block_steps"], origin
        )
        exp = actual["export_price_eur_kwh"].fillna(actual.import_price_eur_kwh * cfg.export_price_ratio).to_numpy()

        net_no_battery = actual.load_kw.to_numpy() - actual.pv_kw.to_numpy()
        baseline = (
            np.maximum(net_no_battery, 0) * actual.import_price_eur_kwh.to_numpy()
            - np.maximum(-net_no_battery, 0) * exp
        ).sum()
        baseline_costs.append(float(baseline))

        for w in CVAR_WEIGHTS:
            sched = solve_schedule(scenarios, actual.pv_kw.to_numpy(), exp, cfg, cvar_alpha=cfg.cvar_alpha, cvar_weight=w)
            net = actual.load_kw.to_numpy() - actual.pv_kw.to_numpy() + sched.charge_kw.to_numpy() - sched.discharge_kw.to_numpy()
            realised = np.maximum(net, 0) * actual.import_price_eur_kwh.to_numpy() - np.maximum(-net, 0) * exp
            costs[w].append(float(realised.sum()))

        print(f"origin {i + 1}/{len(origins)} done ({actual.index[0].date()})")

    baseline_total = sum(baseline_costs)
    results = {"candidate": args.candidate, "baseline_no_battery_eur": baseline_total, "days": len(baseline_costs)}
    for w in CVAR_WEIGHTS:
        total = sum(costs[w])
        results[f"cvar_weight_{w}"] = {
            "total_eur": total,
            "vs_baseline_pct": 100 * (total / baseline_total - 1),
        }

    out_path = output_dir / "cvar_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
