"""Plot actual vs. forecast price for one specific backtest day, with each
series' own top-4 (most expensive) and bottom-4 (cheapest) hours marked.

Built to answer one question directly: which hours cause LEAR's Kendall's tau
to collapse on its worst day in the real_de backtest (see
results/tau_scan_lear.json, produced by scripts/diagnose_tau.py)? A battery's
arbitrage decision depends on getting these top/bottom hours right, not on
average error, so this is the hour-level view behind README.md's tau-vs-MAE
discussion.

Usage:
    python -m scripts.plot_diagnostic_day --config configs/real_de.yaml --date 2018-01-15
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from battery_schedule.config import load_config
from battery_schedule.data import read_history
from battery_schedule.forecast import select_and_forecast

IMAGES = "docs/images"
LABELS = {"seasonal_naive": "Seasonal naive", "gradient_boosting": "Gradient boosting", "lear": "LEAR (Lasso)"}
COLORS = {"seasonal_naive": "#7f7f7f", "gradient_boosting": "#2ca02c", "lear": "#1f77b4"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_de.yaml")
    parser.add_argument("--date", default="2018-01-15", help="date of the backtest day to diagnose")
    parser.add_argument("--focus", default="lear", help="candidate whose top/bottom-4 hours get marked")
    args = parser.parse_args()

    config = load_config(args.config)
    history = read_history(config["data"]["history_csv"], config["data"]["frequency"])
    horizon = config["data"]["horizon_steps"]
    days = config["backtest"]["days"]
    holiday_country = config["forecast"].get("holiday_country")
    lookback_days = config["forecast"].get("lookback_days")

    start = len(history) - days * horizon
    origins = list(range(start, len(history) - horizon + 1, horizon))
    origin = next(o for o in origins if str(history.index[o].date()) == args.date)
    train, actual = history.iloc[:origin], history.iloc[origin:origin + horizon]
    actual_price = actual["import_price_eur_kwh"].to_numpy()
    hours = np.arange(horizon)

    forecasts: dict[str, np.ndarray] = {}
    stats: dict[str, tuple[float, float]] = {}
    for name in LABELS:
        validation_days = min(config["forecast"]["validation_days"], max(2, (len(train) - 192) // 24))
        fc = select_and_forecast(train, horizon, validation_days, [name], holiday_country, lookback_days)
        pred = fc.mean["import_price_eur_kwh"].to_numpy()
        forecasts[name] = pred
        tau, _ = kendalltau(pred, actual_price)
        mae = float(np.mean(np.abs(pred - actual_price)))
        stats[name] = (float(tau), mae)

    def top_bottom(series: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(series)
        return order[-k:], order[:k]

    actual_top, actual_bottom = top_bottom(actual_price)
    focus_top, focus_bottom = top_bottom(forecasts[args.focus])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hours, actual_price, color="#111111", linewidth=2.5, label="Actual", zorder=5)
    for name, label in LABELS.items():
        style = "--" if name == args.focus else ":"
        width = 2.0 if name == args.focus else 1.2
        tau, mae = stats[name]
        ax.plot(hours, forecasts[name], style, color=COLORS[name], linewidth=width,
                 label=f"{label} (τ={tau:.2f}, MAE={mae:.4f})")

    ax.scatter(actual_top, actual_price[actual_top], marker="*", s=220, color="#d62728",
               edgecolor="black", linewidth=0.6, zorder=6, label="Actual top-4 (most expensive)")
    ax.scatter(actual_bottom, actual_price[actual_bottom], marker="*", s=220, color="#2ca02c",
               edgecolor="black", linewidth=0.6, zorder=6, label="Actual bottom-4 (cheapest)")
    ax.scatter(focus_top, forecasts[args.focus][focus_top], marker="^", s=110, color=COLORS[args.focus],
               edgecolor="black", linewidth=0.6, zorder=6, label=f"{LABELS[args.focus]} top-4 (thinks expensive)")
    ax.scatter(focus_bottom, forecasts[args.focus][focus_bottom], marker="v", s=110, color=COLORS[args.focus],
               edgecolor="black", linewidth=0.6, zorder=6, label=f"{LABELS[args.focus]} bottom-4 (thinks cheap)")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day-ahead price (EUR/kWh)")
    ax.set_xticks(hours[::2])
    ax.set_title(f"Why {LABELS[args.focus]}'s ranking breaks down on {args.date} "
                 f"(τ={stats[args.focus][0]:.2f}, its worst day in this backtest)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = f"{IMAGES}/diagnostic_day_{args.focus}_{args.date}.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    for name, (tau, mae) in stats.items():
        print(f"{name}: tau={tau:.3f} mae={mae:.4f}")
    overlap_top = len(set(actual_top) & set(focus_top))
    overlap_bottom = len(set(actual_bottom) & set(focus_bottom))
    print(f"{args.focus}: {overlap_top}/4 actual-expensive hours also in its own top-4, "
          f"{overlap_bottom}/4 actual-cheap hours also in its own bottom-4")


if __name__ == "__main__":
    main()
