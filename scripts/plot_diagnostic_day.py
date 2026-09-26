"""Plot actual vs. forecast price for one backtest day, with the focus model's own top-4 and
bottom-4 hours marked next to the real ones.

Reads the per-day forecast and actual price vectors stored in a completed run's metrics file
(pipeline._backtest()'s `daily` records), so nothing is refitted. By default it picks the day
on which the focus model's Kendall's tau was lowest.

Usage:
    python -m scripts.plot_diagnostic_day --metrics results/household/run_metrics_real_de.json --focus lasso_ar
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.style import COLORS, GRID, INK, INK_MUTED, LABELS, MODEL_ORDER

IMAGES = Path("docs/images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/household/run_metrics_real_de.json")
    parser.add_argument("--focus", default="lasso_ar", help="candidate whose top/bottom-4 hours get marked")
    parser.add_argument("--date", default=None, help="backtest day; default = the focus model's lowest-tau day")
    args = parser.parse_args()

    daily = json.loads(Path(args.metrics).read_text())["backtest"]["daily"]
    focus_rows = [d for d in daily if d["candidate"] == args.focus and d["kendall_tau"] is not None]
    date = args.date or min(focus_rows, key=lambda d: d["kendall_tau"])["date"]
    day = {d["candidate"]: d for d in daily if d["date"] == date}
    actual = np.array(day[args.focus]["actual_price_eur_kwh"]) * 100
    forecasts = {n: np.array(day[n]["forecast_price_eur_kwh"]) * 100 for n in MODEL_ORDER if n in day}
    hours = np.arange(len(actual))

    def top_bottom(series: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(series)
        return order[-k:], order[:k]

    actual_top, actual_bottom = top_bottom(actual)
    focus_top, focus_bottom = top_bottom(forecasts[args.focus])

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(hours, actual, color=INK, linewidth=3.0, label="Actual", zorder=5)
    for name, forecast in forecasts.items():
        stats = f"τ={day[name]['kendall_tau']:.2f}, MAE={day[name]['mae_price_eur_kwh'] * 100:.2f} ct"
        focus = name == args.focus
        ax.plot(hours, forecast, "-" if focus else ":", color=COLORS[name], linewidth=2.4 if focus else 1.5,
                label=f"{LABELS[name]} ({stats})", zorder=4)
    ax.scatter(actual_top, actual[actual_top], marker="*", s=230, color="#e34948", edgecolor=INK, linewidth=0.6,
               zorder=6, label="Actual top-4 (most expensive)")
    ax.scatter(actual_bottom, actual[actual_bottom], marker="*", s=230, color="#ffffff", edgecolor=INK, linewidth=1.0,
               zorder=6, label="Actual bottom-4 (cheapest)")
    ax.scatter(focus_top, forecasts[args.focus][focus_top], marker="^", s=90, color=COLORS[args.focus],
               edgecolor=INK, linewidth=0.6, zorder=6, label=f"{LABELS[args.focus]} top-4")
    ax.scatter(focus_bottom, forecasts[args.focus][focus_bottom], marker="v", s=90, color=COLORS[args.focus],
               edgecolor=INK, linewidth=0.6, zorder=6, label=f"{LABELS[args.focus]} bottom-4")

    ax.set_xlabel("Hour of day", color=INK_MUTED)
    ax.set_ylabel("Day-ahead price (ct/kWh)", color=INK_MUTED)
    ax.set_xticks(hours[::3])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    tau = day[args.focus]["kendall_tau"]
    ax.set_title(f"{date}: {LABELS[args.focus]}'s lowest-τ day (τ={tau:.2f})", fontsize=12, color=INK, loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    out = IMAGES / f"diagnostic_day_{args.focus}_{date}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    overlap_top = len(set(actual_top) & set(focus_top))
    overlap_bottom = len(set(actual_bottom) & set(focus_bottom))
    print(f"{args.focus}: {overlap_top}/4 actual-expensive hours also in its own top-4, "
          f"{overlap_bottom}/4 actual-cheap hours also in its own bottom-4")


if __name__ == "__main__":
    main()
