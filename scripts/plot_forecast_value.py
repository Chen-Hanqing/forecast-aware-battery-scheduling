"""Plot a per-candidate forecast-value comparison from a completed run's
run_metrics.json (produced by `battery-schedule run`, backtest.enabled: true):
which forecast candidate actually makes the battery money, and whether MAE or
rank correlation (Kendall's tau) predicts that better.

Usage:
    battery-schedule run --config configs/real_de.yaml
    python -m scripts.plot_forecast_value --metrics artifacts_real/run_metrics.json --suffix real
    python -m scripts.plot_forecast_value --metrics artifacts_counterfactual/run_metrics.json --suffix counterfactual --title-suffix " — counterfactual: 2022-23 crisis-era prices"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.style import COLORS, LABELS

IMAGES = Path("docs/images")



def plot_value_bars(backtest: dict, suffix: str, title_suffix: str) -> None:
    names = [n for n in LABELS if n in backtest["candidates"]]
    bars = [("No battery", backtest["baseline_no_battery_eur"], "#999999")]
    bars += [(LABELS[n], backtest["candidates"][n]["realised_grid_cost_eur"], COLORS[n]) for n in names]
    bars += [("Perfect foresight\n(oracle)", backtest["oracle_perfect_foresight_eur"], "#d62728")]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel(f"Realised grid cost, {backtest['days']}-day backtest (EUR)")
    ax.set_title(f"What a better forecast is worth to the battery{title_suffix}", fontsize=11, wrap=True)
    for x, (_, v, _) in zip(xs, bars):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 4 if v >= 0 else -12), ha="center", fontsize=9)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(backtest["baseline_no_battery_eur"], color="#999999", linestyle="--", linewidth=1, alpha=0.6)
    fig.tight_layout()
    fig.savefig(IMAGES / f"forecast_value_bars_{suffix}.png", dpi=150)
    plt.close(fig)


def plot_mae_vs_tau(backtest: dict, suffix: str, title_suffix: str) -> None:
    names = [n for n in LABELS if n in backtest["candidates"]]
    # Labels sit above a point, except the two lowest-cost-neighbours that would collide.
    dy = {"gradient_boosting": -14}
    fig, (ax_mae, ax_tau) = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True)

    for ax, key in ((ax_mae, "mean_mae_price_eur_kwh"), (ax_tau, "mean_kendall_tau_price")):
        for name in names:
            r = backtest["candidates"][name]
            ax.scatter(r[key], r["vs_baseline_pct"], s=90, color=COLORS[name], zorder=3)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.margins(x=0.18, y=0.12)
    ax_mae.invert_xaxis()  # so "better by this metric" reads left-to-right on both panels

    for ax, key in ((ax_mae, "mean_mae_price_eur_kwh"), (ax_tau, "mean_kendall_tau_price")):
        fig.canvas.draw()
        for name in names:
            r = backtest["candidates"][name]
            frac = ax.transAxes.inverted().transform(ax.transData.transform((r[key], r["vs_baseline_pct"])))[0]
            ha, dx = ("left", 8) if frac < 0.3 else ("right", -8) if frac > 0.7 else ("center", 0)
            ax.annotate(LABELS[name], (r[key], r["vs_baseline_pct"]), textcoords="offset points",
                        xytext=(dx, dy.get(name, 9)), fontsize=8.5, ha=ha)

    ax_mae.set_xlabel("Price MAE (EUR/kWh), lower is more accurate")
    ax_tau.set_xlabel("Kendall's tau, forecast vs. realised price ranking, higher is better")
    ax_mae.set_ylabel("Realised grid cost vs. no-battery baseline (%)")
    ax_mae.set_title("Price MAE", fontsize=10)
    ax_tau.set_title("Kendall's tau", fontsize=10)
    fig.suptitle(f"Point accuracy vs. rank accuracy as indicators of battery profit{title_suffix}", y=1.0)
    fig.tight_layout()
    fig.savefig(IMAGES / f"mae_vs_tau_scatter_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts_real/run_metrics.json")
    parser.add_argument("--suffix", default="real")
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    backtest = metrics["backtest"]
    IMAGES.mkdir(parents=True, exist_ok=True)
    plot_value_bars(backtest, args.suffix, args.title_suffix)
    plot_mae_vs_tau(backtest, args.suffix, args.title_suffix)
    print(f"Wrote {IMAGES / f'forecast_value_bars_{args.suffix}.png'}")
    print(f"Wrote {IMAGES / f'mae_vs_tau_scatter_{args.suffix}.png'}")


if __name__ == "__main__":
    main()
