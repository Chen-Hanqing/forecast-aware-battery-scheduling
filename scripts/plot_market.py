"""Charts for the market-side study, from the stored per-day records of each market's backtest.

  market_revenue_by_year.png     revenue per MW per year by calendar year, one panel per market;
                                 grey bars are the perfect-foresight oracle, lines are the forecasters
  market_cumulative_revenue.png  cumulative revenue per MW over the backtest, so drawdowns and the
                                 regime changes are visible
  market_metric_vs_capture.png   each forecaster's share of oracle revenue against its MAE and its
                                 Kendall's tau, one marker per forecaster and market

Usage:
    python -m scripts.plot_market --metrics de_lu=results/market/run_metrics_market_de_lu.json \\
        nl=results/market/run_metrics_market_nl.json fr=results/market/run_metrics_market_fr.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.market_report import load
from scripts.style import COLORS, GRID, INK, INK_MUTED, LABELS

IMAGES = Path("docs/images")
MARKETS = {"de_lu": "Germany-Luxembourg", "nl": "Netherlands", "fr": "France"}
MARKER = {"de_lu": "o", "nl": "s", "fr": "^"}


def style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True, help="market=path[,path...] pairs, e.g. de_lu=results/....json")
    args = parser.parse_args()
    runs = {}
    for item in args.metrics:
        key, path = item.split("=", 1)
        runs[key] = load(path.split(","))
    IMAGES.mkdir(parents=True, exist_ok=True)

    # 1. revenue by calendar year
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (key, (backtest, dates, per, oracle)) in zip(axes, runs.items()):
        power_mw = backtest["battery"]["power_kw"] / 1000
        years = np.array([d[:4] for d in dates]); unique = sorted(set(years))
        ax.bar(unique, [oracle[years == y].mean() * 365 / power_mw / 1000 for y in unique], color="#d9d8d2", width=0.7,
               label="Perfect foresight (oracle)", zorder=1)
        for name, series in per.items():
            ax.plot(unique, [series["revenue"][years == y].mean() * 365 / power_mw / 1000 for y in unique], marker="o",
                    color=COLORS[name], linewidth=1.9, markersize=5, label=LABELS[name], zorder=3)
        ax.set_title(MARKETS[key], fontsize=11, color=INK, loc="left")
        ax.set_ylabel("Revenue (kEUR per MW per year)", color=INK_MUTED)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("A 1 MW / 2 MWh day-ahead battery, revenue by year", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(IMAGES / "market_revenue_by_year.png", dpi=150)
    plt.close(fig)

    # 2. cumulative revenue
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (key, (backtest, dates, per, oracle)) in zip(axes, runs.items()):
        power_mw = backtest["battery"]["power_kw"] / 1000
        x = np.array(dates, dtype="datetime64[D]")
        ax.plot(x, np.cumsum(oracle) / power_mw / 1000, color="#8a8983", linewidth=1.4, linestyle="--", label="Perfect foresight (oracle)")
        for name, series in per.items():
            ax.plot(x, np.cumsum(series["revenue"]) / power_mw / 1000, color=COLORS[name], linewidth=1.7, label=LABELS[name])
        ax.set_title(MARKETS[key], fontsize=11, color=INK, loc="left")
        ax.set_ylabel("Cumulative revenue (kEUR per MW)", color=INK_MUTED)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(IMAGES / "market_cumulative_revenue.png", dpi=150)
    plt.close(fig)

    # 3. metric vs capture
    fig, (ax_mae, ax_tau) = plt.subplots(1, 2, figsize=(11, 4.8))
    for key, (backtest, dates, per, oracle) in runs.items():
        for name, series in per.items():
            capture = series["revenue"].sum() / oracle.sum() * 100
            ax_mae.scatter(-series["neg_mae"].mean() * 1000, capture, s=70, marker=MARKER[key], color=COLORS[name], edgecolor=INK, linewidth=0.5, zorder=3)
            ax_tau.scatter(np.nanmean(series["tau"]), capture, s=70, marker=MARKER[key], color=COLORS[name], edgecolor=INK, linewidth=0.5, zorder=3)
    ax_mae.set_xlabel("Mean absolute price error (EUR/MWh), lower is more accurate", color=INK_MUTED)
    ax_tau.set_xlabel("Mean Kendall's tau, higher is better", color=INK_MUTED)
    ax_mae.invert_xaxis()
    ax_mae.set_ylabel("Share of oracle revenue captured (%)", color=INK_MUTED)
    for ax in (ax_mae, ax_tau):
        style_axes(ax)
    model_handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[n], label=LABELS[n]) for n in per]
    market_handles = [plt.Line2D([], [], marker=MARKER[k], linestyle="", color="#8a8983", label=MARKETS[k]) for k in runs]
    fig.legend(handles=model_handles + market_handles, loc="lower center", ncol=len(model_handles) + len(market_handles),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Forecast accuracy and ranking against captured revenue", fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(IMAGES / "market_metric_vs_capture.png", dpi=150)
    plt.close(fig)
    print("Wrote market_revenue_by_year.png, market_cumulative_revenue.png, market_metric_vs_capture.png")


if __name__ == "__main__":
    main()
