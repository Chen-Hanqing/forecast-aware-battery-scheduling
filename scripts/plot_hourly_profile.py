"""Average daily price shape: what each forecaster thinks a day looks like vs. what it looked like.

Averages the stored per-day forecast and actual price vectors (pipeline._backtest()'s `daily`
records) by hour of day. Built to show one mechanism: the two plain Lasso models, whose only
hour-of-day feature is a single sin/cos pair, can draw only one smooth hump per day and put the
peak around midday, while the real evening (and morning) peaks sit elsewhere. Adding 24 hour
dummies is the only difference between "Lasso-AR" and "Lasso-AR + hour dummies".

Usage:
    python -m scripts.plot_hourly_profile          # household study (docs/images/hourly_profile.png)
    python -m scripts.plot_hourly_profile --out market_hourly_profile.png --unit EUR/MWh --scale 1000 \\
        --panel "Germany-Luxembourg=results/market/run_metrics_market_de_lu.json" \\
        --panel "Netherlands=results/market/run_metrics_market_nl.json" \\
        --panel "France=results/market/run_metrics_market_fr.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.style import COLORS, GRID, INK, INK_MUTED, LABELS, MODEL_ORDER

PANELS = [
    ("results/household/run_metrics_real_de.json", "Real German market, Jan 6 – Feb 4, 2018"),
    ("results/household/run_metrics_counterfactual.json", "Constructed scenario, 2022–23 prices"),
]
STYLES = {  # secondary encoding, since three series colours are low-contrast on a light surface
    "seasonal_naive": (":", None), "gradient_boosting": ("--", None),
    "lasso_ar": ("-", None), "lear_asinh": ("-", None), "lasso_ar_hourly": ("-", None),
}


def profiles(metrics_path: str, scale: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    daily = [row for path in metrics_path.split(",") for row in json.loads(Path(path).read_text())["backtest"]["daily"]]
    present = [n for n in MODEL_ORDER if any(d["candidate"] == n for d in daily)]
    actual = np.mean([d["actual_price_eur_kwh"] for d in daily if d["candidate"] == present[0]], axis=0) * scale
    out = {n: np.mean([d["forecast_price_eur_kwh"] for d in daily if d["candidate"] == n], axis=0) * scale for n in present}
    return out, actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", action="append", help='"title=path"; repeatable')
    parser.add_argument("--out", default="hourly_profile.png")
    parser.add_argument("--unit", default="ct/kWh")
    parser.add_argument("--scale", type=float, default=100.0, help="multiplier from EUR/kWh to the plotted unit")
    args = parser.parse_args()
    panels = [tuple(item.split("=", 1)) for item in args.panel] if args.panel else [(t, p) for p, t in PANELS]

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2))
    axes = np.atleast_1d(axes)
    hours = np.arange(24)
    for ax, (title, path) in zip(axes, panels):
        forecasts, actual = profiles(path, args.scale)
        ax.plot(hours, actual, color=INK, linewidth=3.0, label="Actual", zorder=5)
        ax.scatter([actual.argmax()], [actual.max()], s=70, color=INK, zorder=6)
        for name, forecast in forecasts.items():
            style, _ = STYLES.get(name, ("-", None))
            ax.plot(hours, forecast, style, color=COLORS[name], linewidth=1.9, label=LABELS[name], zorder=4)
            peak = forecast.argmax()
            ax.scatter([peak], [forecast[peak]], s=46, color=COLORS[name], edgecolor="#fcfcfb", linewidth=1.2, zorder=6)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Hour of day", color=INK_MUTED)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED)
    axes[0].set_ylabel(f"Average day-ahead price ({args.unit})", color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Average price by hour of day: what each forecaster expects vs. what happened",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, "Dots mark each series' peak hour.", fontsize=9.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    out = Path("docs/images") / args.out
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    for title, path in panels:
        forecasts, actual = profiles(path, args.scale)
        print(title, "| peak hour: actual", int(actual.argmax()), {n: int(v.argmax()) for n, v in forecasts.items()})


if __name__ == "__main__":
    main()
