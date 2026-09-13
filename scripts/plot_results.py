"""Render the day-ahead dispatch chart from a completed `battery-schedule run`.

Reads forecast.csv/schedule.csv and writes docs/images/dispatch_<suffix>.png.
For the cost-comparison / MAE-vs-tau charts (baseline vs. each candidate vs.
oracle), see scripts/plot_forecast_value.py — it reads run_metrics.json's
backtest.candidates breakdown, so it works the same way across every config.

    python scripts/plot_results.py --config configs/default.yaml --suffix synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from battery_schedule.config import load_config


def plot_dispatch(forecast: pd.DataFrame, schedule: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, height_ratios=[1, 1.1]
    )

    ax_top.plot(forecast.index, forecast["load_kw"], color="#1f77b4", label="Forecast load (kW)")
    ax_top.plot(forecast.index, forecast["pv_kw"], color="#2ca02c", label="PV forecast (kW)")
    ax_top.set_ylabel("kW")
    ax_price = ax_top.twinx()
    ax_price.plot(
        forecast.index, forecast["import_price_eur_kwh"],
        color="#d62728", linestyle="--", label="Import price (EUR/kWh)",
    )
    ax_price.set_ylabel("EUR/kWh")
    lines_top = ax_top.get_lines() + ax_price.get_lines()
    ax_top.legend(lines_top, [line.get_label() for line in lines_top], loc="upper left", fontsize=8)
    ax_top.set_title("Day-ahead forecast and optimized battery dispatch")

    width = 0.03
    ax_bottom.bar(schedule.index, schedule["charge_kw"], width=width, color="#2ca02c", label="Charge (kW)")
    ax_bottom.bar(schedule.index, -schedule["discharge_kw"], width=width, color="#d62728", label="Discharge (kW)")
    ax_bottom.set_ylabel("Battery power (kW)")
    ax_bottom.axhline(0, color="black", linewidth=0.6)
    ax_soc = ax_bottom.twinx()
    ax_soc.plot(schedule.index, schedule["soc_kwh"], color="#333333", marker="o", markersize=3, label="SOC (kWh)")
    ax_soc.set_ylabel("State of charge (kWh)")
    lines_bottom = ax_bottom.get_legend_handles_labels()
    handles, labels = lines_bottom
    soc_handles, soc_labels = ax_soc.get_legend_handles_labels()
    ax_bottom.legend(handles + soc_handles, labels + soc_labels, loc="upper left", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--suffix", default="synthetic")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = Path(config["output_dir"])
    images = Path("docs/images")
    images.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(artifacts / "forecast.csv", index_col=0, parse_dates=True)
    schedule = pd.read_csv(artifacts / "schedule.csv", index_col=0, parse_dates=True)

    out_path = images / f"dispatch_{args.suffix}.png"
    plot_dispatch(forecast, schedule, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
