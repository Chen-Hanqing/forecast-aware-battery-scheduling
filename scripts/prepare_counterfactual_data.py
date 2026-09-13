"""Build a COUNTERFACTUAL history CSV: this household's real load/PV *pattern*,
positionally overlaid onto a real recent (2022-23 energy-crisis-era) DE-LU
day-ahead price series.

This is NOT a historical replay — the household never experienced these
prices, and the load/PV values were never actually co-observed with them.
It's a controlled experiment holding the load pattern fixed and swapping only
the price regime, to isolate how much of the earlier (thin-spread, 2015-2018)
result was about that specific price era vs. the household/battery/method.
Compare against configs/real_de.yaml's genuine historical backtest, not
instead of it.

Usage:
    echo 'ENTSOE_API_KEY=...' > .env
    python -m scripts.prepare_counterfactual_data
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.prepare_real_data import (
    fetch_entsoe_prices_chunked,
    load_dotenv,
    load_household_load_and_pv,
)

OUT_CSV = Path("data/raw/history_counterfactual.csv")
PRICE_ZONE = "DE_LU"  # current (post 2018-10-01) German-Luxembourg zone
PRICE_START = pd.Timestamp("2022-08-01", tz="UTC")
PRICE_END = pd.Timestamp("2023-03-01", tz="UTC")  # covers the post-invasion gas-crisis winter


def main() -> None:
    load_dotenv()
    import os

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise SystemExit("Set ENTSOE_API_KEY (env var or .env file) before running this script")

    prices = fetch_entsoe_prices_chunked(api_key, PRICE_START, PRICE_END, zone=PRICE_ZONE)
    print(f"ENTSO-E {PRICE_ZONE} day-ahead prices: {len(prices)} hours, {prices.index.min()} -> {prices.index.max()}")
    print(f"  mean={prices.import_price_eur_kwh.mean():.4f}  std={prices.import_price_eur_kwh.std():.4f}  "
          f"min={prices.import_price_eur_kwh.min():.4f}  max={prices.import_price_eur_kwh.max():.4f} EUR/kWh")

    load_pv = load_household_load_and_pv()
    print(f"Household load/PV pattern: {len(load_pv)} hours available, {load_pv.index.min()} -> {load_pv.index.max()}")

    n = min(len(prices), len(load_pv))
    prices, load_pv = prices.iloc[:n], load_pv.iloc[:n]

    # Positional overlay: row i's load/PV (from the real household) paired with row i's
    # price (from the real recent market) — same hour-of-day/day-of-week alignment since
    # both series start at hour 0 of their respective windows, but NOT the same calendar
    # dates or a real co-occurrence.
    out = pd.DataFrame({
        "timestamp": prices.index,
        "load_kw": load_pv["load_kw"].to_numpy(),
        "import_price_eur_kwh": prices["import_price_eur_kwh"].to_numpy(),
        "pv_kw": load_pv["pv_kw"].to_numpy(),
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} hourly rows ({out.timestamp.min()} -> {out.timestamp.max()}) to {OUT_CSV}")
    print("Reminder: this is a constructed counterfactual (real load pattern + real recent prices, never co-observed).")


if __name__ == "__main__":
    main()
