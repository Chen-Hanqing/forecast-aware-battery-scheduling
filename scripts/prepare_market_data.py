"""Build a market-side history CSV (day-ahead price + residual load, no household) from ENTSO-E.

For a merchant, grid-connected battery there is no site load or PV, so `load_kw` and `pv_kw`
are zero and the only forecasting problem is the price. `residual_load_mw` (actual system load
minus actual wind and solar generation) is kept as the price's fundamental driver.

Day-ahead prices are hourly until 2025-09-30. The EU day-ahead market moved to 15-minute
products on 2025-10-01, so the default end date stops before that rather than mixing two
resolutions.

Usage:
    pip install -e '.[research]'
    echo 'ENTSOE_API_KEY=...' > .env
    python -m scripts.prepare_market_data --zone DE_LU --out data/raw/market_de_lu.csv
    python -m scripts.prepare_market_data --zone NL     --out data/raw/market_nl.csv
    python -m scripts.prepare_market_data --zone FR     --out data/raw/market_fr.csv
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from scripts.entsoe_client import (
    fetch_entsoe_day_ahead_prices,
    fetch_entsoe_residual_load,
)
from scripts.prepare_real_data import load_dotenv


def fetch_monthly(fetch_one_month, start: pd.Timestamp, end: pd.Timestamp, label: str, retries: int = 4) -> pd.DataFrame:
    """One request per month (long requests hit an offset-pagination bug in ENTSO-E), each retried
    with backoff so a transient error does not punch a hole into a multi-year series."""
    start, end = start.tz_localize(None), end.tz_localize(None)  # the fetchers take dates and localise to UTC themselves
    chunks = []
    for month_start in pd.date_range(start.to_period("M").start_time, end.to_period("M").start_time, freq="MS"):
        month_end = min(month_start + pd.DateOffset(months=1), end + pd.Timedelta(days=1))
        for attempt in range(1, retries + 1):
            try:
                chunks.append(fetch_one_month(month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == retries:
                    print(f"  {label} {month_start:%Y-%m}: giving up after {retries} tries ({exc})", flush=True)
                else:
                    time.sleep(5 * attempt)
        print(f"  {label} {month_start:%Y-%m} done", flush=True)
    return pd.concat(chunks, ignore_index=True).drop_duplicates("timestamp").set_index("timestamp").sort_index()


def longest_regular_run(df: pd.DataFrame, max_fill_hours: int = 6) -> pd.DataFrame:
    """Reindex to a strict hourly grid, fill gaps of at most `max_fill_hours`, and keep the longest
    gap-free stretch (the pipeline requires a regular, gap-free index)."""
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(grid).interpolate(limit=max_fill_hours, limit_area="inside")
    valid = df.notna().all(axis=1)
    run_id = (valid != valid.shift()).cumsum()
    best = valid.groupby(run_id).agg(["all", "size"])
    best_run = best[best["all"]]["size"].idxmax()
    return df[(run_id == best_run) & valid]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True, help="ENTSO-E bidding zone, e.g. DE_LU, NL, FR")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-09-30")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise SystemExit("Set ENTSOE_API_KEY (env var or .env file) before running this script")
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")

    prices = fetch_monthly(
        lambda s, e: fetch_entsoe_day_ahead_prices(api_key, args.zone, s, e), start, end, f"{args.zone} price")
    residual = fetch_monthly(
        lambda s, e: fetch_entsoe_residual_load(api_key, args.zone, s, e), start, end, f"{args.zone} residual load")

    merged = prices.join(residual, how="inner")[["import_price_eur_kwh", "residual_load_mw"]]
    merged = merged[(merged.index >= start) & (merged.index < end + pd.Timedelta(days=1))]
    clean = longest_regular_run(merged)
    dropped = len(merged) - len(clean)
    out = clean.rename_axis("timestamp").reset_index()
    out["load_kw"] = 0.0
    out["pv_kw"] = 0.0
    out = out[["timestamp", "load_kw", "import_price_eur_kwh", "pv_kw", "residual_load_mw"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    negative = (out.import_price_eur_kwh < 0).mean()
    print(f"Wrote {len(out)} hourly rows ({out.timestamp.min()} -> {out.timestamp.max()}) to {args.out}; "
          f"dropped {dropped} rows outside the longest gap-free run; {negative:.1%} of hours have negative prices",
          flush=True)


if __name__ == "__main__":
    main()
