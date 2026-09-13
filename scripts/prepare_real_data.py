"""Build a real-world history CSV from OPSD household data + ENTSO-E prices
and system fundamentals.

Data sources:
- Load & PV: OPSD household_data (DE_KN_residential4, Konstanz, southern Germany),
  https://data.open-power-system-data.org/household_data/ — cumulative meter
  readings at 60-min resolution; this script diffs them into hourly kWh.
- Price: ENTSO-E day-ahead auction clearing price for the DE_AT_LU bidding zone
  (Germany-Austria-Luxembourg were a single zone until the 2018-10-01 split;
  residential4's data window falls entirely before that).
- residual_load_mw: ENTSO-E actual system load minus actual wind+solar
  generation for the same zone — the real day-ahead-price driver (see
  scripts/entsoe_client.py's fetch_entsoe_residual_load). One household's own
  load/PV only correlates with this noisily through shared regional weather;
  this is the fundamental the market itself actually clears against.

Usage:
    pip install -e '.[research]'
    curl -o data/raw/opsd_household_60min.csv \\
      https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv
    echo 'ENTSOE_API_KEY=...' > .env
    python scripts/prepare_real_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.entsoe_client import (
    fetch_entsoe_day_ahead_prices,
    fetch_entsoe_residual_load,
)

HOUSEHOLD = "residential4"
RAW_OPSD = Path("data/raw/opsd_household_60min.csv")
OUT_CSV = Path("data/raw/history_de_real.csv")
BIDDING_ZONE = "DE_AT_LU"  # pre 2018-10-01 combined DE/AT/LU day-ahead zone


def load_dotenv(path: Path = Path(".env")) -> None:
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_household_load_and_pv() -> pd.DataFrame:
    """Diff OPSD's cumulative meter columns into hourly kWh (= average kW at 1h resolution)."""
    cols = [f"DE_KN_{HOUSEHOLD}_grid_import", f"DE_KN_{HOUSEHOLD}_pv", f"DE_KN_{HOUSEHOLD}_grid_export"]
    raw = pd.read_csv(RAW_OPSD, usecols=["utc_timestamp", *cols], parse_dates=["utc_timestamp"])
    raw = raw.set_index("utc_timestamp").dropna()

    # Keep only the longest run with no gaps at the expected 1h cadence.
    gap = raw.index.to_series().diff().dt.total_seconds().fillna(3600)
    run_id = (gap != 3600).cumsum()
    longest_run = run_id.value_counts().idxmax()
    raw = raw.loc[run_id == longest_run]

    diffs = raw.diff().iloc[1:]  # first row has no prior cumulative reading
    if (diffs < -1e-6).any().any():
        raise ValueError("Found a decreasing cumulative meter reading (possible meter reset)")

    return pd.DataFrame({
        "load_kw": diffs[f"DE_KN_{HOUSEHOLD}_grid_import"].clip(lower=0),
        "pv_kw": diffs[f"DE_KN_{HOUSEHOLD}_pv"].clip(lower=0),
    })


def _fetch_chunked(fetch_one_month, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch one month at a time and concatenate.

    Querying DE_AT_LU in a single multi-year request hits an offset-pagination
    bug in entsoe-py/the ENTSO-E API for this bidding zone/period (HTTP 400 at
    offset=400); month-sized requests avoid it. Some individual months still
    400 on ENTSO-E's side (a server-side gap, not our bug) — skip and continue.
    """
    chunks = []
    month_starts = pd.date_range(start.to_period("M").start_time, end.to_period("M").start_time, freq="MS")
    for month_start in month_starts:
        month_end = month_start + pd.DateOffset(months=1)
        try:
            chunk = fetch_one_month(month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {month_start.date()}: {exc}")
            continue
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True).drop_duplicates("timestamp").set_index("timestamp").sort_index()


def fetch_entsoe_prices_chunked(api_key: str, start: pd.Timestamp, end: pd.Timestamp, zone: str = BIDDING_ZONE) -> pd.DataFrame:
    """Fetch day-ahead prices one month at a time (see _fetch_chunked)."""
    return _fetch_chunked(
        lambda s, e: fetch_entsoe_day_ahead_prices(api_key=api_key, country_code=zone, start=s, end=e), start, end
    )


def fetch_entsoe_residual_load_chunked(api_key: str, start: pd.Timestamp, end: pd.Timestamp, zone: str = BIDDING_ZONE) -> pd.DataFrame:
    """Fetch actual load + wind/solar generation one month at a time (see _fetch_chunked)."""
    return _fetch_chunked(
        lambda s, e: fetch_entsoe_residual_load(api_key=api_key, country_code=zone, start=s, end=e), start, end
    )


def main() -> None:
    load_dotenv()
    import os

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise SystemExit("Set ENTSOE_API_KEY (env var or .env file) before running this script")

    load_pv = load_household_load_and_pv()
    print(f"Household load/PV: {len(load_pv)} hours, {load_pv.index.min()} -> {load_pv.index.max()}")

    prices = fetch_entsoe_prices_chunked(api_key, load_pv.index.min(), load_pv.index.max())
    print(f"ENTSO-E {BIDDING_ZONE} day-ahead prices: {len(prices)} hours")

    residual_load = fetch_entsoe_residual_load_chunked(api_key, load_pv.index.min(), load_pv.index.max())
    print(f"ENTSO-E {BIDDING_ZONE} residual load (actual load - wind - solar): {len(residual_load)} hours")

    merged = load_pv.join(prices, how="inner").join(residual_load, how="inner").dropna()

    # Real feed-in tariffs (EEG) are fixed 20-year contracts, not spot-linked, so we
    # deliberately leave export_price_eur_kwh out; the pipeline's default (a ratio of
    # the import price) is closer to a market-based alternative than a made-up number.
    gap = merged.index.to_series().diff().dt.total_seconds().fillna(3600)
    run_id = (gap != 3600).cumsum()
    longest_run = run_id.value_counts().idxmax()
    merged = merged.loc[run_id == longest_run]

    out = merged.rename_axis("timestamp").reset_index()
    out = out[["timestamp", "load_kw", "import_price_eur_kwh", "pv_kw", "residual_load_mw"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} hourly rows ({out.timestamp.min()} -> {out.timestamp.max()}) to {OUT_CSV}")


if __name__ == "__main__":
    main()
