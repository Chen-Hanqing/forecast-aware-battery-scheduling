"""ENTSO-E Transparency Platform connectors used by scripts/prepare_real_data.py
and scripts/prepare_counterfactual_data.py to build real-market history CSVs.

Requires the `research` extra (entsoe-py) and an ENTSOE_API_KEY (network calls,
so this isn't exercised by CI or unit tests).
"""
from __future__ import annotations

import pandas as pd


def fetch_entsoe_day_ahead_prices(
    api_key: str,
    country_code: str,
    start: str,
    end: str,
    target_freq: str = "1h",
) -> pd.DataFrame:
    """Fetch and clean ENTSO-E day-ahead prices into the pipeline's input contract.

    Returns a DataFrame with columns [timestamp, import_price_eur_kwh].
    """
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")

    raw = client.query_day_ahead_prices(country_code=country_code, start=start_ts, end=end_ts)
    df = raw.to_frame(name="price_eur_mwh")
    df.index.name = "timestamp"
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Some European markets settle at 15/30-min granularity; resample to the target frequency.
    resampled = df.resample(target_freq).mean()
    resampled["price_eur_mwh"] = resampled["price_eur_mwh"].ffill().bfill()
    resampled["import_price_eur_kwh"] = resampled["price_eur_mwh"] / 1000.0
    return resampled.reset_index()[["timestamp", "import_price_eur_kwh"]]


def fetch_entsoe_residual_load(
    api_key: str,
    country_code: str,
    start: str,
    end: str,
    target_freq: str = "1h",
) -> pd.DataFrame:
    """Fetch ENTSO-E actual system load and wind+solar generation, and compute
    residual load = load - (solar + wind onshore + wind offshore) in MW.

    This is the real day-ahead-price driver (price is set by net/residual
    demand after renewables), as opposed to one household's own load/PV,
    which only correlates with it noisily through shared regional weather.

    Returns a DataFrame with columns [timestamp, residual_load_mw].
    """
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")

    load = client.query_load(country_code, start=start_ts, end=end_ts)["Actual Load"]

    generation = client.query_generation(country_code, start=start_ts, end=end_ts)
    renewable_cols = [
        (name, "Actual Aggregated")
        for name in ("Solar", "Wind Onshore", "Wind Offshore")
        if (name, "Actual Aggregated") in generation.columns
    ]
    renewables = generation[renewable_cols].sum(axis=1)

    residual = (load - renewables).rename("residual_load_mw").to_frame()
    residual.index.name = "timestamp"
    residual.index = residual.index.tz_convert("UTC") if residual.index.tz is not None else residual.index.tz_localize("UTC")
    residual = residual[~residual.index.duplicated(keep="first")].sort_index()
    return residual.resample(target_freq).mean().reset_index()
