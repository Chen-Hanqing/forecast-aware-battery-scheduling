from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"timestamp", "load_kw", "import_price_eur_kwh"}

def read_history(path: str | Path, frequency: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.timestamp.isna().any() or df.timestamp.duplicated().any(): raise ValueError("Invalid or duplicate timestamps")
    df = df.sort_values("timestamp").set_index("timestamp")
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    if not df.index.to_series().diff().dropna().eq(pd.Timedelta(frequency)).all():
        raise ValueError("History must have a regular frequency with no gaps")
    df["pv_kw"] = df.get("pv_kw", 0.0)
    df["export_price_eur_kwh"] = df.get("export_price_eur_kwh", np.nan)
    # Optional: real system-wide residual load (actual load - wind - solar), the true
    # day-ahead-price driver, when available (see scripts/prepare_real_data.py). Absent
    # for synthetic/counterfactual data, in which case forecast.py falls back to using
    # this household's own load/PV as price's exogenous drivers.
    df["residual_load_mw"] = df.get("residual_load_mw", np.nan)
    if (df[["load_kw", "pv_kw"]] < 0).any().any():
        raise ValueError("Load and PV must be non-negative")
    # Day-ahead prices can legitimately go negative (e.g. high-renewables, low-demand
    # hours on the German market), so import/export prices aren't checked here.
    return df

def make_demo_data(path: str | Path, days: int = 120, seed: int = 42) -> None:
    rng = np.random.default_rng(seed); n = days * 24
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    h = idx.hour.to_numpy(); dow = idx.dayofweek.to_numpy()
    load = 1.5 + 1.2*np.exp(-((h-19)/3)**2) + .35*np.exp(-((h-8)/2.5)**2) + rng.normal(0,.12,n)
    pv = np.maximum(0, 3.8*np.sin(np.pi*(h-6)/12)) * (0.75 + .25*rng.random(n))
    price = .14 + .22*np.exp(-((h-19)/3.5)**2) + .025*(dow < 5) + rng.normal(0,.012,n)
    out = pd.DataFrame({"timestamp":idx, "load_kw":load.clip(.1), "pv_kw":pv,
                        "import_price_eur_kwh":price.clip(.03), "export_price_eur_kwh":(price*.75).clip(.02)})
    Path(path).parent.mkdir(parents=True, exist_ok=True); out.to_csv(path, index=False)

