"""Leakage-free multi-step forecasts and joint residual scenarios."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS = ["load_kw", "import_price_eur_kwh"]

def _features(index: pd.DatetimeIndex, history: pd.DataFrame, target: str) -> pd.DataFrame:
    x = pd.DataFrame(index=index)
    hour = index.hour.to_numpy(); dow = index.dayofweek.to_numpy()
    x["hour_sin"] = np.sin(2*np.pi*hour/24); x["hour_cos"] = np.cos(2*np.pi*hour/24)
    x["dow_sin"] = np.sin(2*np.pi*dow/7); x["dow_cos"] = np.cos(2*np.pi*dow/7)
    series = history[target]
    for lag in (1, 2, 24, 48, 168): x[f"lag_{lag}"] = series.reindex(index - pd.Timedelta(hours=lag)).to_numpy()
    return x

@dataclass
class ForecastResult:
    mean: pd.DataFrame
    residuals: pd.DataFrame
    model_name: str
    validation_mae: float

def _seasonal(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return history[TARGETS].iloc[-168:].to_numpy()[np.arange(horizon) % min(168, len(history))].copy()

def _gb(history: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    result = pd.DataFrame(index=future_index, columns=TARGETS, dtype=float)
    working = history.copy()
    for ts in future_index:
        for target in TARGETS:
            train_X = _features(working.index, working, target)
            train = train_X.join(working[target].rename("target")).dropna()
            x_train = train.drop(columns="target")
            y_train = train["target"]
            model = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=1e-3, random_state=42)
            model.fit(x_train, y_train)
            row = _features(pd.DatetimeIndex([ts]), working, target)
            value = float(model.predict(row)[0])
            result.loc[ts, target] = max(0.0, value)
            working.loc[ts, target] = value
        # PV is not forecast by this compact example; use seasonal persistence where it exists.
        working.loc[ts, "pv_kw"] = float(working["pv_kw"].iloc[-168]) if len(working) >= 168 else 0.0
    return result

def _predict(history: pd.DataFrame, horizon: int, name: str) -> pd.DataFrame:
    if len(history) < 192: raise ValueError("Need at least 192 hourly observations for forecasting")
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h", tz=history.index.tz)
    if name == "seasonal_naive":
        out = pd.DataFrame(_seasonal(history, horizon), index=future, columns=TARGETS)
    elif name == "gradient_boosting": out = _gb(history, future)
    else: raise ValueError(f"Unknown model: {name}")
    return out

def select_and_forecast(history: pd.DataFrame, horizon: int, validation_days: int, candidates: list[str]) -> ForecastResult:
    # Rolling daily origins: each forecast only sees data strictly before its origin.
    origins = range(len(history) - validation_days*24, len(history) - horizon + 1, 24)
    scores: dict[str, list[float]] = {name: [] for name in candidates}
    all_residuals: dict[str, list[pd.DataFrame]] = {name: [] for name in candidates}
    for origin in origins:
        train, actual = history.iloc[:origin], history.iloc[origin:origin+horizon][TARGETS]
        for name in candidates:
            pred = _predict(train, horizon, name); pred.index = actual.index
            scores[name].append(float((pred - actual).abs().mean().mean()))
            all_residuals[name].append(actual - pred)
    best = min(candidates, key=lambda n: np.mean(scores[n]))
    forecast = _predict(history, horizon, best)
    residuals = pd.concat(all_residuals[best]).reset_index(drop=True)
    return ForecastResult(forecast, residuals, best, float(np.mean(scores[best])))

def make_scenarios(forecast: ForecastResult, count: int, block_steps: int, seed: int) -> np.ndarray:
    """Return [scenario, time, (load, import_price)]; blocks preserve joint shocks."""
    rng = np.random.default_rng(seed); base = forecast.mean[TARGETS].to_numpy(); residuals = forecast.residuals.to_numpy()
    n, horizon = count, len(base); output = np.empty((n, horizon, 2))
    for s in range(n):
        shocks = []
        while len(shocks) < horizon:
            start = rng.integers(0, max(1, len(residuals)-block_steps+1))
            shocks.extend(residuals[start:start+block_steps])
        output[s] = np.maximum(0, base + np.asarray(shocks[:horizon]))
    return output
