"""Leakage-free multi-step forecasts and joint residual scenarios."""
from collections.abc import Callable
from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["load_kw", "import_price_eur_kwh"]
LAGS = (24, 48, 168)

def has_residual_load(history: pd.DataFrame) -> bool:
    """True if real system-wide residual load (actual load - wind - solar) is available
    for this history (see scripts/prepare_real_data.py) rather than absent/all-NaN
    (synthetic and counterfactual data don't have it)."""
    return "residual_load_mw" in history.columns and history["residual_load_mw"].notna().any()

def _features(index: pd.DatetimeIndex, history: pd.DataFrame, holiday_country: str | None) -> pd.DataFrame:
    """Calendar + autoregressive-lag features, all knowable at `index` (the origin) —
    no lag shorter than 24h, so nothing here depends on a same-run prediction."""
    x = pd.DataFrame(index=index)
    hour = index.hour.to_numpy(); dow = index.dayofweek.to_numpy()
    x["hour_sin"] = np.sin(2*np.pi*hour/24); x["hour_cos"] = np.cos(2*np.pi*hour/24)
    x["dow_sin"] = np.sin(2*np.pi*dow/7); x["dow_cos"] = np.cos(2*np.pi*dow/7)
    lag_cols = [*TARGETS, "residual_load_mw"] if has_residual_load(history) else TARGETS
    for col in lag_cols:
        series = history[col]
        for lag in LAGS: x[f"{col}_lag_{lag}"] = series.reindex(index - pd.Timedelta(hours=lag)).to_numpy()
    if holiday_country:
        local_holidays = holidays.country_holidays(holiday_country)
        x["is_holiday"] = [ts.date() in local_holidays for ts in index]
    return x

@dataclass
class ForecastResult:
    mean: pd.DataFrame
    residuals: pd.DataFrame
    model_name: str
    validation_mae: float

def _seasonal_values(history: pd.DataFrame, horizon: int, columns: list[str]) -> np.ndarray:
    return history[columns].iloc[-168:].to_numpy()[np.arange(horizon) % min(168, len(history))].copy()

def _seasonal(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return _seasonal_values(history, horizon, TARGETS)

def _recent_window(history: pd.DataFrame, lookback_days: int | None) -> pd.DataFrame:
    """Bound training to a trailing window: cheaper to fit, and recent hours are more
    representative of current conditions than data from years earlier."""
    if not lookback_days:
        return history
    return history.iloc[-lookback_days * 24:]

ModelFactory = Callable[[], object]  # sklearn-like estimator: .fit(X, y) / .predict(X)

def _gb_model() -> object:
    return HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=1e-3, random_state=42)

def _lear_model() -> object:
    """LEAR: LASSO-regularized linear autoregression — a standard day-ahead price forecasting
    baseline (Lago et al. 2021, https://doi.org/10.1016/j.apenergy.2021.116983) alongside DNNs."""
    return make_pipeline(StandardScaler(), LassoCV(cv=3, alphas=50, max_iter=10000))

def _lgbm_quantile_model() -> object:
    """Median (q50) LightGBM regression: more robust to price spikes/negative excursions
    than a mean-squared-error objective, while still giving a single point forecast.
    Imported lazily: lightgbm needs libomp on macOS, which isn't always installable
    (no prebuilt bottle -> a from-source LLVM build), so this candidate is opt-in."""
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=150, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=42)

MODEL_FACTORIES: dict[str, ModelFactory] = {
    "gradient_boosting": _gb_model,
    "lear": _lear_model,
    "lightgbm_quantile": _lgbm_quantile_model,
}

def _direct_multi_horizon(
    history: pd.DataFrame, future_index: pd.DatetimeIndex, holiday_country: str | None, model_factory: ModelFactory,
    household_price_exog: bool = True,
) -> pd.DataFrame:
    """Direct multi-step forecast: one model per horizon step, trained only on features
    known at the origin. Unlike a recursive (step-by-step) forecast, nothing here feeds a
    step's own prediction back in as an input to the next step, so errors can't compound
    across the horizon. Price additionally uses load and PV as exogenous drivers (day-ahead
    price is set by net demand, i.e. load minus renewables): each is trained against its own
    historical value at t+step, then predicted using its own day-ahead forecast for that
    step, since the real future value is unknown at decision time — the same
    train-on-actual/deploy-on-forecast substitution real day-ahead price models make for
    TSO-published load/generation forecasts. PV's forecast reuses the same seasonal-persistence
    estimate the pipeline already feeds to the battery LP, so both consumers agree on "the
    day-ahead PV forecast". When real system-wide residual load (actual load minus wind
    and solar generation) is available, it's added as a third exogenous driver alongside
    this household's own load/PV — one household's own numbers only correlate with the
    true market fundamental noisily, through shared regional weather. household_price_exog=False
    drops the household's own load/PV from price's exogenous inputs when residual load is
    present, to test whether they still add anything once the true market fundamental is
    available (they're kept regardless when residual load is absent, since price would
    otherwise have no exogenous driver at all).
    """
    features = _features(history.index, history, holiday_country)
    origin = features.iloc[[-1]]
    horizon = len(future_index)

    def fit_predict(target: str, exogenous: list[tuple[str, pd.Series, list[float]]] | None = None) -> list[float]:
        preds = []
        for step in range(1, horizon + 1):
            x, x_origin = features, origin
            for name, series, forecast in exogenous or []:
                x = x.assign(**{name: series.shift(-step)})
                x_origin = x_origin.assign(**{name: forecast[step - 1]})
            train = x.join(history[target].shift(-step).rename("y")).dropna()
            model = model_factory()
            model.fit(train.drop(columns="y"), train["y"])
            preds.append(max(0.0, float(model.predict(x_origin)[0])))
        return preds

    load_preds = fit_predict("load_kw")
    pv_forecast = _seasonal_values(history, horizon, ["pv_kw"])[:, 0].tolist()
    residual_available = has_residual_load(history)
    price_exogenous = []
    if household_price_exog or not residual_available:
        price_exogenous += [("load_exog", history["load_kw"], load_preds), ("pv_exog", history["pv_kw"], pv_forecast)]
    if residual_available:
        residual_load_preds = fit_predict("residual_load_mw")
        price_exogenous.append(("residual_load_exog", history["residual_load_mw"], residual_load_preds))
    price_preds = fit_predict("import_price_eur_kwh", exogenous=price_exogenous)
    return pd.DataFrame({"load_kw": load_preds, "import_price_eur_kwh": price_preds}, index=future_index)

def _predict(
    history: pd.DataFrame, horizon: int, name: str, holiday_country: str | None = None, lookback_days: int | None = None,
    household_price_exog: bool = True,
) -> pd.DataFrame:
    history = _recent_window(history, lookback_days)
    if len(history) < 192: raise ValueError("Need at least 192 hourly observations for forecasting")
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h", tz=history.index.tz)
    if name == "seasonal_naive":
        out = pd.DataFrame(_seasonal(history, horizon), index=future, columns=TARGETS)
    elif name in MODEL_FACTORIES:
        out = _direct_multi_horizon(history, future, holiday_country, MODEL_FACTORIES[name], household_price_exog)
    else:
        raise ValueError(f"Unknown model: {name}")
    return out

def select_and_forecast(
    history: pd.DataFrame, horizon: int, validation_days: int, candidates: list[str],
    holiday_country: str | None = None, lookback_days: int | None = None, household_price_exog: bool = True,
) -> ForecastResult:
    # Rolling daily origins: each forecast only sees data strictly before its origin.
    origins = range(len(history) - validation_days*24, len(history) - horizon + 1, 24)
    scores: dict[str, list[float]] = {name: [] for name in candidates}
    all_residuals: dict[str, list[pd.DataFrame]] = {name: [] for name in candidates}
    for origin in origins:
        train, actual = history.iloc[:origin], history.iloc[origin:origin+horizon][TARGETS]
        for name in candidates:
            pred = _predict(train, horizon, name, holiday_country, lookback_days, household_price_exog); pred.index = actual.index
            scores[name].append(float((pred - actual).abs().mean().mean()))
            all_residuals[name].append(actual - pred)
    best = min(candidates, key=lambda n: np.mean(scores[n]))
    forecast = _predict(history, horizon, best, holiday_country, lookback_days, household_price_exog)
    residuals = pd.concat(all_residuals[best]).reset_index(drop=True)
    return ForecastResult(forecast, residuals, best, float(np.mean(scores[best])))

def make_scenarios(forecast: ForecastResult, count: int, block_steps: int, seed: int) -> np.ndarray:
    """Return [scenario, time, (load, import_price)]; blocks preserve joint shocks."""
    rng = np.random.default_rng(seed)
    base = forecast.mean[TARGETS].to_numpy()
    residuals = forecast.residuals.to_numpy()
    n = count
    horizon = len(base)
    output = np.empty((n, horizon, 2))
    for s in range(n):
        shocks = []
        while len(shocks) < horizon:
            start = rng.integers(0, max(1, len(residuals)-block_steps+1))
            shocks.extend(residuals[start:start+block_steps])
        output[s] = np.maximum(0, base + np.asarray(shocks[:horizon]))
    return output
