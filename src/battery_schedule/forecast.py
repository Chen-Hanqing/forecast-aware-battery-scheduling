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
ALIGNED_LAGS = (24, 48, 168)  # same-hour-yesterday / two-days-ago / last-week, relative to the *target* hour

def has_residual_load(history: pd.DataFrame) -> bool:
    """True if real system-wide residual load (actual load - wind - solar) is available
    for this history (see scripts/prepare_real_data.py) rather than absent/all-NaN
    (synthetic and counterfactual data don't have it)."""
    return "residual_load_mw" in history.columns and history["residual_load_mw"].notna().any()

def _features(index: pd.DatetimeIndex, history: pd.DataFrame, holiday_country: str | None,
              hour_onehot: bool = False) -> pd.DataFrame:
    """Calendar + autoregressive-lag features, all knowable at `index` (the origin) —
    no lag shorter than 24h, so nothing here depends on a same-run prediction.

    The default calendar block is a single sin/cos pair per cycle, i.e. only the first harmonic
    of the hour of day. A linear model built on it can only draw one sinusoid per horizon step,
    so it cannot represent a morning-and-evening double peak. `hour_onehot=True` adds a dummy per
    hour of day (24 columns) so the linear models can express an arbitrary daily shape."""
    x = pd.DataFrame(index=index)
    hour = index.hour.to_numpy(); dow = index.dayofweek.to_numpy()
    x["hour_sin"] = np.sin(2*np.pi*hour/24); x["hour_cos"] = np.cos(2*np.pi*hour/24)
    x["dow_sin"] = np.sin(2*np.pi*dow/7); x["dow_cos"] = np.cos(2*np.pi*dow/7)
    if hour_onehot:
        for h in range(24): x[f"hour_is_{h}"] = (hour == h).astype(float)
    lag_cols = [*TARGETS, "residual_load_mw"] if has_residual_load(history) else TARGETS
    for col in lag_cols:
        series = history[col]
        for lag in LAGS: x[f"{col}_lag_{lag}"] = series.reindex(index - pd.Timedelta(hours=lag)).to_numpy()
    if holiday_country:
        local_holidays = holidays.country_holidays(holiday_country)
        x["is_holiday"] = [ts.date() in local_holidays for ts in index]
    return x

def _aligned_lag(series: pd.Series, lag: int, step: int) -> pd.Series:
    """Value of `series` `lag` hours before the hour being forecast, as a feature at the origin.
    An origin at t forecasting t+step needs the value at t+step-lag, i.e. `series.shift(lag - step)`.
    That is a shift by a non-negative amount whenever lag >= step (true for every horizon step and lag
    used here), so it only ever looks backwards. The plain LAGS in `_features` are relative to the
    origin instead, which for step k means the price 24+k, 48+k, 168+k hours before the target hour,
    so a model built on them never sees "the same hour yesterday" (which seasonal naive has by
    construction)."""
    if lag < step:
        raise ValueError("aligned lag shorter than the horizon step would look into the future")
    return series.shift(lag - step)

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

def _lasso_ar_model() -> object:
    """Plain LASSO-regularized linear autoregression on raw prices. This is the linear core of
    LEAR (Lago et al. 2021, https://doi.org/10.1016/j.apenergy.2021.116983) but *without* its
    asinh variance-stabilising price transform; see `lear_asinh` for the closer variant."""
    return make_pipeline(StandardScaler(), LassoCV(cv=3, alphas=50, max_iter=10000))

def _asinh_params(prices: pd.Series) -> tuple[float, float]:
    """Median and (normal-consistent) MAD of the training window — the 'invariant' scaling
    epftoolbox applies before the asinh transform. Fitted on training data only."""
    med = float(prices.median())
    mad = float((prices - med).abs().median() * 1.4826)
    return med, (mad if mad > 0 else 1.0)

def _lgbm_quantile_model() -> object:
    """Median (q50) LightGBM regression: more robust to price spikes/negative excursions
    than a mean-squared-error objective, while still giving a single point forecast.
    Imported lazily: lightgbm needs libomp on macOS, which isn't always installable
    (no prebuilt bottle -> a from-source LLVM build), so this candidate is opt-in."""
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=150, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=42)

MODEL_FACTORIES: dict[str, ModelFactory] = {
    "gradient_boosting": _gb_model,
    "lasso_ar": _lasso_ar_model,
    "lear_asinh": _lasso_ar_model,  # same estimator; _predict wraps it in the asinh price transform
    "lasso_ar_hourly": _lasso_ar_model,  # same estimator; _predict adds 24 hour-of-day dummies
    "gradient_boosting_aligned": _gb_model,  # gradient boosting + lags aligned to the target hour
    "lasso_ar_aligned": _lasso_ar_model,  # hour dummies + lags aligned to the target hour
    "lightgbm_quantile": _lgbm_quantile_model,
}

def _direct_multi_horizon(
    history: pd.DataFrame, future_index: pd.DatetimeIndex, holiday_country: str | None, model_factory: ModelFactory,
    household_price_exog: bool = True, hour_onehot: bool = False, aligned_lags: bool = False,
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
    features = _features(history.index, history, holiday_country, hour_onehot)
    origin = features.iloc[[-1]]
    horizon = len(future_index)

    def fit_predict(target: str, exogenous: list[tuple[str, pd.Series, list[float]]] | None = None,
                    floor: float | None = None) -> list[float]:
        preds = []
        for step in range(1, horizon + 1):
            x, x_origin = features, origin
            for name, series, forecast in exogenous or []:
                x = x.assign(**{name: series.shift(-step)})
                x_origin = x_origin.assign(**{name: forecast[step - 1]})
            if aligned_lags:
                for lag in ALIGNED_LAGS:
                    lagged = _aligned_lag(history[target], lag, step)
                    x = x.assign(**{f"{target}_aligned_lag_{lag}": lagged})
                    x_origin = x_origin.assign(**{f"{target}_aligned_lag_{lag}": lagged.iloc[-1]})
            train = x.join(history[target].shift(-step).rename("y")).dropna()
            model = model_factory()
            model.fit(train.drop(columns="y"), train["y"])
            pred = float(model.predict(x_origin)[0])
            preds.append(pred if floor is None else max(floor, pred))
        return preds

    # Only load is physically non-negative. Day-ahead prices (and residual load) can be negative,
    # so clipping them at zero would erase exactly the hours a battery is paid to charge in.
    if history["load_kw"].nunique() <= 1:
        # No site load (a merchant, grid-connected battery): nothing to forecast.
        load_preds = [float(history["load_kw"].iloc[-1])] * horizon
    else:
        load_preds = fit_predict("load_kw", floor=0.0)
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
    elif name == "lear_asinh":
        med, mad = _asinh_params(history["import_price_eur_kwh"])
        transformed = history.copy()
        transformed["import_price_eur_kwh"] = np.arcsinh((history["import_price_eur_kwh"] - med) / mad)
        out = _direct_multi_horizon(transformed, future, holiday_country, MODEL_FACTORIES[name], household_price_exog)
        out["import_price_eur_kwh"] = med + mad * np.sinh(out["import_price_eur_kwh"])
    elif name == "lasso_ar_hourly":
        out = _direct_multi_horizon(history, future, holiday_country, MODEL_FACTORIES[name], household_price_exog, hour_onehot=True)
    elif name == "lasso_ar_aligned":
        out = _direct_multi_horizon(history, future, holiday_country, MODEL_FACTORIES[name], household_price_exog,
                                    hour_onehot=True, aligned_lags=True)
    elif name == "gradient_boosting_aligned":
        out = _direct_multi_horizon(history, future, holiday_country, MODEL_FACTORIES[name], household_price_exog,
                                    aligned_lags=True)
    elif name in MODEL_FACTORIES:
        out = _direct_multi_horizon(history, future, holiday_country, MODEL_FACTORIES[name], household_price_exog)
    else:
        raise ValueError(f"Unknown model: {name}")
    return out

def forecast_once(
    history: pd.DataFrame, horizon: int, name: str, holiday_country: str | None = None,
    lookback_days: int | None = None, household_price_exog: bool = True,
) -> pd.DataFrame:
    """One forecast from `history`, with no validation window and no residuals. Enough when the
    dispatch is deterministic (point forecast only); scenarios need `select_and_forecast`."""
    return _predict(history, horizon, name, holiday_country, lookback_days, household_price_exog)

def select_and_forecast(
    history: pd.DataFrame, horizon: int, validation_days: int, candidates: list[str],
    holiday_country: str | None = None, lookback_days: int | None = None, household_price_exog: bool = True,
    pred_cache: dict | None = None,
) -> ForecastResult:
    """`pred_cache` (optional, only valid while `history` is a prefix of one fixed full history)
    memoises forecasts by (candidate, training length). Successive backtest origins share 13 of
    their 14 validation origins, and each origin's final refit is the next origin's newest
    validation forecast, so this cuts per-origin cost roughly 15x without changing any output."""
    def predict(train: pd.DataFrame, name: str) -> pd.DataFrame:
        key = (name, len(train))
        if pred_cache is not None and key in pred_cache:
            return pred_cache[key].copy()
        out = _predict(train, horizon, name, holiday_country, lookback_days, household_price_exog)
        if pred_cache is not None:
            pred_cache[key] = out.copy()
        return out

    # Rolling daily origins: each forecast only sees data strictly before its origin.
    origins = range(len(history) - validation_days*24, len(history) - horizon + 1, 24)
    scores: dict[str, list[float]] = {name: [] for name in candidates}
    all_residuals: dict[str, list[pd.DataFrame]] = {name: [] for name in candidates}
    for origin in origins:
        train, actual = history.iloc[:origin], history.iloc[origin:origin+horizon][TARGETS]
        for name in candidates:
            pred = predict(train, name); pred.index = actual.index
            scores[name].append(float((pred - actual).abs().mean().mean()))
            all_residuals[name].append(actual - pred)
    best = min(candidates, key=lambda n: np.mean(scores[n]))
    forecast = predict(history, best)
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
        output[s] = base + np.asarray(shocks[:horizon])
    output[..., 0] = np.maximum(0, output[..., 0])  # load can't be negative; prices can
    return output
