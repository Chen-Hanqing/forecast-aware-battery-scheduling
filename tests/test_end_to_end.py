import numpy as np

from battery_schedule.config import BatteryConfig
from battery_schedule.data import make_demo_data, read_history
from battery_schedule.forecast import make_scenarios, select_and_forecast
from battery_schedule.optimise import solve_schedule


def test_forecast_scenario_and_schedule(tmp_path):
    path=tmp_path/"history.csv"; make_demo_data(path, days=18)
    history=read_history(path,"1h")
    fc=select_and_forecast(history, 24, 2, ["seasonal_naive"])
    scenarios=make_scenarios(fc, 4, 2, 1)
    cfg=BatteryConfig(13.5,5,5,.96,.96,6,6,.01)
    output=solve_schedule(scenarios, history.pv_kw.iloc[-24:].to_numpy(), history.export_price_eur_kwh.iloc[-24:].to_numpy(), cfg)
    assert len(output)==24
    assert output.soc_kwh.between(0,13.5).all()
    assert abs(output.soc_kwh.iloc[-1]-6)<1e-6


def test_solve_schedule_allows_net_exporter_day():
    """A day where PV vastly exceeds load is a net *earner*, not a net cost — the LP
    must not silently assume grid cost is bounded below by zero (regression: this used
    to make HiGHS report such days as infeasible)."""
    horizon = 24
    pv_kw = np.full(horizon, 8.0)
    load_kw = np.full(horizon, 0.2)
    price = np.full(horizon, 0.15)
    scenarios = np.stack([np.tile(load_kw, (3, 1)), np.tile(price, (3, 1))], axis=-1)
    cfg = BatteryConfig(13.5, 5, 5, .96, .96, 6, 6, .01)
    output = solve_schedule(scenarios, pv_kw, price * 0.75, cfg)
    assert len(output) == horizon


def _backtest_config(candidates, days=1):
    return {
        "data": {"horizon_steps": 24, "frequency": "1h"},
        "forecast": {"validation_days": 2, "candidates": candidates, "holiday_country": None, "lookback_days": None},
        "scenarios": {"count": 4, "residual_block_steps": 2},
        "optimization": BatteryConfig(13.5, 5, 5, .96, .96, 6, 6, .01),
        "backtest": {"enabled": True, "days": days},
    }


def test_backtest_schedule_cannot_see_the_day_it_is_priced_on(tmp_path):
    """Distort every realised quantity (price, export price, PV, load) on the final backtest day.
    The committed schedule and forecast must not move at all; only the after-the-fact cost may.
    (Regression: the export price fed to the LP used to be derived from the *actual* price,
    which let every candidate see that day's true price shape.)"""
    from battery_schedule.pipeline import _backtest
    path = tmp_path / "history.csv"; make_demo_data(path, days=20)
    base = read_history(path, "1h")
    distorted = base.copy()
    final_day = distorted.index[-24:]
    for col, factor in [("import_price_eur_kwh", 3.0), ("export_price_eur_kwh", 3.0), ("pv_kw", 0.2), ("load_kw", 2.0)]:
        distorted.loc[final_day, col] = distorted.loc[final_day, col] * factor
    config = _backtest_config(["seasonal_naive"])
    a = _backtest(base, config)["daily"][0]
    b = _backtest(distorted, config)["daily"][0]
    assert a["forecast_price_eur_kwh"] == b["forecast_price_eur_kwh"]
    assert a["charge_kw"] == b["charge_kw"] and a["discharge_kw"] == b["discharge_kw"]
    assert a["realised_cost_eur"] != b["realised_cost_eur"]


def test_scenarios_keep_negative_prices_but_not_negative_load():
    import pandas as pd

    from battery_schedule.forecast import ForecastResult
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    mean = pd.DataFrame({"load_kw": 0.1, "import_price_eur_kwh": -0.05}, index=index)
    residuals = pd.DataFrame({"load_kw": -0.5, "import_price_eur_kwh": -0.01}, index=range(48))
    scen = make_scenarios(ForecastResult(mean, residuals, "x", 0.0), 3, 4, 0)
    assert (scen[..., 1] < 0).all()
    assert (scen[..., 0] >= 0).all()


def test_prediction_cache_does_not_change_forecasts(tmp_path):
    """The backtest speed-up (memoised validation forecasts) must be output-neutral."""
    path = tmp_path / "history.csv"; make_demo_data(path, days=16)
    history = read_history(path, "1h")
    cache = {}
    for origin in (len(history) - 48, len(history) - 24):
        train = history.iloc[:origin]
        plain = select_and_forecast(train, 24, 2, ["lasso_ar"])
        cached = select_and_forecast(train, 24, 2, ["lasso_ar"], pred_cache=cache)
        assert np.allclose(plain.mean.to_numpy(), cached.mean.to_numpy())
        assert np.allclose(plain.residuals.to_numpy(), cached.residuals.to_numpy())
    assert cache


def test_asinh_transform_round_trips():
    import pandas as pd

    from battery_schedule.forecast import _asinh_params
    prices = pd.Series([0.02, 0.03, 0.05, -0.01, 0.4, 0.04])
    med, mad = _asinh_params(prices)
    restored = med + mad * np.sinh(np.arcsinh((prices - med) / mad))
    assert np.allclose(restored, prices)


def test_hour_onehot_features_have_exactly_one_hour_flag_per_row(tmp_path):
    from battery_schedule.forecast import _features
    path = tmp_path / "history.csv"; make_demo_data(path, days=10)
    history = read_history(path, "1h")
    plain = _features(history.index, history, None)
    dummies = _features(history.index, history, None, hour_onehot=True)
    hour_cols = [c for c in dummies.columns if c.startswith("hour_is_")]
    assert len(hour_cols) == 24 and not any(c.startswith("hour_is_") for c in plain.columns)
    assert (dummies[hour_cols].sum(axis=1) == 1).all()


def test_per_scenario_export_price_matches_shared_when_scenarios_agree():
    horizon = 24
    price = 0.05 + 0.04 * np.sin(np.linspace(0, 2 * np.pi, horizon))
    scenarios = np.stack([np.full((3, horizon), 0.4), np.tile(price, (3, 1))], axis=-1)
    cfg = BatteryConfig(13.5, 5, 5, .96, .96, 6, 6, .01)
    pv = np.zeros(horizon)
    shared = solve_schedule(scenarios, pv, price * 0.75, cfg)
    per_scenario = solve_schedule(scenarios, pv, np.tile(price * 0.75, (3, 1)), cfg)
    assert np.allclose(shared.charge_kw, per_scenario.charge_kw, atol=1e-6)
    assert np.allclose(shared.discharge_kw, per_scenario.discharge_kw, atol=1e-6)


def test_deterministic_backtest_cannot_see_the_day_it_is_priced_on(tmp_path):
    from battery_schedule.pipeline import _backtest
    path = tmp_path / "history.csv"; make_demo_data(path, days=20)
    base = read_history(path, "1h")
    distorted = base.copy()
    final_day = distorted.index[-24:]
    for col, factor in [("import_price_eur_kwh", 3.0), ("export_price_eur_kwh", 3.0), ("pv_kw", 0.2), ("load_kw", 2.0)]:
        distorted.loc[final_day, col] = distorted.loc[final_day, col] * factor
    config = _backtest_config(["seasonal_naive"]); config["backtest"]["mode"] = "deterministic"
    a = _backtest(base, config)["daily"][0]
    b = _backtest(distorted, config)["daily"][0]
    assert a["charge_kw"] == b["charge_kw"] and a["discharge_kw"] == b["discharge_kw"]
    assert a["realised_cost_eur"] != b["realised_cost_eur"]
    assert a["realised_cost_risk_neutral_eur"] is None


def test_parallel_backtest_matches_sequential(tmp_path):
    from battery_schedule.pipeline import _backtest
    path = tmp_path / "history.csv"; make_demo_data(path, days=20)
    history = read_history(path, "1h")
    config = _backtest_config(["seasonal_naive"], days=6)
    config["backtest"].update(mode="deterministic", stride_days=2)
    sequential = _backtest(history, config)
    config["backtest"]["workers"] = 2
    parallel = _backtest(history, config)
    assert sequential["days"] == parallel["days"] == 3
    assert [d["realised_cost_eur"] for d in sequential["daily"]] == [d["realised_cost_eur"] for d in parallel["daily"]]


def test_cycle_cap_limits_delivered_energy():
    horizon = 24
    price = np.where(np.arange(horizon) % 12 < 6, 0.02, 0.30)  # two big spreads a day: wants >1 cycle
    scenarios = np.stack([np.zeros((1, horizon)), price[None, :]], axis=-1)
    capped = BatteryConfig(2000, 1000, 1000, .92, .92, 1000, 1000, .001, 1.0, .9, 0.0, 1.0)
    uncapped = BatteryConfig(2000, 1000, 1000, .92, .92, 1000, 1000, .001, 1.0, .9, 0.0, None)
    pv = np.zeros(horizon)
    assert solve_schedule(uncapped_scen := scenarios, pv, price, uncapped).discharge_kw.sum() > 2000 + 1e-6
    assert solve_schedule(uncapped_scen, pv, price, capped).discharge_kw.sum() <= 2000 + 1e-6


def test_no_site_load_skips_the_load_model(tmp_path):
    from battery_schedule.forecast import forecast_once
    path = tmp_path / "history.csv"; make_demo_data(path, days=12)
    history = read_history(path, "1h")
    history["load_kw"] = 0.0
    mean = forecast_once(history, 24, "lasso_ar")
    assert (mean["load_kw"] == 0.0).all()


def test_aligned_lag_is_the_value_lag_hours_before_the_target_hour():
    import pandas as pd
    import pytest

    from battery_schedule.forecast import _aligned_lag
    series = pd.Series(range(400), index=pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC"))
    for step in (1, 7, 24):
        for lag in (24, 48, 168):
            assert _aligned_lag(series, lag, step).iloc[300] == series.iloc[300 + step - lag]
    with pytest.raises(ValueError):
        _aligned_lag(series, 12, 24)


def test_aligned_candidates_cannot_see_the_day_they_are_priced_on(tmp_path):
    from battery_schedule.pipeline import _backtest
    path = tmp_path / "history.csv"; make_demo_data(path, days=16)
    base = read_history(path, "1h")
    distorted = base.copy()
    final_day = distorted.index[-24:]
    for col, factor in [("import_price_eur_kwh", 3.0), ("export_price_eur_kwh", 3.0), ("pv_kw", 0.2), ("load_kw", 2.0)]:
        distorted.loc[final_day, col] = distorted.loc[final_day, col] * factor
    config = _backtest_config(["lasso_ar_aligned"]); config["backtest"]["mode"] = "deterministic"
    a = _backtest(base, config)["daily"][0]
    b = _backtest(distorted, config)["daily"][0]
    # Tolerance, not equality: multithreaded linear algebra differs in the last digits between runs.
    # A real leak (the final day is distorted 3x) would move these by orders of magnitude more.
    assert np.allclose(a["forecast_price_eur_kwh"], b["forecast_price_eur_kwh"], atol=1e-9)
    assert np.allclose(a["charge_kw"], b["charge_kw"], atol=1e-6) and np.allclose(a["discharge_kw"], b["discharge_kw"], atol=1e-6)
