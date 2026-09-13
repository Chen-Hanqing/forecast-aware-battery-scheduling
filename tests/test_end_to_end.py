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
