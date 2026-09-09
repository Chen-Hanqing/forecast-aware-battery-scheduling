import pandas as pd
from battery_schedule.data import make_demo_data, read_history
from battery_schedule.forecast import select_and_forecast, make_scenarios
from battery_schedule.optimise import solve_schedule
from battery_schedule.config import BatteryConfig

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
