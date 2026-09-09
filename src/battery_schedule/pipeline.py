from pathlib import Path
import json
import numpy as np
import pandas as pd
from .forecast import select_and_forecast, make_scenarios
from .optimise import solve_schedule

def run(config: dict) -> dict:
    from .data import read_history
    data_cfg=config["data"]; fc_cfg=config["forecast"]; sc_cfg=config["scenarios"]
    history=read_history(data_cfg["history_csv"], data_cfg["frequency"])
    horizon=data_cfg["horizon_steps"]
    forecast=select_and_forecast(history, horizon, fc_cfg["validation_days"], fc_cfg["candidates"])
    scenarios=make_scenarios(forecast, sc_cfg["count"], sc_cfg["residual_block_steps"], fc_cfg["random_seed"])
    # PV's daily profile is usually known/forecast separately. Seasonal persistence is a safe default interface.
    pv=np.resize(history["pv_kw"].iloc[-168:].to_numpy(), horizon)
    exp=history["export_price_eur_kwh"].iloc[-horizon:].fillna(forecast.mean.import_price_eur_kwh * config["optimization"].export_price_ratio).to_numpy()
    schedule=solve_schedule(scenarios, pv, exp, config["optimization"], cvar_alpha=config["optimization"].cvar_alpha, cvar_weight=config["optimization"].cvar_weight)
    schedule.index=forecast.mean.index
    output=Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    forecast_out=forecast.mean.copy(); forecast_out["pv_kw"] = pv
    forecast_out.to_csv(output/"forecast.csv"); schedule.to_csv(output/"schedule.csv")
    metrics={"selected_model":forecast.model_name,"validation_mae":forecast.validation_mae,"scenario_count":sc_cfg["count"],"horizon_steps":horizon}
    (output/"run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if config.get("backtest", {}).get("enabled"): metrics["backtest"]=_backtest(history, config)
    (output/"run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def _backtest(history: pd.DataFrame, config: dict) -> dict:
    """Operational backtest: schedule against forecasts, then price its realised grid exchanges."""
    h=config["data"]["horizon_steps"]; days=config["backtest"]["days"]; start=len(history)-days*h
    costs=[]
    for origin in range(start, len(history)-h+1, h):
        train, actual=history.iloc[:origin],history.iloc[origin:origin+h]
        fc=select_and_forecast(train,h, min(config["forecast"]["validation_days"], max(2,(len(train)-192)//24)),config["forecast"]["candidates"])
        scen=make_scenarios(fc, min(30,config["scenarios"]["count"]),config["scenarios"]["residual_block_steps"],origin)
        exp=actual["export_price_eur_kwh"].fillna(actual.import_price_eur_kwh*config["optimization"].export_price_ratio).to_numpy()
        sched=solve_schedule(scen,actual.pv_kw.to_numpy(),exp,config["optimization"],cvar_alpha=config["optimization"].cvar_alpha,cvar_weight=config["optimization"].cvar_weight)
        net=actual.load_kw.to_numpy()-actual.pv_kw.to_numpy()+sched.charge_kw.to_numpy()-sched.discharge_kw.to_numpy()
        realised=np.maximum(net,0)*actual.import_price_eur_kwh.to_numpy()-np.maximum(-net,0)*exp
        costs.append(float(realised.sum()))
    return {"days":len(costs),"realised_grid_cost_eur":float(sum(costs)),"mean_daily_cost_eur":float(np.mean(costs))}
