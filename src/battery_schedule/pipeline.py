import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .forecast import make_scenarios, select_and_forecast
from .optimise import solve_schedule


def run(config: dict) -> dict:
    from .data import read_history
    data_cfg=config["data"]; fc_cfg=config["forecast"]; sc_cfg=config["scenarios"]
    history=read_history(data_cfg["history_csv"], data_cfg["frequency"])
    horizon=data_cfg["horizon_steps"]
    # Tomorrow's actual dispatch needs one committed schedule, so this step still
    # auto-selects by validation MAE. The backtest below does NOT: every configured
    # candidate gets its own realised economics, since MAE-best is not profit-best
    # (see README) and picking only the "winner" would hide that.
    forecast=select_and_forecast(history, horizon, fc_cfg["validation_days"], fc_cfg["candidates"], fc_cfg.get("holiday_country"), fc_cfg.get("lookback_days"), fc_cfg.get("household_price_exog", True))
    scenarios=make_scenarios(forecast, sc_cfg["count"], sc_cfg["residual_block_steps"], fc_cfg["random_seed"])
    # PV's daily profile is usually known/forecast separately. Seasonal persistence is a safe default interface.
    pv=np.resize(history["pv_kw"].iloc[-168:].to_numpy(), horizon)
    # export_price_eur_kwh (if present) is historical, indexed on the past; the fallback is the
    # forecast, indexed on the future — fillna by position, not by (non-overlapping) index.
    exp_hist=history["export_price_eur_kwh"].iloc[-horizon:].to_numpy()
    exp_fallback=forecast.mean.import_price_eur_kwh.to_numpy() * config["optimization"].export_price_ratio
    exp=np.where(np.isnan(exp_hist), exp_fallback, exp_hist)
    schedule=solve_schedule(scenarios, pv, exp, config["optimization"], cvar_alpha=config["optimization"].cvar_alpha, cvar_weight=config["optimization"].cvar_weight)
    schedule.index=forecast.mean.index
    output=Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    forecast_out=forecast.mean.copy(); forecast_out["pv_kw"] = pv
    forecast_out.to_csv(output/"forecast.csv"); schedule.to_csv(output/"schedule.csv")
    metrics={"next_day_selected_model":forecast.model_name,"next_day_validation_mae":forecast.validation_mae,"scenario_count":sc_cfg["count"],"horizon_steps":horizon}
    (output/"run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if config.get("backtest", {}).get("enabled"): metrics["backtest"]=_backtest(history, config)
    (output/"run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def _realised_cost(actual: pd.DataFrame, charge: np.ndarray, discharge: np.ndarray, exp: np.ndarray) -> float:
    net=actual.load_kw.to_numpy()-actual.pv_kw.to_numpy()+charge-discharge
    return float((np.maximum(net,0)*actual.import_price_eur_kwh.to_numpy()-np.maximum(-net,0)*exp).sum())

def _top_bottom_recall(forecast_price: np.ndarray, actual_price: np.ndarray, k: int = 4) -> tuple[float, float]:
    """Share of the actual top-k (most expensive) / bottom-k (cheapest) hours that the
    forecast also puts in its own top-k / bottom-k. Unlike tau (which weighs every hour
    pair equally), this targets exactly the hours a battery's arbitrage decision depends
    on."""
    actual_top = set(np.argsort(actual_price)[-k:]); actual_bottom = set(np.argsort(actual_price)[:k])
    forecast_top = set(np.argsort(forecast_price)[-k:]); forecast_bottom = set(np.argsort(forecast_price)[:k])
    return len(actual_top & forecast_top) / k, len(actual_bottom & forecast_bottom) / k

def _backtest(history: pd.DataFrame, config: dict) -> dict:
    """Per-candidate operational backtest: every configured forecast candidate is evaluated
    on its own (no MAE-based auto-selection here — that would only ever show the "winner"'s
    economics and hide whether MAE-best and profit-best are the same model, which they
    often aren't; see README), plus a perfect-foresight oracle for context. Also records a
    per-day, per-candidate breakdown (mae/tau/extreme-hour recall/realised cost) under
    "daily", since 30-day aggregates collapse each candidate to one point and hide whether
    tau tracks realised cost day-by-day."""
    h=config["data"]["horizon_steps"]; days=config["backtest"]["days"]; start=len(history)-days*h
    cfg=config["optimization"]; candidates=config["forecast"]["candidates"]
    holiday_country=config["forecast"].get("holiday_country"); lookback_days=config["forecast"].get("lookback_days")
    household_price_exog=config["forecast"].get("household_price_exog", True)

    per_candidate={name: {"mae_load":[], "mae_price":[], "tau":[], "realised_cost":[]} for name in candidates}
    oracle_costs=[]; baseline_costs=[]; daily=[]

    origins=list(range(start, len(history)-h+1, h))
    for i, origin in enumerate(origins):
        train, actual = history.iloc[:origin], history.iloc[origin:origin+h]
        date=str(actual.index[0].date())
        exp=actual["export_price_eur_kwh"].fillna(actual.import_price_eur_kwh*cfg.export_price_ratio).to_numpy()

        net_no_battery=actual.load_kw.to_numpy()-actual.pv_kw.to_numpy()
        baseline_cost=float((np.maximum(net_no_battery,0)*actual.import_price_eur_kwh.to_numpy()-np.maximum(-net_no_battery,0)*exp).sum())
        baseline_costs.append(baseline_cost)

        # Perfect-foresight oracle: same LP, one "scenario" equal to what actually happened.
        oracle_scenario=np.stack([actual.load_kw.to_numpy(), actual.import_price_eur_kwh.to_numpy()], axis=-1)[None,:,:]
        oracle_sched=solve_schedule(oracle_scenario, actual.pv_kw.to_numpy(), exp, cfg, cvar_alpha=cfg.cvar_alpha, cvar_weight=0.0)
        oracle_cost=_realised_cost(actual, oracle_sched.charge_kw.to_numpy(), oracle_sched.discharge_kw.to_numpy(), exp)
        oracle_costs.append(oracle_cost)

        validation_days=min(config["forecast"]["validation_days"], max(2,(len(train)-192)//24))
        for name in candidates:
            # A single-item candidates list: the same rolling-origin validation and final
            # refit select_and_forecast always does, just not competing against the others.
            fc=select_and_forecast(train, h, validation_days, [name], holiday_country, lookback_days, household_price_exog)

            actual_price=actual["import_price_eur_kwh"].to_numpy()
            forecast_price=fc.mean["import_price_eur_kwh"].to_numpy()
            tau,_=kendalltau(forecast_price, actual_price)
            tau=None if tau is None or math.isnan(tau) else float(tau)
            mae_price=float(np.mean(np.abs(forecast_price-actual_price)))
            top4_recall, bottom4_recall=_top_bottom_recall(forecast_price, actual_price)

            scen=make_scenarios(fc, min(30,config["scenarios"]["count"]), config["scenarios"]["residual_block_steps"], origin)
            sched=solve_schedule(scen, actual.pv_kw.to_numpy(), exp, cfg, cvar_alpha=cfg.cvar_alpha, cvar_weight=cfg.cvar_weight)
            realised_cost=_realised_cost(actual, sched.charge_kw.to_numpy(), sched.discharge_kw.to_numpy(), exp)

            per_candidate[name]["mae_load"].append(float(np.mean(np.abs(fc.mean["load_kw"].to_numpy()-actual["load_kw"].to_numpy()))))
            per_candidate[name]["mae_price"].append(mae_price)
            per_candidate[name]["tau"].append(tau)
            per_candidate[name]["realised_cost"].append(realised_cost)
            daily.append({
                "date": date, "candidate": name, "mae_price_eur_kwh": mae_price, "kendall_tau": tau,
                "top4_expensive_recall": top4_recall, "bottom4_cheap_recall": bottom4_recall,
                "realised_cost_eur": realised_cost, "baseline_cost_eur": baseline_cost, "oracle_cost_eur": oracle_cost,
            })

        print(f"backtest day {i + 1}/{len(origins)} done ({date})", flush=True)

    baseline_total=sum(baseline_costs); oracle_total=sum(oracle_costs)
    result={
        "days": len(baseline_costs),
        "baseline_no_battery_eur": baseline_total,
        "oracle_perfect_foresight_eur": oracle_total,
        "oracle_vs_baseline_pct": 100*(oracle_total/baseline_total-1) if baseline_total else None,
        "candidates": {},
        "daily": daily,
    }
    for name in candidates:
        stats=per_candidate[name]; total_cost=sum(stats["realised_cost"])
        taus=[t for t in stats["tau"] if t is not None]
        recalls=[d for d in daily if d["candidate"]==name]
        result["candidates"][name]={
            "mean_mae_load_kw": float(np.mean(stats["mae_load"])),
            "mean_mae_price_eur_kwh": float(np.mean(stats["mae_price"])),
            "mean_kendall_tau_price": float(np.mean(taus)) if taus else None,
            "mean_top4_expensive_recall": float(np.mean([d["top4_expensive_recall"] for d in recalls])),
            "mean_bottom4_cheap_recall": float(np.mean([d["bottom4_cheap_recall"] for d in recalls])),
            "realised_grid_cost_eur": total_cost,
            "mean_daily_cost_eur": float(total_cost/len(baseline_costs)) if baseline_costs else None,
            "vs_baseline_pct": 100*(total_cost/baseline_total-1) if baseline_total else None,
            "gap_to_oracle_eur": total_cost-oracle_total,
        }
    return result
