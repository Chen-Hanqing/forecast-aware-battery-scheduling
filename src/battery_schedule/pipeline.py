import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .forecast import (
    ForecastResult,
    _seasonal_values,
    forecast_once,
    make_scenarios,
    select_and_forecast,
)
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

def _plan_deterministic(train: pd.DataFrame, mean: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Schedule from the point forecast alone. Like `_plan`, it sees only `train` and the forecast:
    PV is a seasonal-persistence forecast and export earns `export_price_ratio` x the forecast price."""
    cfg=config["optimization"]; h=len(mean)
    forecast_price=mean["import_price_eur_kwh"].to_numpy()
    pv=_seasonal_values(train, h, ["pv_kw"])[:,0]
    point=np.stack([mean["load_kw"].to_numpy(), forecast_price], axis=-1)[None,:,:]
    return solve_schedule(point, pv, cfg.export_price_ratio*forecast_price, cfg, cvar_alpha=cfg.cvar_alpha, cvar_weight=0.0)

def _plan(train: pd.DataFrame, forecast: ForecastResult, config: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Commit to a schedule using only what is knowable at the origin: `train` (history before
    the origin) and the candidate's own forecast. There is deliberately no `actual` argument, so
    the future can't leak in. In particular PV is a seasonal-persistence forecast and the export
    price is `export_price_ratio` x the *forecast* price (in this project export earns a share of
    the day-ahead price, so it is as unknown at decision time as the import price is).

    Returns (stochastic, deterministic, risk_neutral):
      stochastic     the scenario/CVaR schedule the pipeline actually uses (cvar_weight from config);
      deterministic  the same LP on the point forecast alone;
      risk_neutral   the same scenarios but cvar_weight=0 (expected cost only).
    Realised cost of deterministic minus risk_neutral is the (ex-post) value of modelling
    uncertainty at all; risk_neutral minus stochastic is what the CVaR risk aversion costs (or
    earns) on top. Comparing deterministic with the CVaR schedule directly would mix the two."""
    cfg=config["optimization"]; h=len(forecast.mean)
    pv=_seasonal_values(train, h, ["pv_kw"])[:,0]
    scen=make_scenarios(forecast, min(30,config["scenarios"]["count"]), config["scenarios"]["residual_block_steps"], seed)
    exp_scen=cfg.export_price_ratio*scen[...,1]  # each scenario exports at its own price path
    stochastic=solve_schedule(scen, pv, exp_scen, cfg, cvar_alpha=cfg.cvar_alpha, cvar_weight=cfg.cvar_weight)
    deterministic=_plan_deterministic(train, forecast.mean, config)
    risk_neutral=solve_schedule(scen, pv, exp_scen, cfg, cvar_alpha=cfg.cvar_alpha, cvar_weight=0.0)
    return stochastic, deterministic, risk_neutral

def _evaluate_origin(history: pd.DataFrame, config: dict, origin: int, pred_cache: dict | None) -> dict:
    """Everything the backtest records for one origin: baseline and oracle cost for the day, and one
    record per candidate. "stochastic" mode fits a validation window, builds scenarios and solves the
    CVaR, deterministic and risk-neutral schedules; "deterministic" mode makes one forecast per
    candidate and solves only the point-forecast LP (cheap enough for multi-year backtests)."""
    h=config["data"]["horizon_steps"]; cfg=config["optimization"]; bt=config["backtest"]
    candidates=config["forecast"]["candidates"]; mode=bt.get("mode","stochastic")
    holiday_country=config["forecast"].get("holiday_country"); lookback_days=config["forecast"].get("lookback_days")
    household_price_exog=config["forecast"].get("household_price_exog", True)

    train, actual = history.iloc[:origin], history.iloc[origin:origin+h]
    date=str(actual.index[0].date())
    # What the meter actually paid/earned: used only to price schedules after the fact.
    exp_real=actual["export_price_eur_kwh"].fillna(actual.import_price_eur_kwh*cfg.export_price_ratio).to_numpy()

    net_no_battery=actual.load_kw.to_numpy()-actual.pv_kw.to_numpy()
    baseline_cost=float((np.maximum(net_no_battery,0)*actual.import_price_eur_kwh.to_numpy()-np.maximum(-net_no_battery,0)*exp_real).sum())

    # Perfect-foresight oracle: same LP, one "scenario" equal to what actually happened.
    oracle_scenario=np.stack([actual.load_kw.to_numpy(), actual.import_price_eur_kwh.to_numpy()], axis=-1)[None,:,:]
    oracle_sched=solve_schedule(oracle_scenario, actual.pv_kw.to_numpy(), exp_real, cfg, cvar_alpha=cfg.cvar_alpha, cvar_weight=0.0)
    oracle_cost=_realised_cost(actual, oracle_sched.charge_kw.to_numpy(), oracle_sched.discharge_kw.to_numpy(), exp_real)

    validation_days=min(config["forecast"]["validation_days"], max(2,(len(train)-192)//24))
    actual_price=actual["import_price_eur_kwh"].to_numpy()
    rows=[]
    for name in candidates:
        if mode=="deterministic":
            mean=forecast_once(train, h, name, holiday_country, lookback_days, household_price_exog)
            sched=sched_det=_plan_deterministic(train, mean, config); sched_rn=None
        else:
            # A single-item candidates list: the same rolling-origin validation and final
            # refit select_and_forecast always does, just not competing against the others.
            fc=select_and_forecast(train, h, validation_days, [name], holiday_country, lookback_days, household_price_exog, pred_cache)
            mean=fc.mean
            sched, sched_det, sched_rn=_plan(train, fc, config, origin)
        forecast_price=mean["import_price_eur_kwh"].to_numpy()
        tau,_=kendalltau(forecast_price, actual_price)
        top4_recall, bottom4_recall=_top_bottom_recall(forecast_price, actual_price)
        cost=lambda sc: _realised_cost(actual, sc.charge_kw.to_numpy(), sc.discharge_kw.to_numpy(), exp_real)
        rows.append({
            "date": date, "candidate": name,
            "mae_price_eur_kwh": float(np.mean(np.abs(forecast_price-actual_price))),
            "mae_load_kw": float(np.mean(np.abs(mean["load_kw"].to_numpy()-actual["load_kw"].to_numpy()))),
            "kendall_tau": None if tau is None or math.isnan(tau) else float(tau),
            "top4_expensive_recall": top4_recall, "bottom4_cheap_recall": bottom4_recall,
            "realised_cost_eur": cost(sched), "realised_cost_deterministic_eur": cost(sched_det),
            "realised_cost_risk_neutral_eur": None if sched_rn is None else cost(sched_rn),
            "baseline_cost_eur": baseline_cost, "oracle_cost_eur": oracle_cost,
            "forecast_price_eur_kwh": [float(v) for v in forecast_price],
            "actual_price_eur_kwh": [float(v) for v in actual_price],
            "charge_kw": [float(v) for v in sched.charge_kw], "discharge_kw": [float(v) for v in sched.discharge_kw],
        })
    return {"date": date, "baseline_cost_eur": baseline_cost, "oracle_cost_eur": oracle_cost, "rows": rows}

_WORKER: dict = {}

def _init_worker(history: pd.DataFrame, config: dict) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)  # one thread per worker process; parallelism comes from the origins
    _WORKER["history"]=history; _WORKER["config"]=config

def _evaluate_origin_in_worker(origin: int) -> dict:
    return _evaluate_origin(_WORKER["history"], _WORKER["config"], origin, None)

def _origins(history: pd.DataFrame, config: dict) -> list[int]:
    """Backtest origins: the last `days` days, every `stride_days`-th day, never earlier than the
    warm-up needed to fit the first model."""
    h=config["data"]["horizon_steps"]; bt=config["backtest"]
    warmup=bt.get("warmup_days", (config["forecast"].get("lookback_days") or 0)+config["forecast"]["validation_days"]+2)
    start=max(len(history)-bt["days"]*h, warmup*h)
    return list(range(start, len(history)-h+1, h*bt.get("stride_days", 1)))

def _backtest(history: pd.DataFrame, config: dict) -> dict:
    """Per-candidate operational backtest: every configured forecast candidate is evaluated
    on its own (no MAE-based auto-selection here — that would only ever show the "winner"'s
    economics and hide whether MAE-best and profit-best are the same model, which they
    often aren't; see README), plus a perfect-foresight oracle for context.

    At each origin the schedule comes from `_plan` (train + forecast only) and is only then
    priced against what actually happened. Also records a per-day, per-candidate breakdown under
    "daily" (metrics, forecast/actual price vectors, schedule, realised costs), since aggregates
    collapse each candidate to one point and hide day-to-day behaviour.

    Config (`backtest`): `days` (window length), `mode` ("stochastic" or "deterministic"),
    `stride_days` (evaluate every n-th day), `workers` (processes; deterministic mode only)."""
    cfg=config["optimization"]; bt=config["backtest"]; candidates=config["forecast"]["candidates"]
    mode=bt.get("mode","stochastic"); workers=int(bt.get("workers",1))
    if workers>1 and mode!="deterministic":
        raise ValueError("backtest.workers > 1 needs mode: deterministic (stochastic mode shares a forecast cache across origins)")
    origins=_origins(history, config)

    results=[]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(history, config)) as pool:
            for i, res in enumerate(pool.map(_evaluate_origin_in_worker, origins)):
                results.append(res); print(f"backtest day {i + 1}/{len(origins)} done ({res['date']})", flush=True)
    else:
        pred_cache={}
        for i, origin in enumerate(origins):
            res=_evaluate_origin(history, config, origin, pred_cache)
            results.append(res); print(f"backtest day {i + 1}/{len(origins)} done ({res['date']})", flush=True)

    daily=[row for res in results for row in res["rows"]]
    baseline_total=sum(r["baseline_cost_eur"] for r in results); oracle_total=sum(r["oracle_cost_eur"] for r in results)
    n_days=len(results)
    result={
        "days": n_days, "mode": mode, "stride_days": bt.get("stride_days", 1),
        "battery": {"power_kw": cfg.charge_power_kw, "capacity_kwh": cfg.capacity_kwh,
                    "max_cycles_per_day": cfg.max_cycles_per_day, "degradation_eur_per_kwh": cfg.degradation_eur_per_kwh},
        "baseline_no_battery_eur": baseline_total,
        "oracle_perfect_foresight_eur": oracle_total,
        "oracle_vs_baseline_pct": 100*(oracle_total/baseline_total-1) if baseline_total else None,
        "candidates": {},
        "daily": daily,
    }
    for name in candidates:
        rows=[d for d in daily if d["candidate"]==name]
        total_cost=sum(d["realised_cost_eur"] for d in rows); total_det=sum(d["realised_cost_deterministic_eur"] for d in rows)
        rn=[d["realised_cost_risk_neutral_eur"] for d in rows]; total_rn=None if any(v is None for v in rn) else sum(rn)
        taus=[d["kendall_tau"] for d in rows if d["kendall_tau"] is not None]
        result["candidates"][name]={
            "mean_mae_load_kw": float(np.mean([d["mae_load_kw"] for d in rows])),
            "mean_mae_price_eur_kwh": float(np.mean([d["mae_price_eur_kwh"] for d in rows])),
            "mean_kendall_tau_price": float(np.mean(taus)) if taus else None,
            "mean_top4_expensive_recall": float(np.mean([d["top4_expensive_recall"] for d in rows])),
            "mean_bottom4_cheap_recall": float(np.mean([d["bottom4_cheap_recall"] for d in rows])),
            "realised_grid_cost_eur": total_cost,
            "realised_grid_cost_deterministic_eur": total_det,
            "realised_grid_cost_risk_neutral_eur": total_rn,
            "value_of_modelling_uncertainty_eur": None if total_rn is None else total_det-total_rn,
            "cost_of_cvar_risk_aversion_eur": None if total_rn is None else total_cost-total_rn,
            "mean_daily_cost_eur": float(total_cost/n_days) if n_days else None,
            "vs_baseline_pct": 100*(total_cost/baseline_total-1) if baseline_total else None,
            "gap_to_oracle_eur": total_cost-oracle_total,
        }
    return result
