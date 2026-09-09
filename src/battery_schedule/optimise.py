from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from .config import BatteryConfig

def solve_schedule(scenarios: np.ndarray, pv_kw: np.ndarray, export_prices: np.ndarray, cfg: BatteryConfig, dt_h: float = 1.0, cvar_alpha: float = .9, cvar_weight: float = .2) -> pd.DataFrame:
    """Scenario LP with non-anticipative battery actions and CVaR of grid cost."""
    S,T,_ = scenarios.shape; n_action = 3*T  # charge, discharge, SOC
    # Scenario-specific imports, exports and total cost; global VaR and excess variables.
    imp0=n_action; exp0=imp0+S*T; cost0=exp0+S*T; z=cost0+S; u0=z+1; N=u0+S
    c=np.zeros(N); c[:T]=cfg.degradation_eur_per_kwh*dt_h; c[T:2*T]=cfg.degradation_eur_per_kwh*dt_h
    c[cost0:cost0+S]=(1-cvar_weight)/S; c[z]=cvar_weight; c[u0:]=cvar_weight/(S*(1-cvar_alpha))
    max_grid_flow = float(np.max(scenarios[..., 0]) + np.max(pv_kw) + cfg.charge_power_kw + cfg.discharge_power_kw)
    bounds=[]
    bounds += [(0,cfg.charge_power_kw)]*T + [(0,cfg.discharge_power_kw)]*T + [(0,cfg.capacity_kwh)]*T
    bounds += [(0,max_grid_flow)]*(2*S*T) + [(0,None)]*S + [(None,None)] + [(0,None)]*S
    Aeq=[]; beq=[]
    # State of charge dynamics.
    for t in range(T):
        row=np.zeros(N); row[2*T+t]=1; row[t]=-cfg.eta_charge*dt_h; row[T+t]=dt_h/cfg.eta_discharge
        if t: row[2*T+t-1]=-1; rhs=0
        else: rhs=cfg.initial_soc_kwh
        Aeq.append(row); beq.append(rhs)
    terminal=np.zeros(N); terminal[3*T-1]=1; Aeq.append(terminal); beq.append(cfg.terminal_soc_kwh)
    # Each scenario's meter balance and cost definition.
    for s in range(S):
        for t in range(T):
            row=np.zeros(N); row[imp0+s*T+t]=1; row[exp0+s*T+t]=-1; row[T+t]=1; row[t]=-1
            Aeq.append(row); beq.append(scenarios[s,t,0]-pv_kw[t])
        row=np.zeros(N); row[cost0+s]=1
        for t in range(T):
            row[imp0+s*T+t] -= scenarios[s,t,1]*dt_h
            row[exp0+s*T+t] += export_prices[t]*dt_h
        Aeq.append(row); beq.append(0)
    Aub=[]; bub=[]
    for s in range(S):
        # cost_s - z <= u_s
        row=np.zeros(N); row[cost0+s]=1; row[z]=-1; row[u0+s]=-1; Aub.append(row); bub.append(0)
    result=linprog(c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub), A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    if not result.success: raise RuntimeError(f"Optimization failed: {result.message}")
    x=result.x
    idx=pd.RangeIndex(T, name="step")
    return pd.DataFrame({"charge_kw":x[:T], "discharge_kw":x[T:2*T], "soc_kwh":x[2*T:3*T]}, index=idx)
