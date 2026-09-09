import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linprog
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass
from typing import List, Dict

# =========================================================================
# 1. 配置定义
# =========================================================================
@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 13.5
    charge_power_kw: float = 5.0
    discharge_power_kw: float = 5.0
    eta_charge: float = 0.96
    eta_discharge: float = 0.96
    initial_soc_kwh: float = 6.0
    terminal_soc_kwh: float = 6.0
    degradation_eur_per_kwh: float = 0.01
    export_price_ratio: float = 0.75
    cvar_alpha: float = 0.90
    cvar_weight: float = 0.20

# =========================================================================
# 2. 特征工程与 LightGBM 分位数预测器
# =========================================================================
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取周期与 Lag 特征（严格日前时序边界，Lag >= 24h）"""
    data = df.copy()
    idx = data.index
    data["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    data["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    
    for col in ["load_kw", "import_price_eur_kwh"]:
        if col in data.columns:
            for lag in [24, 48, 168]:
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)
            shifted_24 = data[col].shift(24)
            data[f"{col}_roll_mean_24h"] = shifted_24.rolling(24, min_periods=12).mean()
    return data

class DirectQuantileForecaster:
    """直接多步分位数回归预测器"""
    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90], horizon: int = 24):
        self.quantiles = sorted(quantiles)
        self.horizon = horizon
        self.models: Dict[str, Dict[int, Dict[float, lgb.LGBMRegressor]]] = {}
        self.feature_cols: List[str] = []

    def fit(self, df_features: pd.DataFrame, target_cols: List[str]):
        self.feature_cols = [
            c for c in df_features.columns 
            if c not in target_cols and not c.startswith("target_")
        ]
        
        for target in target_cols:
            self.models[target] = {}
            train_df = df_features.copy()
            for step in range(1, self.horizon + 1):
                train_df[f"target_step_{step}"] = train_df[target].shift(-step)
            
            clean_train = train_df.dropna()
            X = clean_train[self.feature_cols]
            
            for step in range(1, self.horizon + 1):
                self.models[target][step] = {}
                y = clean_train[f"target_step_{step}"]
                for q in self.quantiles:
                    model = lgb.LGBMRegressor(
                        objective="quantile", alpha=q, n_estimators=100,
                        learning_rate=0.05, num_leaves=15, verbose=-1, random_state=42
                    )
                    model.fit(X, y)
                    self.models[target][step][q] = model

    def predict(self, pred_row: pd.DataFrame, target: str) -> pd.DataFrame:
        X_pred = pred_row[self.feature_cols]
        preds = {f"q_{int(q*100)}": [] for q in self.quantiles}
        
        for step in range(1, self.horizon + 1):
            step_vals = [float(self.models[target][step][q].predict(X_pred)[0]) for q in self.quantiles]
            step_vals_sorted = np.sort(step_vals) # 消除分位数交叉
            for idx, q in enumerate(self.quantiles):
                preds[f"q_{int(q*100)}"].append(max(0.0, step_vals_sorted[idx]))
                
        return pd.DataFrame(preds, index=pd.RangeIndex(1, self.horizon + 1))

# =========================================================================
# 3. Gaussian Copula 联合场景生成器
# =========================================================================
class CopulaScenarioGenerator:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        self.copula_chol = None
        self.dim = 0

    def fit(self, val_residuals: np.ndarray):
        """输入联合残差矩阵 [N_samples, 2*T]"""
        self.dim = val_residuals.shape[1]
        ranks = np.argsort(np.argsort(val_residuals, axis=0), axis=0) + 1
        u_matrix = ranks / (len(val_residuals) + 1.0)
        z_matrix = norm.ppf(u_matrix)
        
        corr_matrix = np.corrcoef(z_matrix, rowvar=False)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2.0
        min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
        if min_eig < 1e-6:
            corr_matrix += (1e-6 - min_eig) * np.eye(self.dim)
        self.copula_chol = np.linalg.cholesky(corr_matrix)

    def generate(self, q_load: pd.DataFrame, q_price: pd.DataFrame, n_scenarios: int = 80) -> np.ndarray:
        T = len(q_load)
        z_uncorr = self.rng.standard_normal((n_scenarios, self.dim))
        u_samples = np.clip(norm.cdf(z_uncorr @ self.copula_chol.T), 1e-4, 1.0 - 1e-4)
        
        u_load, u_price = u_samples[:, :T], u_samples[:, T:]
        scenarios = np.zeros((n_scenarios, T, 2), dtype=float)
        
        for t in range(T):
            scenarios[:, t, 0] = self._invert_pchip(u_load[:, t], q_load.iloc[t].to_numpy())
            scenarios[:, t, 1] = self._invert_pchip(u_price[:, t], q_price.iloc[t].to_numpy())
        return scenarios

    def _invert_pchip(self, u_vals: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
        q10, q50, q90 = q_vals[0], q_vals[1], q_vals[2]
        iqr = max(1e-3, q90 - q10)
        u_anchors = np.array([0.00, 0.10, 0.50, 0.90, 1.00])
        v_anchors = np.array([max(0.0, q10 - 0.8 * iqr), q10, q50, q90, q90 + 0.8 * iqr])
        return np.maximum(0.0, PchipInterpolator(u_anchors, v_anchors)(u_vals))

# =========================================================================
# 4. CVaR 两阶段随机线性规划求解器
# =========================================================================
def solve_schedule(
    scenarios: np.ndarray, 
    pv_kw: np.ndarray, 
    export_prices: np.ndarray, 
    cfg: BatteryConfig, 
    dt_h: float = 1.0
) -> pd.DataFrame:
    S, T, _ = scenarios.shape
    n_action = 3 * T
    imp0 = n_action
    exp0 = imp0 + S * T
    cost0 = exp0 + S * T
    z = cost0 + S
    u0 = z + 1
    N = u0 + S

    # 目标函数系数
    c = np.zeros(N)
    c[:T] = cfg.degradation_eur_per_kwh * dt_h
    c[T:2*T] = cfg.degradation_eur_per_kwh * dt_h
    c[cost0:cost0+S] = (1 - cfg.cvar_weight) / S
    c[z] = cfg.cvar_weight
    c[u0:] = cfg.cvar_weight / (S * (1 - cfg.cvar_alpha))

    max_grid_flow = float(np.max(scenarios[..., 0]) + np.max(pv_kw) + cfg.charge_power_kw + cfg.discharge_power_kw)
    bounds = (
        [(0, cfg.charge_power_kw)] * T +
        [(0, cfg.discharge_power_kw)] * T +
        [(0, cfg.capacity_kwh)] * T +
        [(0, max_grid_flow)] * (2 * S * T) +
        [(0, None)] * S +
        [(None, None)] +
        [(0, None)] * S
    )

    Aeq, beq = [], []
    # 电池电量动态 (SOC Dynamics)
    for t in range(T):
        row = np.zeros(N)
        row[2*T+t] = 1
        row[t] = -cfg.eta_charge * dt_h
        row[T+t] = dt_h / cfg.eta_discharge
        if t:
            row[2*T+t-1] = -1
            rhs = 0.0
        else:
            rhs = cfg.initial_soc_kwh
        Aeq.append(row)
        beq.append(rhs)
        
    # 期末 SOC 约束
    terminal = np.zeros(N)
    terminal[3*T-1] = 1
    Aeq.append(terminal)
    beq.append(cfg.terminal_soc_kwh)

    # 场景电表平衡与购售电成本
    for s in range(S):
        for t in range(T):
            row = np.zeros(N)
            row[imp0+s*T+t] = 1
            row[exp0+s*T+t] = -1
            row[T+t] = 1
            row[t] = -1
            Aeq.append(row)
            beq.append(scenarios[s, t, 0] - pv_kw[t])
            
        row = np.zeros(N)
        row[cost0+s] = 1
        for t in range(T):
            row[imp0+s*T+t] -= scenarios[s, t, 1] * dt_h
            row[exp0+s*T+t] += export_prices[t] * dt_h
        Aeq.append(row)
        beq.append(0.0)

    # CVaR 超额损失约束 (cost_s - z <= u_s)
    Aub, bub = [], []
    for s in range(S):
        row = np.zeros(N)
        row[cost0+s] = 1
        row[z] = -1
        row[u0+s] = -1
        Aub.append(row)
        bub.append(0.0)

    res = linprog(c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub), A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    x = res.x
    return pd.DataFrame({
        "charge_kw": np.round(x[:T], 3),
        "discharge_kw": np.round(x[T:2*T], 3),
        "soc_kwh": np.round(x[2*T:3*T], 3)
    }, index=pd.RangeIndex(T, name="hour"))

# =========================================================================
# 5. 端到端执行主流程 (Main Pipeline Entry)
# =========================================================================
def run_end_to_end_pipeline():
    print("========== 1. Generating Ground Truth History Data ==========")
    rng = np.random.default_rng(42)
    n_hours = 90 * 24
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    h = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    
    # 模拟真实用电负荷、日前电价与光伏
    base_load = 1.8 + 1.2 * np.exp(-((h - 19) / 3)**2) + 0.4 * (dow < 5) + rng.normal(0, 0.15, n_hours)
    base_price = 0.12 + 0.18 * np.exp(-((h - 19) / 3.5)**2) + 0.03 * (dow < 5) + rng.normal(0, 0.015, n_hours)
    base_pv = np.maximum(0, 4.0 * np.sin(np.pi * (h - 6) / 12)) * (0.8 + 0.2 * rng.random(n_hours))
    
    history_df = pd.DataFrame({
        "load_kw": np.maximum(0.1, base_load),
        "import_price_eur_kwh": np.maximum(0.02, base_price),
        "pv_kw": base_pv
    }, index=idx)
    
    print(f"Loaded history records: {len(history_df)} hours (90 days)")

    print("\n========== 2. Training Multi-Step Quantile LightGBM ==========")
    features_df = extract_features(history_df)
    train_cutoff = len(features_df) - 48
    train_features = features_df.iloc[:train_cutoff]
    
    forecaster = DirectQuantileForecaster(quantiles=[0.10, 0.50, 0.90], horizon=24)
    forecaster.fit(train_features, target_cols=["load_kw", "import_price_eur_kwh"])
    
    # 在决策点预测未来 24 步
    decision_origin = train_cutoff
    pred_row = features_df.iloc[[decision_origin]]
    q_load = forecaster.predict(pred_row, target="load_kw")
    q_price = forecaster.predict(pred_row, target="import_price_eur_kwh")
    print("Probabilistic forecasting complete. Sample q50 load (first 4h):", q_load["q_50"].values[:4])

    print("\n========== 3. Fitting Copula & Sampling Joint Scenarios ==========")
    # 构造历史验证集联合残差
    val_samples = 120
    sim_res_load = rng.normal(0, 0.25, (val_samples, 24))
    sim_res_price = rng.normal(0, 0.02, (val_samples, 24))
    joint_val_residuals = np.hstack([sim_res_load, sim_res_price])
    
    copula = CopulaScenarioGenerator(random_seed=42)
    copula.fit(joint_val_residuals)
    scenarios_matrix = copula.generate(q_load, q_price, n_scenarios=80)
    print(f"Generated joint scenario matrix: {scenarios_matrix.shape} (Scenarios x Horizons x [Load, Price])")

    print("\n========== 4. Solving Stochastic CVaR Battery Dispatch ==========")
    cfg = BatteryConfig()
    # 提取未来 24h 的光伏预估与售电价格
    future_slice = history_df.iloc[decision_origin + 1 : decision_origin + 25]
    pv_forecast = future_slice["pv_kw"].to_numpy()
    export_prices = q_price["q_50"].to_numpy() * cfg.export_price_ratio
    
    schedule_df = solve_schedule(
        scenarios=scenarios_matrix,
        pv_kw=pv_forecast,
        export_prices=export_prices,
        cfg=cfg
    )
    
    print("\n========== 5. Optimal Non-Anticipative Battery Schedule ==========")
    print(schedule_df.to_string())

if __name__ == "__main__":
    run_end_to_end_pipeline()