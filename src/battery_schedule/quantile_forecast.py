import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List

# -------------------------------------------------------------------------
# 1. 特征工程：严格遵循日前预测时序边界 (>= 24h 滞后，杜绝数据泄露)
# -------------------------------------------------------------------------
def create_forecasting_features(df: pd.DataFrame, target_col: str = "load_kw") -> pd.DataFrame:
    """
    输入包含 timestamp 索引、target_col 以及 temp_air (气温) 的 DataFrame
    """
    data = df.copy()
    idx = data.index
    
    # 日历与周期特征
    data["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    data["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    data["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    
    # 气象非线性特征 (度时与热惯性)
    if "temp_air" in data.columns:
        temp = data["temp_air"]
        data["hdh"] = np.maximum(0, 15.5 - temp)
        data["cdh"] = np.maximum(0, temp - 22.0)
        data["temp_ewma_6h"] = temp.ewm(span=6).mean()
        data["temp_ewma_24h"] = temp.ewm(span=24).mean()
        
    # 自回归滞后特征 (必须 >= 24h，满足日前预测要求)
    series = data[target_col]
    for lag in [24, 48, 72, 168]:
        data[f"{target_col}_lag_{lag}"] = series.shift(lag)
        
    # 过去 7 天滚动统计量 (基于 24h 前的数据)
    shifted_24 = series.shift(24)
    data[f"{target_col}_roll_mean_24h"] = shifted_24.rolling(24, min_periods=12).mean()
    data[f"{target_col}_roll_std_24h"] = shifted_24.rolling(24, min_periods=12).std()
    data[f"{target_col}_roll_max_24h"] = shifted_24.rolling(24, min_periods=12).max()
    data[f"{target_col}_roll_min_24h"] = shifted_24.rolling(24, min_periods=12).min()
    
    return data

# -------------------------------------------------------------------------
# 2. 多步直接法分位数预测管道 (Direct Multi-Step Quantile Regressor)
# -------------------------------------------------------------------------
class DirectMultiStepQuantileLGBM:
    def __init__(
        self,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        horizon_steps: int = 24,
        lgb_params: dict = None
    ):
        self.quantiles = sorted(quantiles)
        self.horizon_steps = horizon_steps
        self.models: Dict[int, Dict[float, lgb.Booster]] = {}
        self.feature_cols: List[str] = []
        
        # 默认分位数回归超参数
        self.base_params = {
            "objective": "quantile",
            "metric": "quantile",
            "boosting_type": "gbdt",
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "random_state": 42
        }
        if lgb_params:
            self.base_params.update(lgb_params)

    def fit(self, df_features: pd.DataFrame, target_col: str = "load_kw"):
        """针对未来每个步长 step (1..24) 和每个分位数 alpha 分别拟合独立模型"""
        self.feature_cols = [
            c for c in df_features.columns 
            if c != target_col and not c.startswith("target_step_")
        ]
        
        # 构造每个步长未来的目标值: target_{t+step}
        train_data = df_features.copy()
        for step in range(1, self.horizon_steps + 1):
            train_data[f"target_step_{step}"] = train_data[target_col].shift(-step)
            
        # 丢弃含 NaN 的样本 (包含因 shift 产生的历史滞后缺失和期末目标缺失)
        clean_train = train_data.dropna()
        X = clean_train[self.feature_cols]
        
        for step in range(1, self.horizon_steps + 1):
            self.models[step] = {}
            y = clean_train[f"target_step_{step}"]
            
            for q in self.quantiles:
                params = self.base_params.copy()
                params["alpha"] = q
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X, y)
                self.models[step][q] = model

    def predict(self, current_features: pd.DataFrame) -> pd.DataFrame:
        """
        输入决策时刻 t 的特征行 (DataFrame，1 行)，输出未来 24 步的分位数预测表
        """
        X_pred = current_features[self.feature_cols]
        predictions = {f"q_{int(q*100)}": [] for q in self.quantiles}
        
        for step in range(1, self.horizon_steps + 1):
            step_preds = []
            for q in self.quantiles:
                pred_val = self.models[step][q].predict(X_pred)[0]
                step_preds.append(max(0.0, float(pred_val))) # 物理负荷非负截断
            
            # 分位数交叉后处理 (Quantile Crossing Correction): 确保 q10 <= q50 <= q90
            step_preds_sorted = np.sort(step_preds)
            for idx, q in enumerate(self.quantiles):
                predictions[f"q_{int(q*100)}"].append(step_preds_sorted[idx])
                
        forecast_df = pd.DataFrame(predictions, index=pd.RangeIndex(1, self.horizon_steps + 1, name="step_ahead"))
        return forecast_df

# -------------------------------------------------------------------------
# 3. 模拟运行与回测评估示例 (Pinball Loss 计算)
# -------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """计算单个分位数的 Pinball Loss (Score)"""
    residual = y_true - y_pred
    return float(np.mean(np.maximum(alpha * residual, (alpha - 1) * residual)))

if __name__ == "__main__":
    # 1. 生成 120 天每小时的合成测试数据 (带温度与真实负荷模式)
    rng = np.random.default_rng(42)
    n_hours = 120 * 24
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    
    h = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    temp_sim = 10 + 8 * np.sin(2 * np.pi * (h - 9) / 24) - 3 * (idx.month == 1) + rng.normal(0, 1.5, n_hours)
    base_load = 2.0 + 1.5 * np.exp(-((h - 19)/3)**2) + 0.8 * (dow < 5) + 0.15 * np.maximum(0, 15.5 - temp_sim)
    actual_load = np.maximum(0.2, base_load + rng.normal(0, 0.35, n_hours))
    
    raw_df = pd.DataFrame({
        "load_kw": actual_load,
        "temp_air": temp_sim
    }, index=idx)
    
    # 2. 构建时序特征
    features_df = create_forecasting_features(raw_df, target_col="load_kw")
    
    # 3. 划分训练集 (前 100 天) 与测试预测时刻 t (第 101 天中午 12:00)
    train_cutoff = 100 * 24
    train_features = features_df.iloc[:train_cutoff]
    
    # 4. 训练分位数回归模型
    forecaster = DirectMultiStepQuantileLGBM(
        quantiles=[0.10, 0.50, 0.90],
        horizon_steps=24
    )
    print("Training Direct Quantile LightGBM models across 24 horizons...")
    forecaster.fit(train_features, target_col="load_kw")
    
    # 5. 在测试原点进行未来 24 小时日前预测
    origin_loc = train_cutoff + 12 # 假设在中午 12:00 作出日前预测
    pred_features = features_df.iloc[[origin_loc]]
    forecast_results = forecaster.predict(pred_features)
    
    # 对齐未来 24 步的时间戳
    future_timestamps = features_df.index[origin_loc + 1 : origin_loc + 25]
    forecast_results.index = future_timestamps
    actual_series = raw_df.loc[future_timestamps, "load_kw"]
    
    print("\n--- 24-Step Probabilistic Forecast Results (kW) ---")
    print(forecast_results.head(6))
    
    # 6. 计算分位数损失 (Calibration Check)
    for q in [0.10, 0.50, 0.90]:
        col = f"q_{int(q*100)}"
        loss = pinball_loss(actual_series.to_numpy(), forecast_results[col].to_numpy(), alpha=q)
        coverage = np.mean(actual_series.to_numpy() <= forecast_results[col].to_numpy())
        print(f"Quantile {int(q*100)}% | Pinball Loss: {loss:.4f} | Empirical Coverage: {coverage:.1%}")