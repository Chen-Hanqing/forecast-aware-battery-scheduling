import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
from typing import Tuple

class QuantileCopulaScenarioGenerator:
    """
    通过 Gaussian Copula 捕获跨时步与跨变量的联合相关性，
    结合 LightGBM 输出的 (q10, q50, q90) 进行边际反演生成场景矩阵。
    """
    def __init__(self, quantiles: list[float] = [0.10, 0.50, 0.90], random_seed: int = 42):
        self.quantiles = sorted(quantiles)
        self.rng = np.random.default_rng(random_seed)
        self.copula_chol: np.ndarray | None = None
        self.dim: int = 0

    def fit_copula(self, val_residuals_load: np.ndarray, val_residuals_price: np.ndarray):
        """
        利用验证集上的历史残差估计 (Load, Price) 拼接后的 2T 维联合相关性矩阵。
        - val_residuals_*: 形状为 [N_samples, T] 的历史残差矩阵
        """
        # 1. 拼接负荷与电价残差: [N_samples, 2*T]
        joint_residuals = np.hstack([val_residuals_load, val_residuals_price])
        self.dim = joint_residuals.shape[1]
        
        # 2. 将残差转换为经验伪观测值 (Pseudo-observations) -> 均匀分布 U(0, 1)
        # 采用秩变换 (Rank-based PIT: Probability Integral Transform)
        ranks = np.argsort(np.argsort(joint_residuals, axis=0), axis=0) + 1
        u_matrix = ranks / (len(joint_residuals) + 1.0)
        
        # 3. 映射到标准正态空间并计算经验相关系数矩阵
        z_matrix = norm.ppf(u_matrix)
        corr_matrix = np.corrcoef(z_matrix, rowvar=False)
        
        # 4. 保证正定性并进行 Cholesky 分解 (处理数值微小非正定)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2.0
        min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
        if min_eig < 1e-6:
            corr_matrix += (1e-6 - min_eig) * np.eye(self.dim)
            
        self.copula_chol = np.linalg.cholesky(corr_matrix)

    def _sample_copula_uniforms(self, n_scenarios: int) -> np.ndarray:
        """从高斯 Copula 采样 [S, 2*T] 的联合均匀分布样本 U in (0, 1)"""
        z_uncorrelated = self.rng.standard_normal((n_scenarios, self.dim))
        z_correlated = z_uncorrelated @ self.copula_chol.T
        u_samples = norm.cdf(z_correlated)
        # 截断极值规避分位数反演时发生外推溢出
        return np.clip(u_samples, 1e-4, 1.0 - 1e-4)

    def _inverse_quantile_cdf(self, u_vals: np.ndarray, q_preds: np.ndarray) -> np.ndarray:
        """
        针对某一时步 t，利用预测的 (q10, q50, q90) 建立单调分段插值并计算反函数 F^{-1}(u)
        - u_vals: 长度为 S 的均匀分布抽样点
        - q_preds: 形状为 (3,) 对应 [q10, q50, q90]
        """
        q10, q50, q90 = q_preds[0], q_preds[1], q_preds[2]
        
        # 构建锚点：0.01 尾部估计、q10、q50、q90、0.99 尾部估计 (基于 IQR 线性外推)
        iqr = max(1e-3, q90 - q10)
        q01_est = max(0.0, q10 - 0.8 * iqr) # 物理下限非负截断
        q99_est = q90 + 0.8 * iqr
        
        u_anchors = np.array([0.00, 0.10, 0.50, 0.90, 1.00])
        val_anchors = np.array([q01_est, q10, q50, q90, q99_est])
        
        # 使用单调保形分段多项式插值 (Pchip) 避免震荡并保证单调递增
        interpolator = PchipInterpolator(u_anchors, val_anchors)
        sampled_values = interpolator(u_vals)
        return np.maximum(0.0, sampled_values)

    def generate_scenarios(
        self,
        forecast_load: pd.DataFrame,   # 包含列: ['q_10', 'q_50', 'q_90'], 长度 T
        forecast_price: pd.DataFrame,  # 包含列: ['q_10', 'q_50', 'q_90'], 长度 T
        n_scenarios: int = 80
    ) -> np.ndarray:
        """
        生成符合 solve_schedule 要求的 [S, T, 2] 矩阵。
        最后一维: index 0 -> load_kw, index 1 -> import_price_eur_kwh
        """
        T = len(forecast_load)
        if self.copula_chol is None or self.dim != 2 * T:
            raise ValueError(f"Copula 未拟合或维度不匹配 (期望维度 {2*T}, 当前 {self.dim})")

        # 1. 采样联合均匀随机数: [S, 2*T]
        u_samples = self._sample_copula_uniforms(n_scenarios)
        u_load = u_samples[:, :T]       # 前 T 列对应 Load 时序
        u_price = u_samples[:, T:]      # 后 T 列对应 Price 时序

        # 2. 逐时步反演生成连续轨迹
        scenarios = np.zeros((n_scenarios, T, 2), dtype=float)
        
        load_mat = forecast_load[["q_10", "q_50", "q_90"]].to_numpy()
        price_mat = forecast_price[["q_10", "q_50", "q_90"]].to_numpy()

        for t in range(T):
            scenarios[:, t, 0] = self._inverse_quantile_cdf(u_load[:, t], load_mat[t])
            scenarios[:, t, 1] = self._inverse_quantile_cdf(u_price[:, t], price_mat[t])

        return scenarios


# -------------------------------------------------------------------------
# 使用示例：与上一节 LightGBM 预测输出及 solve_schedule 无缝对接
# -------------------------------------------------------------------------
if __name__ == "__main__":
    T_horizon = 24
    N_scenarios = 80

    # 1. 模拟验证集残差 (实际项目中由 Rolling-CV 历史步长真实值 - 预测均值计算得出)
    rng = np.random.default_rng(42)
    val_samples = 150
    # 构造带有跨时步自相关与“电价-负荷”正相关的残差
    base_shock = rng.normal(0, 1, (val_samples, 1))
    res_load = np.cumsum(rng.normal(0, 0.2, (val_samples, T_horizon)), axis=1) + base_shock * 0.3
    res_price = np.cumsum(rng.normal(0, 0.02, (val_samples, T_horizon)), axis=1) + base_shock * 0.03

    # 2. 初始化并拟合 Copula
    copula_gen = QuantileCopulaScenarioGenerator(random_seed=42)
    copula_gen.fit_copula(res_load, res_price)

    # 3. 模拟 LightGBM 输出的 24 步分位数预测表
    hours = np.arange(T_horizon)
    base_l = 2.0 + 1.2 * np.sin(np.pi * (hours - 6) / 12)**2
    forecast_load = pd.DataFrame({
        "q_10": np.maximum(0.2, base_l - 0.5),
        "q_50": base_l,
        "q_90": base_l + 0.7
    })

    base_p = 0.15 + 0.08 * np.sin(np.pi * (hours - 14) / 12)**2
    forecast_price = pd.DataFrame({
        "q_10": np.maximum(0.02, base_p - 0.04),
        "q_50": base_p,
        "q_90": base_p + 0.06
    })

    # 4. 生成多场景三维矩阵 [80, 24, 2]
    scenarios_matrix = copula_gen.generate_scenarios(
        forecast_load=forecast_load,
        forecast_price=forecast_price,
        n_scenarios=N_scenarios
    )

    print("Generated Scenarios Shape:", scenarios_matrix.shape) # 输出 (80, 24, 2)
    print("Scenario 0 Load (kW) Profile sample (first 4h):", np.round(scenarios_matrix[0, :4, 0], 3))
    print("Scenario 0 Price (EUR/kWh) sample (first 4h):", np.round(scenarios_matrix[0, :4, 1], 3))