import numpy as np
import pandas as pd
import holidays

def build_calendar_features(df: pd.DataFrame, country: str = "FR") -> pd.DataFrame:
    idx = df.index
    # 1. 周期三角变换
    hour = idx.hour
    day_of_year = idx.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # 2. 假日与工作日特征
    local_holidays = holidays.country_holidays(country)
    df["is_holiday"] = idx.map(lambda d: d.date() in local_holidays).astype(int)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_business_hour"] = ((idx.hour >= 8) & (idx.hour <= 18) & (df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)
    return df

def build_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df["temp_air"]
    # 1. 采暖度时与制冷度时 (U-Shape 线性拆解)
    df["hdh"] = np.maximum(0, 15.5 - temp)
    df["cdh"] = np.maximum(0, temp - 22.0)

    # 2. 热惯性 EWMA (半衰期模拟 6h 与 24h 建筑热滞后)
    df["temp_ewma_6h"] = temp.ewm(span=6).mean()
    df["temp_ewma_24h"] = temp.ewm(span=24).mean()

    # 3. 气温变化率
    df["temp_diff_1h"] = temp.diff(1).fillna(0)
    return df

def build_solar_features(df: pd.DataFrame, clearsky_ghi: pd.Series, zenith: pd.Series) -> pd.DataFrame:
    # 1. 天文几何特征
    df["cos_zenith"] = np.maximum(0, np.cos(np.radians(zenith)))
    df["is_daylight"] = (zenith < 85).astype(int)

    # 2. 晴空指数 (截断处理规避夜间除以 0)
    df["clearsky_index"] = np.where(
        clearsky_ghi > 10,
        df["ghi"] / clearsky_ghi,
        0.0
    ).clip(0, 1.5)
    return df


def build_strict_lags(df: pd.DataFrame, target_col: str = "load_kw") -> pd.DataFrame:
    series = df[target_col]
    # 严格使用周期性滞后 (满足日前预测边界)
    df[f"{target_col}_lag_24h"] = series.shift(24)
    df[f"{target_col}_lag_48h"] = series.shift(48)
    df[f"{target_col}_lag_168h"] = series.shift(168)
    
    # 历史同星期均值与趋势项
    df[f"{target_col}_diff_24_168"] = df[f"{target_col}_lag_24h"] - df[f"{target_col}_lag_168h"]
    return df

def build_lagged_rolling_features(df: pd.DataFrame, target_col: str = "load_kw") -> pd.DataFrame:
    # 基于昨日 24h 前的数据做滚动计算
    shifted = df[target_col].shift(24)
    df[f"{target_col}_roll_mean_24h"] = shifted.rolling(window=24, min_periods=12).mean()
    df[f"{target_col}_roll_max_24h"] = shifted.rolling(window=24, min_periods=12).max()
    df[f"{target_col}_roll_min_24h"] = shifted.rolling(window=24, min_periods=12).min()
    df[f"{target_col}_roll_std_24h"] = shifted.rolling(window=24, min_periods=12).std()
    return df