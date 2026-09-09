import os
from pathlib import Path
import pandas as pd
from entsoe import EntsoePandasClient

def fetch_and_clean_entsoe_prices(
    api_key: str,
    country_code: str = "FR",
    start_str: str = "2024-01-01",
    end_str: str = "2024-03-01",
    target_freq: str = "1h",
) -> pd.DataFrame:
    """从 ENTSO-E 拉取日前电价，清洗为规则的 UTC 小时级时序数据。"""
    client = EntsoePandasClient(api_key=api_key)

    # 1. 明确时区定义（ENTSO-E 要求带有时区的 Timestamp，欧洲主要为 UTC 或 CET/CEST）
    start = pd.Timestamp(start_str, tz="UTC")
    end = pd.Timestamp(end_str, tz="UTC")

    print(f"Fetching Day-Ahead prices for {country_code} from {start} to {end}...")
    
    # 2. 调用 API 获取日前电价 (EUR/MWh)
    # raw_series 的 index 为带有本地/UTC 时区的 DatetimeIndex
    raw_series: pd.Series = client.query_day_ahead_prices(
        country_code=country_code,
        start=start,
        end=end
    )

    # 3. 规范化为 DataFrame
    df = raw_series.to_frame(name="price_eur_mwh")
    df.index.name = "timestamp"

    # 4. 转换到统一时区（推荐统一为 UTC，规避夏令时 23h/25h 跳变带来的断点和重复）
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # 5. 去除重复时间戳（若存在），按时间升序排序
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # 6. 重采样对齐到目标频率（如 1h），处理部分欧洲市场 15min/30min 结价粒度
    # 使用 mean() 聚合或直接 resample 保证时序连续无空缺
    df_resampled = df.resample(target_freq).mean()

    # 7. 缺失值处理（短时空缺使用前向填充/线性插值）
    if df_resampled["price_eur_mwh"].isna().any():
        n_missing = df_resampled["price_eur_mwh"].isna().sum()
        print(f"Warning: Found {n_missing} missing hours. Applying forward fill.")
        df_resampled["price_eur_mwh"] = df_resampled["price_eur_mwh"].ffill().bfill()

    # 8. 单位换算：将大电网维度的 EUR/MWh 转换为户用/BTM 侧常用的 EUR/kWh
    df_resampled["import_price_eur_kwh"] = df_resampled["price_eur_mwh"] / 1000.0

    # 9. 格式整理：重置索引以符合输入契约 (timestamp, import_price_eur_kwh)
    output_df = df_resampled.reset_index()[["timestamp", "import_price_eur_kwh"]]

    return output_df


if __name__ == "__main__":
    # 从环境变量读取 Token，或直接填入字符串
    API_KEY = os.getenv("ENTSOE_API_KEY", "your-entsoe-api-key-here")

    try:
        # 获取法国（FR）2024 年初 60 天数据
        prices = fetch_and_clean_entsoe_prices(
            api_key=API_KEY,
            country_code="FR",
            start_str="2024-01-01",
            end_str="2024-03-01",
            target_freq="1h",
        )

        # 保存为 CSV
        out_path = Path("data/raw/entsoe_prices_fr.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(out_path, index=False)

        print(f"Successfully saved {len(prices)} rows to {out_path}")
        print(prices.head())

    except Exception as e:
        print(f"Error fetching data: {e}")