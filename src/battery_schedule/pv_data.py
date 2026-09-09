import pandas as pd
import requests
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain

def generate_realistic_pv_profile(
    latitude: float = 43.6047,   # 例如：图卢兹 (Toulouse, France)
    longitude: float = 1.4442,
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-01",
    system_capacity_kw: float = 6.0  # 6 kWp 户用/小型商业屋顶光伏
) -> pd.DataFrame:
    """
    1. 从 Open-Meteo 获取真实历史地面辐照与温度
    2. 使用 pvlib 物理光学/热学模型计算实际交流输出功率 (kW)
    """
    # 1. 获取小时级气象数据
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["direct_normal_irradiance", "diffuse_radiation", "temperature_2m", "wind_speed_10m"],
        "timezone": "UTC"
    }
    r = requests.get(url, params=params).json()
    hourly = r["hourly"]
    
    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "dni": hourly["direct_normal_irradiance"],
        "dhi": hourly["diffuse_radiation"],
        "temp_air": hourly["temperature_2m"],
        "wind_speed": hourly["wind_speed_10m"]
    }).set_index("timestamp")
    
    # 2. 估算水平面总辐射 GHI = DNI * cos(zenith) + DHI
    location = Location(latitude, longitude, tz="UTC")
    solar_position = location.get_solarposition(weather.index)
    weather["ghi"] = weather["dni"] * pvlib.tools.cosd(solar_position["zenith"]) + weather["dhi"]
    weather["ghi"] = weather["ghi"].clip(lower=0)

    # 3. 构建标准光伏组件与逆变器物理模型 (以 6kWp 南向 30度倾角 为例)
    module_params = {"pdc0": system_capacity_kw, "gamma_pdc": -0.004} # 功率温度系数
    inverter_params = {"pdc0": system_capacity_kw, "eta_inv_nom": 0.96}  # 96% 逆变效率
    system = PVSystem(
        surface_tilt=30,
        surface_azimuth=180, # 正南
        module_parameters=module_params,
        inverter_parameters=inverter_params,
        temperature_model_parameters={"a": -3.56, "b": -0.075, "deltaT": 3}
    )
    
    # 4. 运行 pvlib 物理转换链路
    mc = ModelChain(system, location, dc_model="pvwatts", ac_model="pvwatts", spectral_model="no_loss", aoi_model="physical")
    mc.run_model(weather)
    
    pv_df = pd.DataFrame({
        "timestamp": weather.index,
        "pv_kw": (mc.results.ac.fillna(0) / system_capacity_kw * system_capacity_kw).clip(lower=0)
    }).reset_index(drop=True)
    
    return pv_df

if __name__ == "__main__":
    pv_data = generate_realistic_pv_profile()
    print("Realistic Physical PV Output (kW):")
    print(pv_data.tail())
    print(pv_data[300:400])


# plan b: get PV data directly from OPSD dataset
# import pandas as pd
# from pathlib import Path

# def download_and_clean_opsd_household() -> pd.DataFrame:
#     """
#     自动拉取 OPSD 欧洲真实户用小时级实测数据 (包含真实 Load 与真实 PV)
#     """
#     url = "https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv"
#     print("Downloading OPSD real household dataset...")
    
#     # 读取原始数据
#     df = pd.read_csv(url, parse_dates=["utc_timestamp"])
    
#     # 选取其中一户家庭 (例如 Household 1: DE_KN_residential1)
#     # 包含总用电负荷 grid_import/load 与屋顶光伏 pv
#     cols = {
#         "utc_timestamp": "timestamp",
#         "DE_KN_residential1_grid_import": "load_kw",
#         "DE_KN_residential1_pv": "pv_kw"
#     }
    
#     sub_df = df[list(cols.keys())].rename(columns=cols)
#     sub_df["timestamp"] = pd.to_datetime(sub_df["timestamp"], utc=True)
#     sub_df = sub_df.sort_values("timestamp").dropna().reset_index(drop=True)
    
#     return sub_df

# if __name__ == "__main__":
#     df_household = download_and_clean_opsd_household()
#     print("Real Household Load & PV data:")
#     print(df_household.head())