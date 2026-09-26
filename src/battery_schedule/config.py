from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float; charge_power_kw: float; discharge_power_kw: float
    eta_charge: float; eta_discharge: float; initial_soc_kwh: float; terminal_soc_kwh: float
    degradation_eur_per_kwh: float
    export_price_ratio: float = 0.75
    cvar_alpha: float = 0.90
    cvar_weight: float = 0.20
    max_cycles_per_day: float | None = None  # delivered energy per day / capacity; None = unlimited

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["optimization"] = BatteryConfig(**config["optimization"])
    return config
