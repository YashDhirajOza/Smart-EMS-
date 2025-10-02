from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class AssetProfile:
    asset_id: str
    asset_type: str
    dataset_path: Path
    base_topic: str
    metric_mappings: Dict[str, str]
    publish_interval_seconds: range


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

ASSET_PROFILES: List[AssetProfile] = [
    AssetProfile(
        asset_id="solar_panel_01",
        asset_type="solar",
        dataset_path=DATA_DIR / "solar_generation.csv",
        base_topic="urjanet/telemetry/solar",
        metric_mappings={
            "voltage": "voltage",
            "current": "current",
            "power_kw": "power_kw",
            "temperature": "temperature",
        },
        publish_interval_seconds=range(2, 5),
    ),
    AssetProfile(
        asset_id="solar_panel_02",
        asset_type="solar",
        dataset_path=DATA_DIR / "solar_generation.csv",
        base_topic="urjanet/telemetry/solar",
        metric_mappings={
            "voltage": "voltage",
            "current": "current",
            "power_kw": "power_kw",
            "temperature": "temperature",
        },
        publish_interval_seconds=range(2, 6),
    ),
    AssetProfile(
        asset_id="battery_bank_01",
        asset_type="battery",
        dataset_path=DATA_DIR / "battery_profile.csv",
        base_topic="urjanet/telemetry/battery",
        metric_mappings={
            "voltage": "voltage",
            "current": "current",
            "state_of_charge": "state_of_charge",
            "temperature": "temperature",
        },
        publish_interval_seconds=range(3, 6),
    ),
    AssetProfile(
        asset_id="ev_charger_01",
        asset_type="ev_charger",
        dataset_path=DATA_DIR / "ev_charger_sessions.csv",
        base_topic="urjanet/telemetry/ev",
        metric_mappings={
            "voltage": "voltage",
            "current": "current",
            "power_kw": "power_kw",
        },
        publish_interval_seconds=range(2, 5),
    ),
    AssetProfile(
        asset_id="ev_charger_02",
        asset_type="ev_charger",
        dataset_path=DATA_DIR / "ev_charger_sessions.csv",
        base_topic="urjanet/telemetry/ev",
        metric_mappings={
            "voltage": "voltage",
            "current": "current",
            "power_kw": "power_kw",
        },
        publish_interval_seconds=range(2, 5),
    ),
]
