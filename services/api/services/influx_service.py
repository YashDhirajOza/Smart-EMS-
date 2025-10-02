from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import os

import pandas as pd

from dependencies import get_influx_client

INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "urjanet_telemetry")
ORGANISATION = os.getenv("INFLUXDB_ORG", "urjanet")


def _query_pivot(asset_filter: str, start: str, stop: str | None = None) -> Optional[pd.DataFrame]:
    stop_clause = f", stop: {stop}" if stop else ""
    flux = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start}{stop_clause})
      |> filter(fn: (r) => r["_measurement"] == "asset_metrics")
      |> filter(fn: (r) => {asset_filter})
      |> keep(columns: ["_time", "_field", "_value", "asset_id", "asset_type"])
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    return _query_data_frame(flux)


def _query_data_frame(flux: str) -> Optional[pd.DataFrame]:
    client = get_influx_client()
    tables = client.query_api().query_data_frame(org=ORGANISATION, query=flux)
    if not tables:
        return None
    frame = tables[0]
    if frame.empty:
        return None
    frame = frame.drop(columns=[col for col in ["result", "table"] if col in frame.columns])
    frame = frame.set_index("_time")
    frame.index = pd.to_datetime(frame.index)
    return frame


def fetch_latest_metrics(asset_id: str) -> Optional[Dict[str, float]]:
    frame = _query_pivot(asset_filter=f'r["asset_id"] == "{asset_id}"', start="-10m")
    if frame is None or frame.empty:
        return None
    latest = frame.iloc[-1]
    return {
        column: float(latest[column])
        for column in frame.columns
        if column not in {"asset_id", "asset_type"} and pd.notna(latest[column])
    }


def fetch_timeseries(asset_id: str, start: datetime, end: datetime | None = None) -> List[Dict[str, object]]:
    stop = f"{end.isoformat()}" if end else None
    frame = _query_pivot(
        asset_filter=f'r["asset_id"] == "{asset_id}"',
        start=f"{start.isoformat()}",
        stop=stop,
    )
    if frame is None:
        return []
    return [
        {
            "timestamp": index.to_pydatetime(),
            "metrics": {
                col: float(row[col])
                for col in frame.columns
                if col not in {"asset_id", "asset_type"} and pd.notna(row[col])
            },
        }
        for index, row in frame.iterrows()
    ]


def build_energy_flow_snapshot() -> Dict[str, object]:
    now = datetime.utcnow()
    frame = _query_pivot(
        asset_filter="true",
        start=f"{(now - timedelta(minutes=5)).isoformat()}Z",
    )
    nodes: List[Dict[str, object]] = []
    edges: List[Dict[str, object]] = []
    if frame is not None and not frame.empty:
        if "asset_id" in frame.columns:
            grouped = frame.groupby(frame["asset_id"])
            for asset_id, data in grouped:
                latest = data.iloc[-1]
                power = float(latest.get("power_kw", 0.0))
                nodes.append({
                    "node_id": asset_id,
                    "label": asset_id.replace("_", " ").title(),
                    "power_kw": power,
                })
                if power >= 0:
                    edges.append({
                        "source": asset_id,
                        "target": "load" if latest.get("power_kw", 0) >= 0 else "grid",
                        "power_kw": abs(power),
                    })

    nodes.append({"node_id": "grid", "label": "Grid", "power_kw": 0.0})
    nodes.append({"node_id": "load", "label": "Load", "power_kw": 0.0})

    return {
        "generated_at": now,
        "nodes": nodes,
        "edges": edges,
    }
