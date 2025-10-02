from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from dependencies import get_current_subject, get_db
from schemas import ForecastBundle
from services.influx_service import fetch_timeseries
from services.postgres_service import fetch_assets_with_health

router = APIRouter(prefix="/api/digital-twin", tags=["digital_twin"])


def _linear_projection(points: List[Dict[str, object]], metric: str) -> Dict[str, object] | None:
    if len(points) < 3:
        return None
    times = np.array([(p["timestamp"] - points[0]["timestamp"]).total_seconds() for p in points])
    values = np.array([p["metrics"].get(metric) for p in points], dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() < 3:
        return None
    times = times[mask]
    values = values[mask]
    slope, intercept = np.polyfit(times, values, 1)
    if slope >= 0:
        return None
    seconds_to_threshold = (0.2 - intercept) / slope
    if seconds_to_threshold <= 0:
        return None
    forecast_time = points[0]["timestamp"] + timedelta(seconds=float(seconds_to_threshold))
    return {
        "metric": metric,
        "forecast_type": "time_to_empty",
        "value": forecast_time.isoformat(),
        "baseline": float(values[-1]),
    }


@router.get("/forecasts", response_model=ForecastBundle)
def get_forecasts(
    db: Session = Depends(get_db),
    _: Dict = Depends(get_current_subject),
):
    assets = fetch_assets_with_health(db)
    forecasts = []
    for asset in assets:
        points = fetch_timeseries(
            asset_id=asset["asset_id"],
            start=datetime.utcnow() - timedelta(minutes=30),
        )
        entries = []
        if asset["asset_type"] == "battery":
            projection = _linear_projection(points, "state_of_charge")
            if projection:
                entries.append(projection)
        if entries:
            forecasts.append({"asset_id": asset["asset_id"], "entries": entries})
    return {"generated_at": datetime.utcnow(), "assets": forecasts}
