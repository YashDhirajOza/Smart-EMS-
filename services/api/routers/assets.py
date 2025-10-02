from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dependencies import get_current_subject, get_db
from schemas import AssetListResponse, TelemetrySeries, TelemetrySnapshot
from services.influx_service import fetch_latest_metrics, fetch_timeseries
from services.postgres_service import fetch_assets_with_health

router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("", response_model=AssetListResponse)
def list_assets(
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_subject),
):
    assets = fetch_assets_with_health(db)
    return {"assets": assets}


@router.get("/{asset_id}/realtime", response_model=TelemetrySnapshot)
def get_realtime(
    asset_id: str,
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_subject),
):
    asset = next((a for a in fetch_assets_with_health(db) if a["asset_id"] == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    metrics = fetch_latest_metrics(asset_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="No telemetry available")
    return {
        "asset_id": asset_id,
        "asset_type": asset["asset_type"],
        "metrics": metrics,
        "timestamp": datetime.utcnow(),
    }


@router.get("/{asset_id}/history", response_model=TelemetrySeries)
def get_history(
    asset_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    window_minutes: int = 60,
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_subject),
):
    if not start:
        start = datetime.utcnow() - timedelta(minutes=window_minutes)
    asset = next((a for a in fetch_assets_with_health(db) if a["asset_id"] == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    points = fetch_timeseries(asset_id, start=start, end=end)
    return {"asset_id": asset_id, "points": points}
