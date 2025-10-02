from __future__ import annotations

from fastapi import APIRouter, Depends

from dependencies import get_current_subject
from schemas import EnergyFlowState
from services.cache_service import energy_flow_cache
from services.influx_service import build_energy_flow_snapshot

router = APIRouter(prefix="/api/energy-flow", tags=["energy-flow"])


@router.get("/live", response_model=EnergyFlowState)
def get_energy_flow(_: dict = Depends(get_current_subject)):
    cached = energy_flow_cache.get("live")
    if cached:
        return cached
    snapshot = build_energy_flow_snapshot()
    energy_flow_cache["live"] = snapshot
    return snapshot
