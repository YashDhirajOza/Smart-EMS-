from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Asset(BaseModel):
    asset_id: str
    asset_type: str
    name: str
    location: Optional[str]
    rated_power_kw: Optional[float]
    health_score: Optional[float]


class AssetListResponse(BaseModel):
    assets: List[Asset]


class TelemetryPoint(BaseModel):
    timestamp: datetime
    metrics: Dict[str, float]


class TelemetrySnapshot(BaseModel):
    asset_id: str
    asset_type: str
    metrics: Dict[str, float]
    timestamp: datetime


class TelemetrySeries(BaseModel):
    asset_id: str
    points: List[TelemetryPoint]


class EnergyFlowNode(BaseModel):
    node_id: str
    label: str
    power_kw: float


class EnergyFlowEdge(BaseModel):
    source: str
    target: str
    power_kw: float


class EnergyFlowState(BaseModel):
    generated_at: datetime
    nodes: List[EnergyFlowNode]
    edges: List[EnergyFlowEdge]


class Alert(BaseModel):
    id: str
    asset_id: str
    metric_name: str
    severity: str
    message: str
    resolver_action: Dict[str, object] = Field(default_factory=dict)
    raised_at: datetime


class AlertList(BaseModel):
    alerts: List[Alert]


class SystemHealth(BaseModel):
    generated_at: datetime
    module_status: Dict[str, str]
    data_freshness_seconds: Dict[str, int]


class ForecastEntry(BaseModel):
    metric: str
    forecast_type: str
    value: str | float | None
    baseline: float | None = None


class AssetForecast(BaseModel):
    asset_id: str
    entries: List[ForecastEntry]


class ForecastBundle(BaseModel):
    generated_at: datetime
    assets: List[AssetForecast]


class RecommendationIn(BaseModel):
    source: str = Field(default="rl_service")
    asset_scope: Dict[str, object]
    action: Dict[str, object]
    confidence: float | None = None


class Recommendation(BaseModel):
    id: str
    source: str
    asset_scope: Dict[str, object]
    action: Dict[str, object]
    confidence: float | None
    received_at: datetime


class RecommendationAck(BaseModel):
    status: str = "accepted"
    recommendation_id: str
