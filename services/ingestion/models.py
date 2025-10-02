from __future__ import annotations

from datetime import datetime
from typing import Dict

from pydantic import BaseModel, Field, validator


class TelemetryPayload(BaseModel):
    timestamp: datetime
    asset_id: str = Field(..., regex=r"^[a-z0-9_]+$")
    asset_type: str
    metrics: Dict[str, float]

    @validator("metrics")
    def ensure_metrics(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("metrics must not be empty")
        return value
