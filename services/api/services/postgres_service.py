from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session


def fetch_assets_with_health(db: Session) -> List[Dict[str, object]]:
    query = text(
        """
        SELECT a.asset_id, a.asset_type, a.name, a.location, a.rated_power_kw, h.health_score
        FROM assets a
        LEFT JOIN asset_health h ON a.asset_id = h.asset_id
        ORDER BY a.asset_id
        """
    )
    result = db.execute(query)
    return [dict(row._mapping) for row in result]


def fetch_active_alerts(db: Session) -> List[Dict[str, object]]:
    query = text(
        "SELECT id, asset_id, metric_name, severity, message, resolver_action, raised_at FROM alerts WHERE resolved_at IS NULL ORDER BY raised_at DESC"
    )
    return [dict(row._mapping) for row in db.execute(query)]


def insert_recommendation(db: Session, payload: Dict[str, object]) -> str:
    recommendation_id = payload.get("id") or str(uuid4())
    query = text(
        """
        INSERT INTO recommendations (id, source, asset_scope, action, confidence, received_at)
        VALUES (:id, :source, :asset_scope::jsonb, :action::jsonb, :confidence, NOW())
        RETURNING id
        """
    )
    db.execute(
        query,
        {
            "id": recommendation_id,
            "source": payload.get("source", "rl_service"),
            "asset_scope": payload.get("asset_scope"),
            "action": payload.get("action"),
            "confidence": payload.get("confidence"),
        },
    )
    return recommendation_id


def fetch_latest_recommendation(db: Session) -> Optional[Dict[str, object]]:
    query = text(
        """
        SELECT id, source, asset_scope, action, confidence, received_at
        FROM recommendations
        ORDER BY received_at DESC
        LIMIT 1
        """
    )
    row = db.execute(query).mappings().first()
    return dict(row) if row else None


def fetch_system_health(db: Session) -> Dict[str, object]:
    assets = db.execute(text("SELECT asset_id, updated_at FROM asset_health"))
    freshness = {row.asset_id: (datetime.utcnow() - row.updated_at).seconds for row in assets}
    return {
        "module_status": {
            "ingestion": "ok",
            "diagnostics": "ok",
            "digital_twin": "ok",
        },
        "data_freshness_seconds": freshness,
    }
