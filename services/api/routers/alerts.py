from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from dependencies import get_current_subject, get_db
from schemas import AlertList
from services.postgres_service import fetch_active_alerts

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=AlertList)
def list_alerts(
    db: Session = Depends(get_db),
    _: dict = Depends(get_current_subject),
):
    alerts = fetch_active_alerts(db)
    for alert in alerts:
        resolver_action = alert.get("resolver_action")
        if isinstance(resolver_action, str):
            alert["resolver_action"] = json.loads(resolver_action)
    return {"alerts": alerts}
