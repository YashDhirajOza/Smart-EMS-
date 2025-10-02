from __future__ import annotations

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dependencies import get_current_subject, get_db
from schemas import Recommendation, RecommendationAck, RecommendationIn, SystemHealth
from services.postgres_service import (
    fetch_latest_recommendation,
    fetch_system_health,
    insert_recommendation,
)

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=SystemHealth)
def get_system_health(db: Session = Depends(get_db), _: Dict = Depends(get_current_subject)):
    snapshot = fetch_system_health(db)
    return {"generated_at": datetime.utcnow(), **snapshot}


@router.post("/recommendation", response_model=RecommendationAck)
def post_recommendation(
    payload: RecommendationIn,
    db: Session = Depends(get_db),
    _: Dict = Depends(get_current_subject),
):
    recommendation_id = insert_recommendation(db, payload.dict())
    db.commit()
    return {"recommendation_id": recommendation_id}


@router.get("/recommendation", response_model=Recommendation)
def get_recommendation(db: Session = Depends(get_db), _: Dict = Depends(get_current_subject)):
    recommendation = fetch_latest_recommendation(db)
    if recommendation is None:
        raise HTTPException(status_code=404, detail="No recommendation available")
    return recommendation
