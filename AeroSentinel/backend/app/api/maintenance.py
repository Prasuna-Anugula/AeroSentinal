from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import MaintenanceRecommendationCreate, MaintenanceStatusUpdate
from app.services.maintenance_service import (
    generate_recommendation, get_recommendations, update_recommendation_status
)

router = APIRouter(prefix="/maintenance")


@router.post("/recommendations")
def create_recommendation(req: MaintenanceRecommendationCreate, db: Session = Depends(get_db)):
    try:
        return generate_recommendation(db, req.dataset, req.engine_unit)
    except Exception as e:
        raise HTTPException(500, f"Recommendation generation failed: {e}")


@router.get("/history")
def maintenance_history(limit: int = 100, db: Session = Depends(get_db)):
    return get_recommendations(db, limit)


@router.patch("/{rec_id}/status")
def update_status(rec_id: int, body: MaintenanceStatusUpdate, db: Session = Depends(get_db)):
    try:
        result = update_recommendation_status(db, rec_id, body.status)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result is None:
        raise HTTPException(404, f"Recommendation {rec_id} not found")
    return result
