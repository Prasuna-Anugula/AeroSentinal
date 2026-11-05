from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import AnomalyDetectRequest
from app.services.anomaly_service import (
    run_anomaly_detection, get_anomaly_summary, get_anomaly_history
)

router = APIRouter(prefix="/anomaly")


@router.post("/detect")
def detect_anomaly(req: AnomalyDetectRequest, db: Session = Depends(get_db)):
    try:
        return run_anomaly_detection(db, req.dataset, req.engine_unit)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Anomaly detection failed: {e}")


@router.get("/summary")
def anomaly_summary(dataset: str = Query("FD001")):
    try:
        return get_anomaly_summary(dataset)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/history")
def anomaly_history(limit: int = 50, db: Session = Depends(get_db)):
    return get_anomaly_history(db, limit)
