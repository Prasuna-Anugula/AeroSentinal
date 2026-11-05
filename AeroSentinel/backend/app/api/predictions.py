from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import (
    PredictRULRequest, PredictFailureRequest,
)
from app.services.prediction_service import (
    run_rul_prediction, run_failure_prediction, get_prediction_history
)

router = APIRouter(prefix="/predict")


@router.post("/rul")
def predict_rul(req: PredictRULRequest, db: Session = Depends(get_db)):
    try:
        result = run_rul_prediction(db, req.dataset, req.engine_unit, req.model_key)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@router.post("/failure")
def predict_failure(req: PredictFailureRequest, db: Session = Depends(get_db)):
    try:
        result = run_failure_prediction(db, req.dataset, req.engine_unit, req.model_key)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@router.get("/history")
def prediction_history(limit: int = 50, db: Session = Depends(get_db)):
    return get_prediction_history(db, limit)
