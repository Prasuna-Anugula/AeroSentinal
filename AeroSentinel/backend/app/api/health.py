from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.ml.model_loader import ModelRegistry
from app.services.data_service import get_available_subsets
from app.schemas.schemas import HealthResponse, SystemStatusResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status(db: Session = Depends(get_db)):
    reg = ModelRegistry.get()
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return SystemStatusResponse(
        backend="AeroSentinel API v1.0",
        models_ready=reg.is_ready(),
        datasets_available=get_available_subsets(),
        model_status=reg.status(),
        db_connected=db_ok,
    )
