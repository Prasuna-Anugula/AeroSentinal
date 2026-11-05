from fastapi import APIRouter, HTTPException, Query
from app.services.explanation_service import (
    get_feature_importance, get_sensor_correlation_matrix, get_engine_explanation
)

router = APIRouter(prefix="/explanations")


@router.get("/feature-importance")
def feature_importance(model_key: str = Query("rul_rf"), top_n: int = Query(30)):
    return get_feature_importance(model_key, top_n)


@router.get("/sensor-correlations")
def sensor_correlations(subset: str = Query("FD001")):
    return get_sensor_correlation_matrix(subset)


@router.get("/engine")
def engine_explanation(
    dataset:    str = Query("FD001"),
    engine_unit: int = Query(...),
    model_key:  str = Query("rul_rf"),
):
    try:
        return get_engine_explanation(dataset, engine_unit, model_key)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
