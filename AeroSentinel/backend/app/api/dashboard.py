from fastapi import APIRouter, Query
from app.services.prediction_service import get_dashboard_summary, get_fleet_overview
from app.ml.evaluation import get_fd001_metrics

router = APIRouter(prefix="/dashboard")


@router.get("/summary")
def dashboard_summary(dataset: str = Query("FD001")):
    return get_dashboard_summary(dataset)


@router.get("/fleet")
def fleet_overview(dataset: str = Query("FD001")):
    return get_fleet_overview(dataset)


@router.get("/metrics/fd001")
def fd001_metrics():
    """Return the verified FD001 training-run evaluation metrics."""
    m = get_fd001_metrics()
    return {
        "source_label": "FD001 — EXISTING TRAINING RUN",
        "rul_regression": {
            "random_forest":      {k: v for k, v in m.get("rul_rf", {}).items()},
            "gradient_boosting":  {k: v for k, v in m.get("rul_gb", {}).items()},
            "preferred_model":    "Random Forest",
        },
        "failure_classification": {
            "random_forest":      {k: v for k, v in m.get("fail_rf", {}).items()},
            "gradient_boosting":  {k: v for k, v in m.get("fail_gb", {}).items()},
            "preferred_model":    "Gradient Boosting",
        },
        "production_monitoring":  m.get("production_monitoring", {}),
        "dataset_info": {
            "train_samples": m.get("train_samples"),
            "test_samples":  m.get("test_samples"),
            "feature_count": m.get("feature_count"),
        },
    }
