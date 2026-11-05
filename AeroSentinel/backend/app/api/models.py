from fastapi import APIRouter
from app.ml.model_loader import ModelRegistry
from app.ml.evaluation   import get_fd001_metrics

router = APIRouter(prefix="/models")

MODEL_CATALOG = [
    {"model_key": "rul_rf",  "display_name": "Random Forest — RUL",               "model_type": "regression",     "dataset": "FD001"},
    {"model_key": "rul_gb",  "display_name": "Gradient Boosting — RUL",            "model_type": "regression",     "dataset": "FD001"},
    {"model_key": "fail_rf", "display_name": "Random Forest — Failure Classifier", "model_type": "classification", "dataset": "FD001"},
    {"model_key": "fail_gb", "display_name": "Gradient Boosting — Failure Classifier", "model_type": "classification", "dataset": "FD001"},
]


@router.get("")
def list_models():
    reg     = ModelRegistry.get()
    metrics = get_fd001_metrics()
    result  = []
    for m in MODEL_CATALOG:
        key  = m["model_key"]
        met  = metrics.get(key, {})
        result.append({
            **m,
            "is_available": getattr(reg, key) is not None,
            "metrics":      {k: v for k, v in met.items()},
        })
    return result


@router.get("/status")
def model_status():
    reg = ModelRegistry.get()
    return reg.status()


@router.get("/metrics")
def model_metrics():
    return get_fd001_metrics()


@router.get("/comparison")
def model_comparison():
    metrics = get_fd001_metrics()
    return {
        "source_label": "FD001 — EXISTING TRAINING RUN",
        "rul": [
            {"model": "Random Forest",     **{k: v for k, v in metrics.get("rul_rf", {}).items()}, "preferred": True},
            {"model": "Gradient Boosting", **{k: v for k, v in metrics.get("rul_gb", {}).items()}, "preferred": False},
        ],
        "classification": [
            {"model": "Random Forest",     **{k: v for k, v in metrics.get("fail_rf", {}).items()}, "preferred": False},
            {"model": "Gradient Boosting", **{k: v for k, v in metrics.get("fail_gb", {}).items()}, "preferred": True},
        ],
    }
