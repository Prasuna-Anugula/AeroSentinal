from fastapi import APIRouter, HTTPException, Query
from app.services.data_service import (
    get_dataset_summary, get_available_subsets,
    get_engine_units, get_engine_sensor_history,
    get_engine_sensor_snapshot, get_lifecycle_stats,
    get_rul_distribution,
)
from app.ml.evaluation import get_sensor_correlations
from app.services.data_service import get_train_df

router = APIRouter(prefix="/datasets")

VALID = {"FD001", "FD002", "FD003", "FD004"}


def _validate(subset: str):
    if subset not in VALID:
        raise HTTPException(400, f"subset must be one of {VALID}")


@router.get("")
def list_datasets():
    available = get_available_subsets()
    return [get_dataset_summary(s) for s in available]


@router.get("/{subset}/summary")
def dataset_summary(subset: str):
    _validate(subset)
    return get_dataset_summary(subset)


@router.get("/{subset}/engines")
def list_engines(subset: str, source: str = Query("test")):
    _validate(subset)
    units = get_engine_units(subset, source)
    return {"subset": subset, "source": source, "engines": units, "count": len(units)}


@router.get("/{subset}/engines/{engine_id}/sensors")
def engine_sensors(subset: str, engine_id: int):
    _validate(subset)
    history = get_engine_sensor_history(subset, engine_id)
    if not history:
        raise HTTPException(404, f"Engine {engine_id} not found in {subset}")
    return {"engine_unit": engine_id, "dataset": subset, "sensors": history}


@router.get("/{subset}/engines/{engine_id}/snapshot")
def engine_snapshot(subset: str, engine_id: int):
    _validate(subset)
    snap = get_engine_sensor_snapshot(subset, engine_id)
    if not snap:
        raise HTTPException(404, f"Engine {engine_id} not found in {subset}")
    return {"engine_unit": engine_id, "dataset": subset, "sensors": snap}


@router.get("/{subset}/lifecycle")
def lifecycle_stats(subset: str):
    _validate(subset)
    return get_lifecycle_stats(subset)


@router.get("/{subset}/rul-distribution")
def rul_dist(subset: str):
    _validate(subset)
    return get_rul_distribution(subset)


@router.get("/{subset}/correlations")
def sensor_corr(subset: str):
    _validate(subset)
    df = get_train_df(subset)
    return get_sensor_correlations(df)
