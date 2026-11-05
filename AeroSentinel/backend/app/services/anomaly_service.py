"""
AeroSentinel — Anomaly Service
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Dict, List

from sqlalchemy.orm import Session

from app.ml.anomaly_detector import detect_anomalies_engine, detect_anomalies_all
from app.models.orm import Anomaly, Engine as EngineORM

logger = logging.getLogger(__name__)


def _get_or_create_engine(db: Session, dataset: str, engine_unit: int) -> EngineORM:
    obj = db.query(EngineORM).filter_by(dataset_subset=dataset, engine_unit=engine_unit).first()
    if obj is None:
        obj = EngineORM(engine_unit=engine_unit, dataset_subset=dataset)
        db.add(obj)
        db.flush()
    return obj


def run_anomaly_detection(db: Session, dataset: str, engine_unit: int) -> Dict:
    result = detect_anomalies_engine(dataset, engine_unit)

    eng = _get_or_create_engine(db, dataset, engine_unit)
    cycles_data = result.get("cycles", [])
    last = cycles_data[-1] if cycles_data else {}

    anomaly_rec = Anomaly(
        engine_id            = eng.id,
        cycle                = last.get("cycle"),
        anomaly_score        = result["overall_score"],
        severity             = result["severity"],
        contributing_sensors = json.dumps(last.get("top_sensors", [])),
    )
    db.add(anomaly_rec)
    db.commit()

    return result


def get_anomaly_summary(dataset: str) -> List[Dict]:
    return detect_anomalies_all(dataset)


def get_anomaly_history(db: Session, limit: int = 50) -> List[Dict]:
    rows = (
        db.query(Anomaly, EngineORM)
        .join(EngineORM, Anomaly.engine_id == EngineORM.id)
        .order_by(Anomaly.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":           a.id,
            "engine_unit":  eng.engine_unit,
            "dataset":      eng.dataset_subset,
            "cycle":        a.cycle,
            "score":        a.anomaly_score,
            "severity":     a.severity,
            "top_sensors":  a.get_contributing_sensors(),
            "created_at":   a.created_at.isoformat(),
        }
        for a, eng in rows
    ]
