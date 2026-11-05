"""
AeroSentinel — Maintenance Service
Deterministic rule-based recommendation engine.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.orm import MaintenanceRecommendation, Engine as EngineORM

logger = logging.getLogger(__name__)


def _get_or_create_engine(db: Session, dataset: str, engine_unit: int) -> EngineORM:
    obj = db.query(EngineORM).filter_by(dataset_subset=dataset, engine_unit=engine_unit).first()
    if obj is None:
        obj = EngineORM(engine_unit=engine_unit, dataset_subset=dataset)
        db.add(obj)
        db.flush()
    return obj


def _build_recommendation(
    predicted_rul: float,
    failure_prob: Optional[float],
    anomaly_score: Optional[float],
    anomaly_severity: Optional[str],
) -> Dict:
    """Apply deterministic rules to produce a priority + recommendation text."""

    evidence = {
        "predicted_rul":    round(predicted_rul, 2),
        "failure_prob":     round(failure_prob, 4) if failure_prob is not None else None,
        "anomaly_score":    round(anomaly_score, 4) if anomaly_score is not None else None,
        "anomaly_severity": anomaly_severity,
    }

    if predicted_rul < 15:
        return {
            "priority":       "critical",
            "recommendation": (
                "IMMEDIATE ENGINEERING REVIEW REQUIRED. Engine predicted RUL is below 15 cycles. "
                "Ground all operations pending inspection. Review sensor telemetry for confirmed degradation."
            ),
            "evidence": evidence,
        }

    if predicted_rul < 30:
        return {
            "priority":       "high",
            "recommendation": (
                "PRIORITY INSPECTION SCHEDULED. Predicted RUL is below 30 cycles. "
                "Schedule maintenance inspection within the next operational window. "
                "Verify failure-risk indicators and anomalous sensors."
            ),
            "evidence": evidence,
        }

    if failure_prob is not None and failure_prob > 0.7:
        return {
            "priority":       "high",
            "recommendation": (
                "HIGH FAILURE RISK DETECTED. Failure probability exceeds 0.70. "
                "Review failure-risk sensor indicators and cross-validate with RUL estimate. "
                "Schedule precautionary inspection."
            ),
            "evidence": evidence,
        }

    if anomaly_severity in ("high_anomaly",):
        return {
            "priority":       "high",
            "recommendation": (
                "ABNORMAL SENSOR BEHAVIOR DETECTED. Multiple sensor readings are outside normal operating range. "
                "Inspect sensor array and review degradation trend. Consider early maintenance."
            ),
            "evidence": evidence,
        }

    if anomaly_severity == "unusual":
        return {
            "priority":       "medium",
            "recommendation": (
                "UNUSUAL SENSOR PATTERNS OBSERVED. Some sensors show deviation from baseline. "
                "Increase monitoring frequency. Schedule inspection at next planned maintenance window."
            ),
            "evidence": evidence,
        }

    return {
        "priority":       "normal",
        "recommendation": (
            "ROUTINE MONITORING. No critical indicators detected. "
            "Continue standard monitoring schedule. Re-evaluate at next cycle checkpoint."
        ),
        "evidence": evidence,
    }


def generate_recommendation(db: Session, dataset: str, engine_unit: int) -> Dict:
    """Generate and persist a maintenance recommendation for an engine."""
    from app.ml.rul_predictor     import predict_single_engine
    from app.ml.failure_predictor import predict_failure_single
    from app.ml.anomaly_detector  import detect_anomalies_engine

    rul_result  = predict_single_engine(dataset, engine_unit)
    fail_result = predict_failure_single(dataset, engine_unit)

    try:
        anomaly_result = detect_anomalies_engine(dataset, engine_unit)
        anomaly_score  = anomaly_result["overall_score"]
        anomaly_severity = anomaly_result["severity"]
    except Exception:
        anomaly_score    = None
        anomaly_severity = None

    rec = _build_recommendation(
        predicted_rul    = rul_result["predicted_rul"],
        failure_prob     = fail_result["failure_probability"],
        anomaly_score    = anomaly_score,
        anomaly_severity = anomaly_severity,
    )

    eng = _get_or_create_engine(db, dataset, engine_unit)
    eng.last_cycle = rul_result["last_cycle"]

    maint = MaintenanceRecommendation(
        engine_id      = eng.id,
        recommendation = rec["recommendation"],
        priority       = rec["priority"],
        evidence       = json.dumps(rec["evidence"]),
        status         = "pending",
    )
    db.add(maint)
    db.commit()
    db.refresh(maint)

    return {
        "id":             maint.id,
        "engine_unit":    engine_unit,
        "dataset_subset": dataset,
        "recommendation": maint.recommendation,
        "priority":       maint.priority,
        "evidence":       rec["evidence"],
        "status":         maint.status,
        "created_at":     maint.created_at.isoformat(),
    }


def get_recommendations(db: Session, limit: int = 100) -> List[Dict]:
    rows = (
        db.query(MaintenanceRecommendation, EngineORM)
        .join(EngineORM, MaintenanceRecommendation.engine_id == EngineORM.id)
        .order_by(MaintenanceRecommendation.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":             m.id,
            "engine_unit":    eng.engine_unit,
            "dataset_subset": eng.dataset_subset,
            "recommendation": m.recommendation,
            "priority":       m.priority,
            "evidence":       m.get_evidence(),
            "status":         m.status,
            "created_at":     m.created_at.isoformat(),
            "updated_at":     m.updated_at.isoformat() if m.updated_at else None,
        }
        for m, eng in rows
    ]


def update_recommendation_status(db: Session, rec_id: int, status: str) -> Optional[Dict]:
    valid = {"pending", "reviewed", "resolved"}
    if status not in valid:
        raise ValueError(f"Status must be one of {valid}")

    maint = db.query(MaintenanceRecommendation).filter_by(id=rec_id).first()
    if maint is None:
        return None
    maint.status     = status
    maint.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(maint)

    eng = db.query(EngineORM).filter_by(id=maint.engine_id).first()
    return {
        "id":       maint.id,
        "status":   maint.status,
        "engine_unit": eng.engine_unit if eng else None,
        "updated_at":  maint.updated_at.isoformat(),
    }
