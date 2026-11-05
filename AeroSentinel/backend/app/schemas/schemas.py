"""
AeroSentinel — Pydantic schemas (request/response contracts)
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Health / System ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class SystemStatusResponse(BaseModel):
    backend: str
    models_ready: bool
    datasets_available: List[str]
    model_status: Dict[str, Any]
    db_connected: bool


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictRULRequest(BaseModel):
    dataset: str = Field(..., description="FD001 / FD002 / FD003 / FD004")
    engine_unit: int = Field(..., ge=1)
    model_key: str = Field("rul_rf", description="rul_rf | rul_gb")


class PredictRULResponse(BaseModel):
    engine_unit: int
    dataset: str
    last_cycle: int
    predicted_rul: float
    true_rul: Optional[float]
    model: str
    health_status: str
    source: str
    timestamp: str


class PredictFailureRequest(BaseModel):
    dataset: str
    engine_unit: int
    model_key: str = Field("fail_gb", description="fail_rf | fail_gb")


class PredictFailureResponse(BaseModel):
    engine_unit: int
    dataset: str
    last_cycle: int
    failure_probability: float
    failure_predicted: bool
    risk_level: str
    model: str
    timestamp: str


class PredictionHistoryItem(BaseModel):
    id: int
    engine_unit: int
    dataset_subset: str
    predicted_rul: Optional[float]
    failure_probability: Optional[float]
    model_name: Optional[str]
    health_status: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


# ── Anomaly ───────────────────────────────────────────────────────────────────

class AnomalyDetectRequest(BaseModel):
    dataset: str
    engine_unit: int


class CycleAnomalyResult(BaseModel):
    cycle: int
    anomaly_score: float
    is_anomaly: bool
    severity: str
    top_sensors: List[str]


class AnomalyDetectResponse(BaseModel):
    engine_unit: int
    dataset: str
    overall_score: float
    severity: str
    anomaly_count: int
    total_cycles: int
    cycles: List[CycleAnomalyResult]


class AnomalySummaryItem(BaseModel):
    engine_unit: int
    last_cycle: int
    anomaly_score: float
    severity: str
    anomaly_count: int


# ── Explanations ──────────────────────────────────────────────────────────────

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    model_key: str
    items: List[FeatureImportanceItem]


class SensorCorrelationItem(BaseModel):
    sensor_a: str
    sensor_b: str
    correlation: float


# ── Fleet / Engine ────────────────────────────────────────────────────────────

class EngineFleetItem(BaseModel):
    engine_unit: int
    last_cycle: int
    predicted_rul: Optional[float]
    true_rul: Optional[float]
    health_status: str
    failure_probability: Optional[float]
    risk_level: Optional[str]
    anomaly_severity: Optional[str]


class EngineSensorSnapshot(BaseModel):
    sensor: str
    latest_value: float
    mean_value: float
    min_value: float
    max_value: float


class EngineSensorHistory(BaseModel):
    sensor: str
    cycles: List[int]
    values: List[float]


# ── Dataset ───────────────────────────────────────────────────────────────────

class DatasetSummary(BaseModel):
    subset: str
    conditions: int
    fault_modes: int
    train_engines: int
    test_engines: int
    train_observations: int
    test_observations: int
    sensor_count: int
    complexity: str
    models_available: bool


# ── Maintenance ───────────────────────────────────────────────────────────────

class MaintenanceRecommendationCreate(BaseModel):
    dataset: str
    engine_unit: int


class MaintenanceRecommendationResponse(BaseModel):
    id: int
    engine_unit: int
    dataset_subset: str
    recommendation: str
    priority: str
    evidence: Dict[str, Any]
    status: str
    created_at: str

    class Config:
        from_attributes = True


class MaintenanceStatusUpdate(BaseModel):
    status: str = Field(..., description="pending | reviewed | resolved")


# ── Models page ───────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_key: str
    display_name: str
    model_type: str
    dataset: str
    is_available: bool
    metrics: Dict[str, Any]


class DashboardSummary(BaseModel):
    dataset: str
    total_engines: int
    avg_predicted_rul: float
    engines_rul_below_30: int
    engines_rul_below_15: int
    engines_high_failure_risk: int
    critical_engines: int
    high_risk_engines: int
    warning_engines: int
    healthy_engines: int
    rul_mae: float
    rul_r2: float
    failure_f1: float
    failure_roc_auc: float
    source_label: str
