"""
AeroSentinel — SQLAlchemy ORM models
"""
import json
from datetime import datetime
from sqlalchemy import (Column, Integer, Float, String, DateTime,
                         Text, ForeignKey, Enum as SAEnum)
from sqlalchemy.orm import relationship
from app.database import Base


class Engine(Base):
    __tablename__ = "engines"

    id             = Column(Integer, primary_key=True, index=True)
    engine_unit    = Column(Integer, nullable=False)
    dataset_subset = Column(String(8), nullable=False)
    last_cycle     = Column(Integer)
    created_at     = Column(DateTime, default=datetime.utcnow)

    predictions     = relationship("Prediction",            back_populates="engine", cascade="all, delete-orphan")
    anomalies       = relationship("Anomaly",               back_populates="engine", cascade="all, delete-orphan")
    recommendations = relationship("MaintenanceRecommendation", back_populates="engine", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id                  = Column(Integer, primary_key=True, index=True)
    engine_id           = Column(Integer, ForeignKey("engines.id"), nullable=False)
    predicted_rul       = Column(Float)
    failure_probability = Column(Float)
    model_name          = Column(String(64))
    health_status       = Column(String(32))
    created_at          = Column(DateTime, default=datetime.utcnow)

    engine = relationship("Engine", back_populates="predictions")


class Anomaly(Base):
    __tablename__ = "anomalies"

    id                   = Column(Integer, primary_key=True, index=True)
    engine_id            = Column(Integer, ForeignKey("engines.id"), nullable=False)
    cycle                = Column(Integer)
    anomaly_score        = Column(Float)
    severity             = Column(String(32))
    contributing_sensors = Column(Text)   # JSON list
    created_at           = Column(DateTime, default=datetime.utcnow)

    engine = relationship("Engine", back_populates="anomalies")

    def get_contributing_sensors(self):
        try:
            return json.loads(self.contributing_sensors or "[]")
        except Exception:
            return []


class MaintenanceRecommendation(Base):
    __tablename__ = "maintenance_recommendations"

    id             = Column(Integer, primary_key=True, index=True)
    engine_id      = Column(Integer, ForeignKey("engines.id"), nullable=False)
    recommendation = Column(Text)
    priority       = Column(String(32))   # critical / high / medium / normal
    evidence       = Column(Text)         # JSON dict
    status         = Column(String(32), default="pending")  # pending/reviewed/resolved
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    engine = relationship("Engine", back_populates="recommendations")

    def get_evidence(self):
        try:
            return json.loads(self.evidence or "{}")
        except Exception:
            return {}


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id                 = Column(Integer, primary_key=True, index=True)
    model_name         = Column(String(64), unique=True)
    model_type         = Column(String(32))   # regression / classification / anomaly
    dataset_subset     = Column(String(8))
    evaluation_metrics = Column(Text)         # JSON
    artifact_path      = Column(String(256))
    is_available       = Column(Integer, default=0)  # 0/1 bool
    created_at         = Column(DateTime, default=datetime.utcnow)

    def get_metrics(self):
        try:
            return json.loads(self.evaluation_metrics or "{}")
        except Exception:
            return {}
