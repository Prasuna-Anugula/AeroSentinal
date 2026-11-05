"""
AeroSentinel — Anomaly Detector
Isolation Forest trained on normalised sensor readings.
"""
from __future__ import annotations
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import joblib

from app.ml.preprocessing import load_raw, engineer_features, SENSOR_NAMES
from app.config import DATASET_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_if_fd001.joblib"
ANOMALY_SCALER_PATH = MODELS_DIR / "anomaly_scaler_fd001.joblib"

# Sensor columns used for anomaly detection (raw sensors only)
ANOMALY_FEATURES = SENSOR_NAMES  # sensor1..sensor21

_anomaly_model  = None
_anomaly_scaler = None


def _load_or_train():
    global _anomaly_model, _anomaly_scaler
    if _anomaly_model is not None:
        return

    if ANOMALY_MODEL_PATH.exists() and ANOMALY_SCALER_PATH.exists():
        _anomaly_model  = joblib.load(ANOMALY_MODEL_PATH)
        _anomaly_scaler = joblib.load(ANOMALY_SCALER_PATH)
        logger.info("Anomaly model loaded from disk.")
        return

    # Train on FD001 training data
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    logger.info("Training Isolation Forest for anomaly detection …")
    train_file = DATASET_DIR / "train_FD001.txt"
    df = load_raw(str(train_file))
    X  = df[ANOMALY_FEATURES].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_sc)

    joblib.dump(model,  ANOMALY_MODEL_PATH)
    joblib.dump(scaler, ANOMALY_SCALER_PATH)

    _anomaly_model  = model
    _anomaly_scaler = scaler
    logger.info("Anomaly model trained and saved.")


def detect_anomalies_engine(dataset: str, engine_unit: int) -> Dict:
    """Return anomaly scores for every cycle of a given engine."""
    _load_or_train()

    test_file  = DATASET_DIR / f"test_{dataset}.txt"
    train_file = DATASET_DIR / f"train_{dataset}.txt"

    df_test = load_raw(str(test_file))
    engine_df = df_test[df_test["unit"] == engine_unit].copy()
    if engine_df.empty:
        df_train = load_raw(str(train_file))
        engine_df = df_train[df_train["unit"] == engine_unit].copy()
    if engine_df.empty:
        raise ValueError(f"Engine {engine_unit} not found in {dataset}")

    X      = engine_df[ANOMALY_FEATURES].values
    X_sc   = _anomaly_scaler.transform(X)
    scores = _anomaly_model.decision_function(X_sc)   # more negative = more anomalous
    labels = _anomaly_model.predict(X_sc)              # -1 = anomaly, 1 = normal

    def _severity(s: float) -> str:
        if s < -0.20:
            return "high_anomaly"
        elif s < -0.10:
            return "unusual"
        return "normal"

    # Per-sensor contribution: deviation from mean in scaled space
    X_mean = _anomaly_scaler.mean_
    X_std  = _anomaly_scaler.scale_
    deviations = np.abs((X - X_mean) / (X_std + 1e-8))

    cycle_results = []
    for i, (_, row) in enumerate(engine_df.iterrows()):
        top_sensors = np.argsort(deviations[i])[::-1][:5]
        top_sensor_names = [ANOMALY_FEATURES[j] for j in top_sensors]
        cycle_results.append({
            "cycle":         int(row["cycle"]),
            "anomaly_score": round(float(scores[i]), 4),
            "is_anomaly":    bool(labels[i] == -1),
            "severity":      _severity(float(scores[i])),
            "top_sensors":   top_sensor_names,
        })

    last_score = float(scores[-1])
    return {
        "engine_unit":   engine_unit,
        "dataset":       dataset,
        "overall_score": round(last_score, 4),
        "severity":      _severity(last_score),
        "anomaly_count": int(np.sum(labels == -1)),
        "total_cycles":  len(cycle_results),
        "cycles":        cycle_results,
    }


def detect_anomalies_all(dataset: str) -> List[Dict]:
    """Return last-cycle anomaly summary for all engines in a dataset."""
    _load_or_train()

    test_file = DATASET_DIR / f"test_{dataset}.txt"
    df = load_raw(str(test_file))

    results = []
    for unit in sorted(df["unit"].unique()):
        engine_df = df[df["unit"] == unit]
        X      = engine_df[ANOMALY_FEATURES].values
        X_sc   = _anomaly_scaler.transform(X)
        scores = _anomaly_model.decision_function(X_sc)
        labels = _anomaly_model.predict(X_sc)

        last_score = float(scores[-1])

        def _sev(s): return "high_anomaly" if s < -0.20 else ("unusual" if s < -0.10 else "normal")

        results.append({
            "engine_unit":   int(unit),
            "last_cycle":    int(engine_df["cycle"].max()),
            "anomaly_score": round(last_score, 4),
            "severity":      _sev(last_score),
            "anomaly_count": int(np.sum(labels == -1)),
        })
    return results
