"""
AeroSentinel — Failure Predictor
Binary classification: will engine fail within 30 cycles?
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

from app.ml.model_loader import ModelRegistry
from app.ml.preprocessing import (engineer_features, get_feature_matrix,
                                   extract_last_cycle, load_raw, load_rul,
                                   health_status_from_rul)
from app.config import DATASET_DIR


def predict_failure_single(dataset: str, engine_unit: int, model_key: str = "fail_gb") -> Dict:
    reg = ModelRegistry.get()
    test_file = DATASET_DIR / f"test_{dataset}.txt"
    train_file = DATASET_DIR / f"train_{dataset}.txt"

    test_df = load_raw(str(test_file))
    engine_df = test_df[test_df["unit"] == engine_unit].copy()
    if engine_df.empty:
        train_df = load_raw(str(train_file))
        engine_df = train_df[train_df["unit"] == engine_unit].copy()

    if engine_df.empty:
        raise ValueError(f"Engine {engine_unit} not found in {dataset}")

    eng = engineer_features(engine_df)
    last_row = eng.iloc[[-1]]
    X = get_feature_matrix(last_row, reg.feature_cols)
    X_scaled = reg.scaler.transform(X)

    model = getattr(reg, model_key)
    prob  = float(model.predict_proba(X_scaled)[0, 1])
    pred  = int(model.predict(X_scaled)[0])

    risk = "high" if prob >= 0.7 else ("medium" if prob >= 0.4 else "low")

    return {
        "engine_unit":          engine_unit,
        "dataset":              dataset,
        "failure_probability":  round(prob, 4),
        "failure_predicted":    bool(pred),
        "risk_level":           risk,
        "model":                model_key,
        "last_cycle":           int(engine_df["cycle"].max()),
    }


def predict_failure_all(dataset: str, model_key: str = "fail_gb") -> List[Dict]:
    reg = ModelRegistry.get()
    test_file = DATASET_DIR / f"test_{dataset}.txt"

    test_df = load_raw(str(test_file))
    eng_df  = engineer_features(test_df)
    last_cycles = extract_last_cycle(eng_df)

    X = get_feature_matrix(last_cycles, reg.feature_cols)
    X_scaled = reg.scaler.transform(X)

    model = getattr(reg, model_key)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    results = []
    for i, (idx, row) in enumerate(last_cycles.iterrows()):
        prob = round(float(probs[i]), 4)
        results.append({
            "engine_unit":         int(row["unit"]),
            "last_cycle":          int(row["cycle"]),
            "failure_probability": prob,
            "failure_predicted":   bool(preds[i]),
            "risk_level":          "high" if prob >= 0.7 else ("medium" if prob >= 0.4 else "low"),
        })
    return results
