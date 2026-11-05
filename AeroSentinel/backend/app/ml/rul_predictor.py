"""
AeroSentinel — RUL Predictor
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict

from app.ml.model_loader import ModelRegistry
from app.ml.preprocessing import (engineer_features, get_feature_matrix,
                                   extract_last_cycle, load_raw, load_rul,
                                   health_status_from_rul)
from app.config import DATASET_DIR


def predict_single_engine(dataset: str, engine_unit: int, model_key: str = "rul_rf") -> Dict:
    """Predict RUL for a single engine unit using its full trajectory."""
    reg = ModelRegistry.get()
    train_file = DATASET_DIR / f"train_{dataset}.txt"
    test_file  = DATASET_DIR / f"test_{dataset}.txt"
    rul_file   = DATASET_DIR / f"RUL_{dataset}.txt"

    # Use test file for inference (same as original evaluation)
    test_df = load_raw(str(test_file))
    engine_df = test_df[test_df["unit"] == engine_unit].copy()
    if engine_df.empty:
        # Fall back to train file
        train_df = load_raw(str(train_file))
        engine_df = train_df[train_df["unit"] == engine_unit].copy()
        source = "train"
    else:
        source = "test"

    if engine_df.empty:
        raise ValueError(f"Engine unit {engine_unit} not found in {dataset}")

    # Engineer features (full trajectory)
    eng = engineer_features(engine_df)
    last_row = eng.iloc[[-1]]   # last cycle only
    X = get_feature_matrix(last_row, reg.feature_cols)
    X_scaled = reg.scaler.transform(X)

    model = getattr(reg, model_key)
    rul_pred = float(model.predict(X_scaled)[0])
    rul_pred = max(0.0, rul_pred)

    # Ground truth if available
    rul_true = None
    if source == "test" and rul_file.exists():
        rul_series = load_rul(str(rul_file))
        # RUL file is indexed 0-based in order of engine units
        all_units = sorted(test_df["unit"].unique())
        idx = all_units.index(engine_unit)
        if idx < len(rul_series):
            rul_true = float(rul_series.iloc[idx])

    return {
        "engine_unit":   engine_unit,
        "dataset":       dataset,
        "last_cycle":    int(engine_df["cycle"].max()),
        "predicted_rul": round(rul_pred, 2),
        "true_rul":      rul_true,
        "model":         model_key,
        "health_status": health_status_from_rul(rul_pred),
        "source":        source,
    }


def predict_all_engines(dataset: str, model_key: str = "rul_rf") -> List[Dict]:
    """Predict RUL for every engine in the test set of a dataset."""
    reg = ModelRegistry.get()
    test_file = DATASET_DIR / f"test_{dataset}.txt"
    rul_file  = DATASET_DIR / f"RUL_{dataset}.txt"

    test_df = load_raw(str(test_file))
    eng_df  = engineer_features(test_df)

    last_cycles = extract_last_cycle(eng_df)
    X = get_feature_matrix(last_cycles, reg.feature_cols)
    X_scaled = reg.scaler.transform(X)

    model = getattr(reg, model_key)
    preds = model.predict(X_scaled)
    preds = np.maximum(preds, 0)

    rul_true_series = None
    if rul_file.exists():
        rul_true_series = load_rul(str(rul_file))

    results = []
    for i, row in last_cycles.iterrows():
        unit = int(row["unit"])
        pred = round(float(preds[i - last_cycles.index[0]]), 2)
        true = float(rul_true_series.iloc[i - last_cycles.index[0]]) if rul_true_series is not None else None
        results.append({
            "engine_unit":   unit,
            "last_cycle":    int(row["cycle"]),
            "predicted_rul": pred,
            "true_rul":      true,
            "health_status": health_status_from_rul(pred),
        })
    return results
