"""
AeroSentinel — Explanation Service
Feature importance + sensor correlations.
"""
from __future__ import annotations
import logging
from typing import Dict, List

from app.ml.model_loader import ModelRegistry
from app.ml.evaluation   import get_sensor_correlations
from app.services.data_service import get_train_df

logger = logging.getLogger(__name__)


def get_feature_importance(model_key: str = "rul_rf", top_n: int = 30) -> Dict:
    reg = ModelRegistry.get()
    items = reg.feature_importances(model_key)[:top_n]
    return {"model_key": model_key, "items": items}


def get_sensor_correlation_matrix(subset: str = "FD001") -> List[Dict]:
    df = get_train_df(subset)
    return get_sensor_correlations(df)


def get_engine_explanation(dataset: str, engine_unit: int, model_key: str = "rul_rf") -> Dict:
    """
    Local explanation: which features contributed most for this engine's last cycle.
    Uses difference from the training-set feature mean as a simple local contribution proxy
    (no SHAP dependency — compatible with any sklearn model).
    """
    import numpy as np
    from app.ml.preprocessing import (engineer_features, get_feature_matrix,
                                       load_raw, SENSOR_NAMES)
    from app.config import DATASET_DIR

    reg = ModelRegistry.get()

    test_file  = DATASET_DIR / f"test_{dataset}.txt"
    train_file = DATASET_DIR / f"train_{dataset}.txt"
    df_test  = load_raw(str(test_file))
    engine_df = df_test[df_test["unit"] == engine_unit].copy()
    if engine_df.empty:
        df_train  = load_raw(str(train_file))
        engine_df = df_train[df_train["unit"] == engine_unit].copy()

    eng = engineer_features(engine_df)
    last_row = eng.iloc[[-1]]
    X = get_feature_matrix(last_row, reg.feature_cols)
    X_scaled = reg.scaler.transform(X)

    # Global feature importances as weights
    model = getattr(reg, model_key)
    importances = model.feature_importances_

    # Local deviation from training mean (in scaled space)
    feature_values = X_scaled[0]
    contributions  = importances * np.abs(feature_values)
    top_idx        = np.argsort(contributions)[::-1][:20]

    items = []
    for i in top_idx:
        items.append({
            "feature":      reg.feature_cols[i],
            "importance":   round(float(importances[i]), 6),
            "scaled_value": round(float(feature_values[i]), 4),
            "contribution": round(float(contributions[i]), 6),
            "rank":         int(np.where(np.argsort(contributions)[::-1] == i)[0][0]) + 1,
        })

    model_obj = getattr(reg, model_key)
    pred = float(model_obj.predict(X_scaled)[0])

    return {
        "engine_unit": engine_unit,
        "dataset":     dataset,
        "model_key":   model_key,
        "prediction":  round(pred, 2),
        "method":      "local_feature_contribution",
        "method_note": "Approximation: global importance × |scaled feature value|. Not TreeSHAP.",
        "top_features": items,
    }
