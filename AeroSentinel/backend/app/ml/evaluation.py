"""
AeroSentinel — Model Evaluation helpers
"""
from __future__ import annotations
import json
import numpy as np
from app.config import MODELS_DIR


def get_fd001_metrics() -> dict:
    """Return the pre-computed FD001 evaluation metrics."""
    path = MODELS_DIR / "model_metrics_fd001.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_sensor_correlations(df) -> list:
    """Return sensor-vs-sensor correlation pairs."""
    from app.ml.preprocessing import SENSOR_NAMES
    sensor_cols = [c for c in SENSOR_NAMES if c in df.columns]
    corr = df[sensor_cols].corr()
    pairs = []
    for i, c1 in enumerate(sensor_cols):
        for j, c2 in enumerate(sensor_cols):
            if j <= i:
                continue
            pairs.append({"sensor_a": c1, "sensor_b": c2, "correlation": round(float(corr.loc[c1, c2]), 4)})
    return pairs
