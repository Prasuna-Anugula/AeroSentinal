"""
AeroSentinel — Data Service
Caches dataset DataFrames to avoid repeated disk reads.
"""
from __future__ import annotations
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.ml.preprocessing import load_raw, load_rul, engineer_features, SENSOR_NAMES
from app.config import DATASET_DIR, DATASET_META

logger = logging.getLogger(__name__)

# ── In-process cache ──────────────────────────────────────────────────────────
_cache: Dict[str, pd.DataFrame] = {}


def _get_df(key: str) -> Optional[pd.DataFrame]:
    return _cache.get(key)


def _set_df(key: str, df: pd.DataFrame):
    _cache[key] = df


def get_train_df(subset: str) -> pd.DataFrame:
    k = f"train_{subset}"
    if k not in _cache:
        _cache[k] = load_raw(str(DATASET_DIR / f"train_{subset}.txt"))
    return _cache[k]


def get_test_df(subset: str) -> pd.DataFrame:
    k = f"test_{subset}"
    if k not in _cache:
        _cache[k] = load_raw(str(DATASET_DIR / f"test_{subset}.txt"))
    return _cache[k]


def get_rul_series(subset: str) -> pd.Series:
    k = f"rul_{subset}"
    if k not in _cache:
        _cache[k] = load_rul(str(DATASET_DIR / f"RUL_{subset}.txt"))
    return _cache[k]


# ── Dataset summary ───────────────────────────────────────────────────────────

def get_dataset_summary(subset: str) -> dict:
    meta  = DATASET_META.get(subset, {})
    train = get_train_df(subset)
    test  = get_test_df(subset)
    from app.ml.model_loader import ModelRegistry
    reg   = ModelRegistry.get()
    return {
        "subset":             subset,
        "conditions":         meta.get("conditions", 0),
        "fault_modes":        meta.get("fault_modes", 0),
        "train_engines":      int(train["unit"].nunique()),
        "test_engines":       int(test["unit"].nunique()),
        "train_observations": int(len(train)),
        "test_observations":  int(len(test)),
        "sensor_count":       21,
        "complexity":         meta.get("complexity", "Unknown"),
        "models_available":   reg.is_ready() and subset == "FD001",
    }


def get_available_subsets() -> List[str]:
    available = []
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        train_file = DATASET_DIR / f"train_{ds}.txt"
        test_file  = DATASET_DIR / f"test_{ds}.txt"
        if train_file.exists() and test_file.exists():
            available.append(ds)
    return available


# ── Engine data ───────────────────────────────────────────────────────────────

def get_engine_units(subset: str, source: str = "test") -> List[int]:
    df = get_test_df(subset) if source == "test" else get_train_df(subset)
    return sorted(df["unit"].unique().tolist())


def get_engine_sensor_history(subset: str, engine_unit: int) -> List[dict]:
    test_df   = get_test_df(subset)
    engine_df = test_df[test_df["unit"] == engine_unit]
    if engine_df.empty:
        train_df  = get_train_df(subset)
        engine_df = train_df[train_df["unit"] == engine_unit]
    if engine_df.empty:
        return []

    result = []
    cycles = engine_df["cycle"].tolist()
    for sensor in SENSOR_NAMES:
        if sensor in engine_df.columns:
            result.append({
                "sensor": sensor,
                "cycles": cycles,
                "values": engine_df[sensor].tolist(),
            })
    return result


def get_engine_sensor_snapshot(subset: str, engine_unit: int) -> List[dict]:
    test_df   = get_test_df(subset)
    engine_df = test_df[test_df["unit"] == engine_unit]
    if engine_df.empty:
        train_df  = get_train_df(subset)
        engine_df = train_df[train_df["unit"] == engine_unit]
    if engine_df.empty:
        return []

    result = []
    for sensor in SENSOR_NAMES:
        if sensor in engine_df.columns:
            vals = engine_df[sensor]
            result.append({
                "sensor":        sensor,
                "latest_value":  round(float(vals.iloc[-1]), 4),
                "mean_value":    round(float(vals.mean()),   4),
                "min_value":     round(float(vals.min()),    4),
                "max_value":     round(float(vals.max()),    4),
            })
    return result


# ── EDA statistics ────────────────────────────────────────────────────────────

def get_lifecycle_stats(subset: str) -> dict:
    train = get_train_df(subset)
    lifecycle = train.groupby("unit")["cycle"].max()
    return {
        "mean":   round(float(lifecycle.mean()), 1),
        "median": round(float(lifecycle.median()), 1),
        "min":    int(lifecycle.min()),
        "max":    int(lifecycle.max()),
        "std":    round(float(lifecycle.std()), 1),
        "distribution": lifecycle.tolist(),
    }


def get_rul_distribution(subset: str) -> dict:
    rul = get_rul_series(subset)
    return {
        "mean":         round(float(rul.mean()), 1),
        "median":       round(float(rul.median()), 1),
        "min":          int(rul.min()),
        "max":          int(rul.max()),
        "std":          round(float(rul.std()), 1),
        "values":       rul.tolist(),
    }
