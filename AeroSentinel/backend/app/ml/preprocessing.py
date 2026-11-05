"""
AeroSentinel — Preprocessing
Replicates v1.py feature engineering exactly.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

INDEX_NAMES   = ["unit", "cycle"]
SETTING_NAMES = ["setting1", "setting2", "setting3"]
SENSOR_NAMES  = [f"sensor{i}" for i in range(1, 22)]
COL_NAMES     = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

ROLLING_WINDOWS = [5, 10]


# ── Raw loading ───────────────────────────────────────────────────────────────

def load_raw(file_path: str) -> pd.DataFrame:
    """Load a raw C-MAPSS txt file into a DataFrame."""
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=COL_NAMES)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_rul(file_path: str) -> pd.Series:
    """Load the ground-truth RUL file."""
    return pd.read_csv(file_path, sep=r"\s+", header=None, names=["RUL"])["RUL"]


# ── Feature engineering (identical to v1.py) ─────────────────────────────────

def add_rul_target(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df.drop("max_cycle", axis=1)


def add_rolling_features(df: pd.DataFrame, window_sizes: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("sensor") and "rolling" not in c]
    new_cols: dict = {}
    for w in window_sizes:
        for s in sensor_cols:
            new_cols[f"{s}_rolling_mean_{w}"] = (
                df.groupby("unit")[s].transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            new_cols[f"{s}_rolling_std_{w}"] = (
                df.groupby("unit")[s].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
            )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_degradation_features(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("sensor") and "rolling" not in c]
    max_cycle_per_unit = df.groupby("unit")["cycle"].max().reset_index()
    max_cycle_per_unit.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycle_per_unit, on="unit", how="left")
    new_cols: dict = {}
    for s in sensor_cols:
        new_cols[f"{s}_diff_from_start"] = df.groupby("unit")[s].transform(lambda x: x - x.iloc[0])
        new_cols[f"{s}_normalized"]      = df[s] / (df["max_cycle"] + 1)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.drop("max_cycle", axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_rul_target(df)
    df = add_rolling_features(df, ROLLING_WINDOWS)
    df = add_degradation_features(df)
    df["failure_within_30"] = (df["RUL"] <= 30).astype(int)
    return df


EXCLUDE_COLS = {"unit", "cycle", "RUL", "failure_within_30"}


def get_feature_matrix(df_engineered: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Extract the feature matrix in the exact column order used during training."""
    missing = [c for c in feature_cols if c not in df_engineered.columns]
    if missing:
        raise ValueError(f"Missing engineered columns: {missing[:5]} …")
    return df_engineered[feature_cols].values


def extract_last_cycle(df_test: pd.DataFrame, true_rul_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """Get the final-cycle row for each engine in a test file."""
    last_idx = df_test.groupby("unit")["cycle"].idxmax()
    result = df_test.loc[last_idx].sort_values("unit").reset_index(drop=True)
    if true_rul_series is not None:
        result["RUL_true"] = true_rul_series.values
    return result


def health_status_from_rul(rul: float) -> str:
    if rul > 100:
        return "healthy"
    elif rul > 50:
        return "warning"
    elif rul > 20:
        return "high_risk"
    else:
        return "critical"
