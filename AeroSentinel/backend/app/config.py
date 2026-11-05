"""
AeroSentinel — Backend Configuration
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # AeroSentinel/

EXISTING_ML_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "NASATurbofanEnginePredictiveMaintenance"
)

MODELS_DIR   = EXISTING_ML_DIR / "models"
DATASET_DIR  = EXISTING_ML_DIR
STATIC_DIR   = EXISTING_ML_DIR          # PNGs live here
STORAGE_DIR  = ROOT / "storage"

DB_PATH = STORAGE_DIR / "aerosentinel.db"

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASET_META = {
    "FD001": {"conditions": 1, "fault_modes": 1, "train_engines": 100,  "test_engines": 100,  "complexity": "Simple"},
    "FD002": {"conditions": 6, "fault_modes": 1, "train_engines": 260,  "test_engines": 259,  "complexity": "Medium"},
    "FD003": {"conditions": 1, "fault_modes": 2, "train_engines": 100,  "test_engines": 100,  "complexity": "Medium"},
    "FD004": {"conditions": 6, "fault_modes": 2, "train_engines": 249,  "test_engines": 248,  "complexity": "Complex"},
}

# ── Health thresholds ────────────────────────────────────────────────────────
RUL_THRESHOLDS = {
    "healthy":   100,  # > 100 cycles
    "warning":   50,   # 50–100 cycles
    "high_risk": 20,   # 20–49 cycles
    "critical":  0,    # < 20 cycles
}

# ── Anomaly thresholds ────────────────────────────────────────────────────────
ANOMALY_THRESHOLDS = {
    "normal":       -0.05,
    "unusual":      -0.10,
    "high_anomaly": -0.20,
}
