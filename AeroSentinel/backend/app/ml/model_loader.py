"""
AeroSentinel — Model Loader
Loads all saved artifacts once at startup; provides graceful fallback.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from app.config import MODELS_DIR

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton that holds all loaded model artifacts."""

    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        self.rul_rf:    Any = None
        self.rul_gb:    Any = None
        self.fail_rf:   Any = None
        self.fail_gb:   Any = None
        self.scaler:    Any = None
        self.feature_cols: list[str] = []
        self.metrics:   Dict = {}
        self._loaded  = False

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def load(self):
        if self._loaded:
            return
        try:
            self.rul_rf  = self._load("rul_rf_fd001.joblib")
            self.rul_gb  = self._load("rul_gb_fd001.joblib")
            self.fail_rf = self._load("failure_rf_fd001.joblib")
            self.fail_gb = self._load("failure_gb_fd001.joblib")
            self.scaler  = self._load("scaler_fd001.joblib")

            fc_path = MODELS_DIR / "feature_cols_fd001.json"
            if fc_path.exists():
                self.feature_cols = json.loads(fc_path.read_text())

            mm_path = MODELS_DIR / "model_metrics_fd001.json"
            if mm_path.exists():
                self.metrics = json.loads(mm_path.read_text())

            self._loaded = True
            logger.info("All FD001 model artifacts loaded successfully.")
        except Exception as exc:
            logger.error(f"Model loading failed: {exc}")
            raise

    def _load(self, filename: str) -> Any:
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        model = joblib.load(path)
        logger.info(f"Loaded {filename}")
        return model

    # ── convenience ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._loaded and self.scaler is not None and len(self.feature_cols) > 0

    def status(self) -> Dict:
        return {
            "rul_rf":        self.rul_rf  is not None,
            "rul_gb":        self.rul_gb  is not None,
            "fail_rf":       self.fail_rf is not None,
            "fail_gb":       self.fail_gb is not None,
            "scaler":        self.scaler  is not None,
            "feature_cols":  len(self.feature_cols),
            "ready":         self.is_ready(),
        }

    def feature_importances(self, model_key: str = "rul_rf") -> list[dict]:
        model = getattr(self, model_key, None)
        if model is None or not hasattr(model, "feature_importances_"):
            return []
        imps = model.feature_importances_
        pairs = sorted(
            zip(self.feature_cols, imps.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return [{"feature": f, "importance": round(v, 6), "rank": i + 1}
                for i, (f, v) in enumerate(pairs)]
