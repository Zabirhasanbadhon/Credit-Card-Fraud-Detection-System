"""Fraud detection engine: loads the saved bundle and scores transactions."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_SETTINGS, FEATURES, LABELS, MODEL_PATH

log = logging.getLogger(__name__)


class ModelNotAvailableError(Exception):
    """Raised when the trained model file is missing or cannot be loaded."""


class FraudDetector:
    """Wraps preprocessor + calibrated model + threshold. Only FEATURES are ever passed to the model."""

    def __init__(self, bundle: dict, settings: dict | None = None):
        self.bundle = bundle
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.model_name = bundle["model_name"]
        self.trained_threshold = float(bundle["threshold"])
        self.feature_ranges = bundle.get("feature_ranges", {})

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH, settings: dict | None = None) -> "FraudDetector":
        path = Path(path)
        if not path.exists():
            raise ModelNotAvailableError("No trained model found. Train the models first "
                                         "(Settings -> Retrain Model, or: python -m src.train_model).")
        try:
            bundle = joblib.load(path)   # local file produced by this app only - do not load untrusted pickles
            for key in ("preprocessor", "model", "model_name", "threshold"):
                if key not in bundle:
                    raise KeyError(key)
        except Exception as exc:
            log.exception("Model loading failed")
            raise ModelNotAvailableError(f"The model file could not be loaded ({exc}). "
                                         "Please retrain the model.") from exc
        return cls(bundle, settings)

    @property
    def threshold(self) -> float:
        override = self.settings.get("threshold_override")
        return float(override) if override not in (None, "") else self.trained_threshold

    def risk_category(self, score: float) -> str:
        """Configurable bands relative to the decision threshold (not statistically validated bands)."""
        if score >= self.threshold:
            return "High Risk"
        if score >= self.threshold * float(self.settings.get("risk_medium_ratio", 0.5)):
            return "Medium Risk"
        return "Low Risk"

    def out_of_range_features(self, df: pd.DataFrame) -> pd.Series:
        """Comma-separated list of features outside the training range, per row (empty if none)."""
        flags = pd.DataFrame(index=df.index)
        for col, (lo, hi) in self.feature_ranges.items():
            if col in df:
                flags[col] = (df[col] < lo) | (df[col] > hi)
        if flags.empty:
            return pd.Series([""] * len(df), index=df.index)
        arr, cols = flags.to_numpy(), np.array(flags.columns)
        return pd.Series([", ".join(cols[row]) for row in arr], index=df.index)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame holding all FEATURES (extra columns such as 'Class' are ignored)."""
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        X = df[FEATURES].astype(float)
        scores = self.bundle["model"].predict_proba(self.bundle["preprocessor"].transform(X))[:, 1]
        pred = (scores >= self.threshold).astype(int)
        return pd.DataFrame({
            "predicted_class": pred,
            "prediction_label": [LABELS[int(p)] for p in pred],
            "fraud_probability": scores,
            "risk_category": [self.risk_category(s) for s in scores],
            "out_of_range_features": self.out_of_range_features(X).to_numpy(),
        }, index=df.index)

    def predict_single(self, values: dict) -> dict:
        """Score one validated transaction and return a result dictionary."""
        res = self.predict_frame(pd.DataFrame([values]))
        row = res.iloc[0].to_dict()
        row.update({"model_name": self.model_name, "amount": values["Amount"], "time_feature": values["Time"],
                    "threshold": self.threshold, "predicted_at": datetime.now().isoformat(timespec="seconds")})
        row["predicted_class"] = int(row["predicted_class"])
        return row
