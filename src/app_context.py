"""Shared application state passed to GUI pages (no Tkinter imports here)."""
from __future__ import annotations

import logging

from .analytics import load_metrics
from .config import load_settings, save_settings
from .database import Database
from .prediction import FraudDetector, ModelNotAvailableError

log = logging.getLogger(__name__)


class AppContext:
    def __init__(self):
        self.settings = load_settings()
        self.db = Database()
        self.detector: FraudDetector | None = None
        self.model_error: str | None = None
        self.metrics: dict | None = None
        self.reload_model()

    def reload_model(self) -> None:
        self.metrics = load_metrics()
        try:
            self.detector = FraudDetector.load(settings=self.settings)
            self.model_error = None
        except ModelNotAvailableError as exc:
            self.detector, self.model_error = None, str(exc)
            log.warning("Model unavailable: %s", exc)

    def update_settings(self, **changes) -> None:
        self.settings.update(changes)
        save_settings(self.settings)
        if self.detector:
            self.detector.settings = {**self.detector.settings, **self.settings}

    def history_record(self, row: dict, source: str, features: dict | None = None) -> dict:
        """Build a DB record. Full feature values are only kept if the user enabled that privacy option."""
        return {"amount": float(row["amount"]), "time_feature": float(row["time_feature"]),
                "predicted_class": int(row["predicted_class"]), "fraud_score": float(row["fraud_probability"]),
                "risk_category": row["risk_category"], "model_name": self.detector.model_name,
                "predicted_at": row.get("predicted_at"), "source": source,
                "features_json": features if (features and self.settings.get("store_full_features")) else None}
