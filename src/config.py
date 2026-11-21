"""Central configuration: paths, constants, colours, settings persistence, logging."""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_NAME = "ZaBir 🛡️ Credit Card Fraud Detection System"
APP_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "fraud_detection_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
CONFIG_PATH = MODELS_DIR / "model_config.json"
DB_PATH = BASE_DIR / "database" / "fraud_detection.db"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"
SETTINGS_PATH = BASE_DIR / "settings.json"

# Exact feature names/order used by the Kaggle "creditcardfraud" dataset.
V_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time"] + V_FEATURES + ["Amount"]
TARGET = "Class"
RANDOM_STATE = 42

LABELS = {0: "Legitimate", 1: "Potential Fraud"}
REVIEW_STATUSES = ["Pending Review", "Reviewed", "Confirmed Fraud", "False Positive"]
RISK_LEVELS = ["Low Risk", "Medium Risk", "High Risk"]

DEFAULT_SETTINGS = {
    "threshold_override": None,      # None -> use the threshold chosen at training time
    "risk_medium_ratio": 0.5,        # Medium risk starts at ratio * decision threshold
    "store_full_features": False,    # privacy: do NOT store V1-V28 by default
    "threshold_beta": 1.0,           # F-beta used when choosing the threshold (training)
}

COLORS = {
    "navy": "#0B1F3A", "navy_light": "#13294B", "navy_hover": "#1C3A66",
    "blue": "#2F80ED", "blue_dark": "#1E5FBF", "bg": "#F3F6FB", "white": "#FFFFFF",
    "text": "#1B2A41", "muted": "#6B7A90", "border": "#D9E1EC",
    "green": "#27AE60", "orange": "#F2994A", "red": "#EB5757",
    "green_bg": "#E6F6EC", "orange_bg": "#FEF1E4", "red_bg": "#FDE9E9",
}


def load_settings() -> dict:
    """Load user settings, falling back to defaults on any problem."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_PATH.exists():
            settings.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Could not read settings.json: %s", exc)
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def setup_logging() -> None:
    """Configure rotating file logging + console logging (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        return
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(LOGS_DIR / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
