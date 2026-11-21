"""Shared fixtures.

IMPORTANT: `synthetic_frame` creates a tiny dataset with the real column schema ONLY to exercise code
paths in unit tests. It is never used to train the model that the application ships or uses.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FEATURES, TARGET  # noqa: E402


def make_frame(n=1500, fraud_rate=0.06, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28)) + y[:, None] * np.linspace(1.5, 0, 28)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = rng.gamma(2.0, 40.0, n)
    df[TARGET] = y
    return df[FEATURES + [TARGET]]


@pytest.fixture(scope="session")
def synthetic_frame():
    return make_frame()


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic_frame):
    """Run the real training workflow on the synthetic fixture inside a temp directory."""
    import src.train_model as tm
    tmp = tmp_path_factory.mktemp("models")
    csv_path = tmp / "creditcard.csv"
    synthetic_frame.to_csv(csv_path, index=False)
    patch = {"MODELS_DIR": tmp, "MODEL_PATH": tmp / "m.pkl", "METRICS_PATH": tmp / "metrics.json",
             "CONFIG_PATH": tmp / "config.json"}
    old = {k: getattr(tm, k) for k in patch}
    for k, v in patch.items():
        setattr(tm, k, v)
    try:
        metrics = tm.train_all(csv_path, progress=lambda m: None)
    finally:
        for k, v in old.items():
            setattr(tm, k, v)
    return {"metrics": metrics, "model_path": tmp / "m.pkl", "csv": csv_path}
