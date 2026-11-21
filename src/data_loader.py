"""Dataset loading, validation and statistics (real data only - nothing is generated)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import DATASET_PATH, FEATURES, TARGET

log = logging.getLogger(__name__)

MANUAL_INSTRUCTIONS = f"""The dataset file was not found.

Manual download (recommended):
  1. Open https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (free Kaggle account required).
  2. Click "Download" and extract creditcard.zip.
  3. Copy creditcard.csv to:
     {DATASET_PATH}

Alternative (automatic, needs internet): run  python -m src.data_loader --download
(downloads the same ULB dataset from OpenML).
"""


class DatasetError(Exception):
    """Raised when the dataset is missing, corrupted or has an invalid structure."""


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the credit-card dataset."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(MANUAL_INSTRUCTIONS)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas raises several error types
        raise DatasetError(f"Could not read '{path.name}': {exc}") from exc
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {', '.join(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip("'\" "), errors="coerce")
    if df.empty:
        raise DatasetError("Dataset contains no rows.")
    valid_target = df[TARGET].isin([0, 1])
    if valid_target.sum() == 0:
        raise DatasetError("Target column 'Class' has no valid 0/1 labels.")
    if (~valid_target).any():
        log.warning("Dropping %d rows with invalid 'Class' values", int((~valid_target).sum()))
        df = df[valid_target].copy()
    df[TARGET] = df[TARGET].astype(int)
    if df[TARGET].nunique() < 2:
        raise DatasetError("Dataset needs both legitimate (0) and fraudulent (1) examples.")
    return df.reset_index(drop=True)


def dataset_statistics(df: pd.DataFrame) -> dict:
    """Shape, missing values, duplicates, class balance and amount statistics."""
    counts = df[TARGET].value_counts().to_dict()
    legit, fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df[FEATURES + [TARGET]].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "legitimate": legit,
        "fraudulent": fraud,
        "fraud_percentage": round(100.0 * fraud / max(len(df), 1), 4),
        "amount_mean": float(df["Amount"].mean()),
        "amount_max": float(df["Amount"].max()),
        "time_max": float(df["Time"].max()),
    }


def sample_real_rows(label: int, n: int = 1, path: Path | str = DATASET_PATH, seed=None) -> pd.DataFrame:
    """Return real rows from the local dataset (used by the 'load example' buttons)."""
    df = load_dataset(path)
    subset = df[df[TARGET] == label]
    return subset.sample(n=min(n, len(subset)), random_state=seed)


def download_from_openml(dest: Path | str = DATASET_PATH) -> Path:
    """Download the same ULB dataset from OpenML (data id 1597). Needs internet."""
    from sklearn.datasets import fetch_openml
    try:
        bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    except Exception as exc:
        raise DatasetError(f"Automatic download failed ({exc}).\n\n{MANUAL_INSTRUCTIONS}") from exc
    df = bunch.frame.copy()
    df[TARGET] = pd.to_numeric(df[TARGET].astype(str).str.strip("'\""), errors="coerce")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    log.info("Dataset saved to %s", dest)
    return dest


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        print(download_from_openml())
    else:
        print(dataset_statistics(load_dataset()))
