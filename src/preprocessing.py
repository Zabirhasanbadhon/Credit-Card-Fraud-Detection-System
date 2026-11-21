"""Preprocessing: cleaning, leakage-free splitting, scaling and training-only resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import FEATURES, RANDOM_STATE, TARGET, V_FEATURES


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate transactions. Missing feature values are imputed later,
    inside the pipeline, using statistics learned from the training data only."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, {"rows_before": before, "rows_after": len(cleaned),
                     "duplicates_removed": before - len(cleaned)}


def stratified_split(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=RANDOM_STATE) -> dict:
    """Stratified train / validation / test split (original 0/1 labels preserved).
    Validation (used for model selection, calibration and threshold choice) is carved
    out of the training portion, so the test set is never touched until final scoring."""
    X, y = df[FEATURES], df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=seed)
    return {"X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}


def time_based_split(df: pd.DataFrame, test_fraction=0.2, val_fraction=0.2) -> dict:
    """Chronological split: oldest transactions train, newest test (more realistic for fraud)."""
    d = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n_test = int(len(d) * test_fraction)
    test, rest = d.iloc[len(d) - n_test:], d.iloc[:len(d) - n_test]
    n_val = int(len(rest) * val_fraction)
    val, train = rest.iloc[len(rest) - n_val:], rest.iloc[:len(rest) - n_val]
    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        out[f"X_{name}"], out[f"y_{name}"] = part[FEATURES], part[TARGET]
    return out


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + robust-scale Time and Amount; median-impute V1-V28 (already PCA-scaled).
    Must be fitted on training data only."""
    scaled = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    return ColumnTransformer(
        [("scaled", scaled, ["Time", "Amount"]), ("v", SimpleImputer(strategy="median"), V_FEATURES)],
        remainder="drop")


def feature_ranges(X: pd.DataFrame) -> dict:
    """Observed min/max per feature in the training data (used for range warnings)."""
    return {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES}


# ---------------------------------------------------------------- resampling (TRAIN ONLY)
def random_undersample(X, y, ratio=10.0, seed=RANDOM_STATE):
    """Keep all frauds and `ratio` x as many randomly chosen legitimate rows."""
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keep_neg = rng.choice(neg, size=min(len(neg), int(len(pos) * ratio)), replace=False)
    idx = rng.permutation(np.concatenate([pos, keep_neg]))
    return X[idx], y[idx]


def random_oversample(X, y, ratio=0.1, seed=RANDOM_STATE):
    """Duplicate frauds (with replacement) until frauds = ratio x legitimate rows."""
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    extra = rng.choice(pos, size=max(int(len(neg) * ratio) - len(pos), 0), replace=True)
    idx = rng.permutation(np.concatenate([neg, pos, extra]))
    return X[idx], y[idx]


def smote_resample(X, y, seed=RANDOM_STATE, sampling_strategy=0.1):
    """SMOTE via imbalanced-learn (applied to the training split only)."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise RuntimeError("SMOTE needs imbalanced-learn: pip install imbalanced-learn") from exc
    return SMOTE(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)


def apply_strategy(strategy: str, X, y, seed=RANDOM_STATE):
    """Apply an imbalance strategy to *training* data. 'class_weight' / 'none' leave data unchanged
    (class weighting is configured on the estimator itself)."""
    if strategy == "undersample":
        return random_undersample(X, y, seed=seed)
    if strategy == "oversample":
        return random_oversample(X, y, seed=seed)
    if strategy == "smote":
        return smote_resample(X, y, seed=seed)
    return np.asarray(X), np.asarray(y)
