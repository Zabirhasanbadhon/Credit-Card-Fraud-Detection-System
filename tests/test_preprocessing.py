import numpy as np
import pandas as pd
from src.config import FEATURES, TARGET
from src.preprocessing import (build_preprocessor, clean_dataset, random_oversample, random_undersample,
                               stratified_split, time_based_split)


def test_clean_removes_duplicates(synthetic_frame):
    df = pd.concat([synthetic_frame, synthetic_frame.iloc[[0, 1]]], ignore_index=True)
    out, info = clean_dataset(df)
    assert info["duplicates_removed"] == 2 and len(out) == len(synthetic_frame)


def test_stratified_split_preserves_ratio_and_labels(synthetic_frame):
    sp = stratified_split(synthetic_frame)
    overall = synthetic_frame[TARGET].mean()
    for part in ("train", "val", "test"):
        assert abs(sp[f"y_{part}"].mean() - overall) < 0.02
        assert set(sp[f"y_{part}"].unique()) <= {0, 1}
    total = sum(len(sp[f"y_{p}"]) for p in ("train", "val", "test"))
    assert total == len(synthetic_frame)
    assert not set(sp["X_train"].index) & set(sp["X_test"].index)   # no overlap


def test_time_split_is_chronological(synthetic_frame):
    sp = time_based_split(synthetic_frame)
    assert sp["X_train"]["Time"].max() <= sp["X_val"]["Time"].min() <= sp["X_test"]["Time"].min()


def test_scaler_fitted_on_train_only(synthetic_frame):
    sp = stratified_split(synthetic_frame)
    pre = build_preprocessor().fit(sp["X_train"])
    scaler = pre.named_transformers_["scaled"].named_steps["scale"]
    assert np.isclose(scaler.center_[1], sp["X_train"]["Amount"].median())   # column 1 == Amount


def test_resampling_changes_only_what_it_is_given(synthetic_frame):
    X, y = synthetic_frame[FEATURES].to_numpy(), synthetic_frame[TARGET].to_numpy()
    Xu, yu = random_undersample(X, y, ratio=3)
    assert yu.sum() == y.sum() and (yu == 0).sum() == 3 * y.sum()
    Xo, yo = random_oversample(X, y, ratio=0.5)
    assert (yo == 0).sum() == (y == 0).sum() and yo.sum() > y.sum()
