import pandas as pd
import pytest

from src.batch_detection import CSVError, run_batch, validate_csv
from src.config import FEATURES, TARGET
from src.prediction import FraudDetector


def test_batch_scores_all_rows_and_keeps_target_separate(trained, synthetic_frame, tmp_path):
    det = FraudDetector.load(trained["model_path"])
    res, summary = run_batch(trained["csv"], det, chunk_size=400)        # file includes Class
    assert summary["total_scored"] == len(synthetic_frame) and summary["has_target"]
    assert "actual_class" in res.columns and summary["flagged"] + summary["legitimate"] == len(res)
    # predictions must be identical whether or not the Class column is present
    no_target = tmp_path / "nt.csv"
    synthetic_frame[FEATURES].to_csv(no_target, index=False)
    res2, s2 = run_batch(no_target, det, chunk_size=700)
    assert not s2["has_target"] and (res["predicted_class"].to_numpy() == res2["predicted_class"].to_numpy()).all()


def test_missing_columns_and_bad_file(trained, tmp_path):
    det = FraudDetector.load(trained["model_path"])
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Time": [1], "Amount": [2]}).to_csv(bad, index=False)
    assert not validate_csv(bad)["ok"]
    with pytest.raises(CSVError):
        run_batch(bad, det)
    with pytest.raises(CSVError):
        validate_csv(tmp_path / "missing.csv")


def test_invalid_rows_skipped_and_counted(trained, synthetic_frame, tmp_path):
    det = FraudDetector.load(trained["model_path"])
    df = synthetic_frame[FEATURES].head(20).astype(object).copy()
    df.loc[3, "V4"] = "oops"
    df.loc[5, "Amount"] = -10
    p = tmp_path / "inv.csv"
    df.to_csv(p, index=False)
    res, s = run_batch(p, det)
    assert s["skipped_invalid"] == 2 and s["total_scored"] == 18
