import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.prediction import FraudDetector, ModelNotAvailableError
from src.threshold_optimizer import optimal_threshold


def test_training_produces_all_required_models_and_metrics(trained):
    m = trained["metrics"]
    names = {r["name"] for r in m["models"]}
    assert {"Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors",
            "Support Vector Machine", "Gradient Boosting", "Extra Trees"} <= names
    assert m["isolation_forest"] is not None
    for key in ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "confusion_matrix"):
        assert key in m["models"][0]["test"]
    assert m["selected_model"] in names


def test_detector_scores_frame_and_ignores_target(trained, synthetic_frame):
    det = FraudDetector.load(trained["model_path"])
    out = det.predict_frame(synthetic_frame.head(50))          # contains 'Class' column -> must be ignored
    assert len(out) == 50 and out["fraud_probability"].between(0, 1).all()
    assert set(out["predicted_class"]) <= {0, 1}
    assert set(out["risk_category"]) <= {"Low Risk", "Medium Risk", "High Risk"}
    flagged = out["fraud_probability"] >= det.threshold
    assert (out["predicted_class"] == flagged.astype(int)).all()


def test_threshold_override_and_single_prediction(trained, synthetic_frame):
    det = FraudDetector.load(trained["model_path"], {"threshold_override": 0.0})
    row = synthetic_frame.iloc[0][FEATURES].to_dict()
    assert det.predict_single(row)["predicted_class"] == 1       # threshold 0 flags everything
    det.settings["threshold_override"] = 1.01
    assert det.predict_single(row)["predicted_class"] == 0


def test_missing_columns_raise(trained):
    det = FraudDetector.load(trained["model_path"])
    with pytest.raises(ValueError):
        det.predict_frame(pd.DataFrame({"Time": [1], "Amount": [2]}))


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        FraudDetector.load(tmp_path / "nope.pkl")
    bad = tmp_path / "bad.pkl"
    bad.write_text("not a pickle")
    with pytest.raises(ModelNotAvailableError):
        FraudDetector.load(bad)


def test_optimal_threshold_uses_labels_correctly():
    y = np.array([0, 0, 0, 1, 1])
    thr, info = optimal_threshold(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert 0.3 < thr <= 0.8 and info["recall"] == 1.0
    with pytest.raises(ValueError):
        optimal_threshold(np.zeros(5), np.arange(5))


def test_time_based_check_present(trained):
    assert trained["metrics"]["time_based_check"] is not None
