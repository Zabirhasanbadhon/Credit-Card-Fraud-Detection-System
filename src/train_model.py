"""Train, compare, calibrate and save fraud-detection models.

Methodology (all fitting uses TRAINING data only):
  1. Load + validate dataset, drop exact duplicates.
  2. Stratified split: train / validation / test (60/20/20).
  3. Fit the preprocessor on the training part only.
  4. For each model: apply its imbalance strategy to the training part only, fit it,
     calibrate scores on the validation part (sigmoid), choose the decision threshold on
     the validation part, and only then score the untouched test part.
  5. Select the best supervised model by a balanced validation score
     (0.5 * PR-AUC + 0.5 * F1 at the validation-optimal threshold).
  6. Repeat the selected model on a chronological (time-based) split as a robustness check.
  7. Isolation Forest (unsupervised) is evaluated separately and is never auto-selected.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              IsolationForest, RandomForestClassifier)
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import (APP_VERSION, CONFIG_PATH, DATASET_PATH, FEATURES, METRICS_PATH, MODEL_PATH,
                     MODELS_DIR, RANDOM_STATE, TARGET, load_settings)
from .data_loader import dataset_statistics, load_dataset
from .preprocessing import (apply_strategy, build_preprocessor, clean_dataset, feature_ranges,
                            stratified_split, time_based_split)
from .threshold_optimizer import optimal_threshold

log = logging.getLogger(__name__)
RS = RANDOM_STATE

# name -> (estimator factory, default imbalance strategy)
MODEL_SPECS: dict[str, tuple[Callable, str]] = {
    "Logistic Regression": (lambda: LogisticRegression(max_iter=1000, class_weight="balanced"), "class_weight"),
    "Decision Tree": (lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced",
                                                     random_state=RS), "class_weight"),
    "Random Forest": (lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                     class_weight="balanced_subsample", random_state=RS), "class_weight"),
    "K-Nearest Neighbors": (lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "undersample"),
    # SVM scores are calibrated downstream (CalibratedClassifierCV), so probability=True is not needed.
    "Support Vector Machine": (lambda: SVC(kernel="rbf", random_state=RS), "undersample"),
    "Gradient Boosting": (lambda: GradientBoostingClassifier(random_state=RS), "undersample"),
    "Extra Trees": (lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                 class_weight="balanced_subsample", random_state=RS), "class_weight"),
}
ISOLATION_FOREST = "Isolation Forest (unsupervised)"


# ------------------------------------------------------------------ metrics helpers
def _downsample(*arrays, n=300):
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, min(n, len(arrays[0]))).astype(int))
    return [np.asarray(a)[idx].tolist() for a in arrays]


def compute_metrics(y_true, scores, threshold: float) -> dict:
    """All requested metrics at a fixed threshold (no hidden false positives/negatives)."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_true, pred, labels=[0, 1], target_names=["Legitimate", "Fraud"], output_dict=True, zero_division=0),
        "roc_curve": dict(zip(("fpr", "tpr"), _downsample(fpr, tpr))),
        "pr_curve": dict(zip(("precision", "recall"), _downsample(prec, rec))),
        "n_samples": int(len(y_true)), "n_fraud": int(y_true.sum()),
    }


def fit_candidate(name: str, strategy: str, pre, X_train, y_train, X_val, y_val, X_test, y_test, beta=1.0):
    """Fit preprocessing + one model on train, calibrate/threshold on val, score on test.
    Returns (result dict, fitted preprocessor, calibrated model)."""
    pre = pre
    Xt = pre.fit_transform(X_train)                    # fitted on TRAIN only
    Xv, Xs = pre.transform(X_val), pre.transform(X_test)
    factory, _ = MODEL_SPECS[name]
    Xr, yr = apply_strategy(strategy, Xt, y_train, seed=RS)   # resampling: TRAIN only
    est = factory().fit(Xr, yr)
    model = CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid").fit(Xv, y_val)
    val_scores, test_scores = model.predict_proba(Xv)[:, 1], model.predict_proba(Xs)[:, 1]
    thr, _ = optimal_threshold(y_val, val_scores, beta)
    val_m, test_m = compute_metrics(y_val, val_scores, thr), compute_metrics(y_test, test_scores, thr)
    result = {"name": name, "strategy": strategy, "threshold": thr, "validation": val_m, "test": test_m,
              "selection_score": 0.5 * val_m["pr_auc"] + 0.5 * val_m["f1"], "train_rows_used": int(len(yr))}
    return result, pre, model


def fit_isolation_forest(pre, X_train, y_train, X_val, y_val, X_test, y_test, beta=1.0) -> dict:
    """Unsupervised anomaly detector: labels are used only to choose a threshold on validation."""
    Xt = pre.fit_transform(X_train)
    Xv, Xs = pre.transform(X_val), pre.transform(X_test)
    contamination = float(np.clip(np.mean(y_train), 1e-4, 0.5))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=RS, n_jobs=-1).fit(Xt)
    sv, ss = -iso.score_samples(Xv), -iso.score_samples(Xs)   # higher = more anomalous
    thr, _ = optimal_threshold(y_val, sv, beta)
    return {"name": ISOLATION_FOREST, "strategy": "unsupervised (no labels used for fitting)",
            "threshold": thr, "validation": compute_metrics(y_val, sv, thr), "test": compute_metrics(y_test, ss, thr),
            "note": "Raw anomaly scores, not probabilities. Evaluated separately; never auto-selected."}


# ------------------------------------------------------------------ main entry point
def train_all(dataset_path=DATASET_PATH, progress: Callable[[str], None] | None = None,
              strategy_overrides: dict | None = None, beta: float | None = None) -> dict:
    """Run the whole training workflow and save model + metrics + config. Returns the metrics dict."""
    say = progress or (lambda m: log.info(m))
    beta = beta if beta is not None else float(load_settings().get("threshold_beta", 1.0))
    strategy_overrides = strategy_overrides or {}

    say("Loading and validating dataset ...")
    raw = load_dataset(dataset_path)
    stats = dataset_statistics(raw)
    df, clean_info = clean_dataset(raw)
    say(f"Rows: {stats['rows']:,} | frauds: {stats['fraudulent']:,} ({stats['fraud_percentage']}%) | "
        f"duplicates removed: {clean_info['duplicates_removed']:,}")

    sp = stratified_split(df)
    args = (sp["X_train"], sp["y_train"], sp["X_val"], sp["y_val"], sp["X_test"], sp["y_test"])
    say(f"Split (stratified): train={len(sp['y_train']):,} val={len(sp['y_val']):,} test={len(sp['y_test']):,}")

    results, fitted, failures = [], {}, {}
    for i, name in enumerate(MODEL_SPECS, 1):
        strategy = strategy_overrides.get(name, MODEL_SPECS[name][1])
        say(f"[{i}/{len(MODEL_SPECS)}] Training {name} (imbalance strategy: {strategy}) ...")
        try:
            res, pre, model = fit_candidate(name, strategy, build_preprocessor(), *args, beta=beta)
            results.append(res)
            fitted[name] = (pre, model)
            t = res["test"]
            say(f"    test recall={t['recall']:.3f} precision={t['precision']:.3f} PR-AUC={t['pr_auc']:.3f}")
        except Exception as exc:  # keep going: one failing model must not abort the run
            log.exception("Training failed for %s", name)
            failures[name] = str(exc)
            say(f"    FAILED: {exc}")
    if not results:
        raise RuntimeError("No model could be trained. See logs/app.log for details.")

    say("Training Isolation Forest (separate, unsupervised) ...")
    try:
        iso_result = fit_isolation_forest(build_preprocessor(), *args, beta=beta)
    except Exception as exc:
        log.exception("Isolation Forest failed")
        iso_result, failures[ISOLATION_FOREST] = None, str(exc)

    best = max(results, key=lambda r: r["selection_score"])
    say(f"Best supervised model (validation): {best['name']}")

    say("Time-based robustness check for the selected model ...")
    time_check = None
    try:
        ts = time_based_split(df)
        tres, _, _ = fit_candidate(best["name"], best["strategy"], build_preprocessor(),
                                   ts["X_train"], ts["y_train"], ts["X_val"], ts["y_val"],
                                   ts["X_test"], ts["y_test"], beta=beta)
        time_check = {"model": best["name"], "test": tres["test"], "threshold": tres["threshold"]}
    except Exception as exc:
        log.exception("Time-based check failed")
        failures["time_based_check"] = str(exc)

    pre, model = fitted[best["name"]]
    trained_at = datetime.now().isoformat(timespec="seconds")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"preprocessor": pre, "model": model, "model_name": best["name"], "threshold": best["threshold"],
                 "feature_names": FEATURES, "feature_ranges": feature_ranges(sp["X_train"]),
                 "trained_at": trained_at, "sklearn_version": sklearn.__version__}, MODEL_PATH)

    legit_amt, fraud_amt = df.loc[df[TARGET] == 0, "Amount"], df.loc[df[TARGET] == 1, "Amount"]
    edges = np.linspace(0, float(np.percentile(df["Amount"], 99)), 41)
    metrics = {
        "trained_at": trained_at, "dataset": stats, "cleaning": clean_info, "selected_model": best["name"],
        "selection_rule": "0.5*PR-AUC + 0.5*F1 on the validation split (supervised models only)",
        "models": results, "isolation_forest": iso_result, "time_based_check": time_check, "failures": failures,
        "amount_histograms": {"edges": edges.tolist(),
                              "legitimate": np.histogram(legit_amt, bins=edges)[0].tolist(),
                              "fraud": np.histogram(fraud_amt, bins=edges)[0].tolist(),
                              "note": "Amounts above the 99th percentile are excluded from the histogram."},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CONFIG_PATH.write_text(json.dumps({
        "app_version": APP_VERSION, "selected_model": best["name"], "decision_threshold": best["threshold"],
        "imbalance_strategy": best["strategy"], "feature_names": FEATURES, "random_state": RS,
        "split": "stratified 60/20/20 (train/validation/test)", "calibration": "sigmoid, fitted on validation split",
        "threshold_beta": beta, "trained_at": trained_at, "sklearn_version": sklearn.__version__,
        "feature_ranges": feature_ranges(sp["X_train"]),
    }, indent=2), encoding="utf-8")
    say("Training finished. Model, metrics and config saved.")
    return metrics


if __name__ == "__main__":
    from .config import setup_logging
    setup_logging()
    train_all(progress=print)
