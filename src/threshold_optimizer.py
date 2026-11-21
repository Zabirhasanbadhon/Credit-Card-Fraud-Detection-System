"""Choose a decision threshold from VALIDATION data (never from the test set)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def optimal_threshold(y_true, scores, beta: float = 1.0) -> tuple[float, dict]:
    """Threshold maximising F-beta (beta>1 favours recall, beta<1 favours precision)."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        raise ValueError("Cannot optimise a threshold without positive (fraud) examples.")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    p, r = precision[:-1], recall[:-1]
    b2 = beta ** 2
    denom = b2 * p + r
    fbeta = np.divide((1 + b2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(fbeta))
    return float(thresholds[i]), {"precision": float(p[i]), "recall": float(r[i]), "fbeta": float(fbeta[i])}
