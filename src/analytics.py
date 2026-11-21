"""Matplotlib figure builders. Dataset/model figures come from saved training metrics;
history figures come from the prediction database. Dataset labels are ACTUAL labels,
history figures show model PREDICTIONS - titles say which is which."""
from __future__ import annotations

import json

import numpy as np
from matplotlib.figure import Figure

from .config import COLORS, METRICS_PATH

C = COLORS
PALETTE = [C["blue"], C["green"], C["orange"], C["red"], "#9B51E0", "#56CCF2", "#F2C94C", "#4F4F4F"]


def load_metrics() -> dict | None:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None
    except (OSError, ValueError):
        return None


def _fig(title: str, size=(7, 4.2)):
    fig = Figure(figsize=size, dpi=100, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=11, fontweight="bold", color=C["navy"])
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def _empty(title: str, msg: str):
    fig, ax = _fig(title)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, color=C["muted"], fontsize=10)
    return fig


def class_distribution(metrics: dict | None, stats: dict):
    fig = Figure(figsize=(8, 4.2), dpi=100, facecolor="white")
    a1, a2 = fig.add_subplot(121), fig.add_subplot(122)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    if metrics:
        d = metrics["dataset"]
        a1.bar(["Legitimate", "Fraud"], [d["legitimate"], d["fraudulent"]], color=[C["green"], C["red"]])
        a1.set_yscale("log")
    a1.set_title("Dataset: ACTUAL labels (log scale)", fontsize=10, fontweight="bold")
    a2.bar(["Legitimate", "Flagged"], [stats["legitimate"], stats["flagged"]], color=[C["green"], C["red"]])
    a2.set_title("History: model PREDICTIONS", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def trends(stats: dict):
    days = stats["daily"]
    if not days:
        return _empty("Fraud Detection Trends Over Time", "No prediction history yet.")
    fig, ax = _fig("Flagged Transactions Over Time (model predictions)")
    x = [d["day"] for d in days]
    ax.plot(x, [d["total"] for d in days], marker="o", color=C["blue"], label="Analyzed")
    ax.plot(x, [d["flagged"] or 0 for d in days], marker="o", color=C["red"], label="Flagged")
    ax.set_ylabel("Transactions"); ax.legend()
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def amount_distribution(metrics: dict | None, fraud: bool):
    title = f"{'Fraudulent' if fraud else 'Legitimate'} Transaction Amounts (dataset, actual labels)"
    if not metrics or "amount_histograms" not in metrics:
        return _empty(title, "Train the models to compute dataset statistics.")
    h = metrics["amount_histograms"]
    fig, ax = _fig(title)
    edges = np.array(h["edges"])
    ax.bar(edges[:-1], h["fraud" if fraud else "legitimate"], width=np.diff(edges), align="edge",
           color=C["red"] if fraud else C["green"], edgecolor="white")
    ax.set_xlabel("Amount (up to 99th percentile)"); ax.set_ylabel("Transactions")
    fig.tight_layout()
    return fig


def model_comparison(metrics: dict | None, metric: str):
    title = f"Model {metric.replace('_', ' ').upper() if metric == 'f1' else metric.title()} Comparison (test set)"
    if not metrics:
        return _empty(title, "Train the models first.")
    rows = metrics["models"] + ([metrics["isolation_forest"]] if metrics.get("isolation_forest") else [])
    names = [r["name"].replace(" (unsupervised)", "*") for r in rows]
    vals = [r["test"][metric] for r in rows]
    fig, ax = _fig(title, (8, 4.4))
    bars = ax.bar(range(len(vals)), vals, color=[PALETTE[i % len(PALETTE)] for i in range(len(vals))])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    if any("unsupervised" in r["name"] for r in rows):
        ax.text(0.99, -0.42, "* unsupervised anomaly detector (separate)", transform=ax.transAxes,
                ha="right", fontsize=7, color=C["muted"])
    fig.tight_layout()
    return fig


def _find(metrics, name):
    for r in metrics["models"] + ([metrics["isolation_forest"]] if metrics.get("isolation_forest") else []):
        if r["name"] == name:
            return r
    return None


def confusion_matrix_fig(metrics: dict | None, name: str | None = None):
    if not metrics:
        return _empty("Confusion Matrix", "Train the models first.")
    name = name or metrics["selected_model"]
    r = _find(metrics, name)
    cm = r["test"]["confusion_matrix"]
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = _fig(f"Confusion Matrix - {name} (test set, threshold {r['test']['threshold']:.3f})")
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. Legitimate", "Pred. Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Legitimate", "Actual Fraud"])
    labels = [["True Negatives", "False Positives"], ["False Negatives", "True Positives"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{arr[i, j]:,}\n{labels[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if arr[i, j] > arr.max() / 2 else C["text"])
    fig.tight_layout()
    return fig


def curve_fig(metrics: dict | None, kind: str):
    title = "ROC Curve (test set)" if kind == "roc" else "Precision-Recall Curve (test set)"
    if not metrics:
        return _empty(title, "Train the models first.")
    fig, ax = _fig(title, (7, 4.6))
    rows = metrics["models"] + ([metrics["isolation_forest"]] if metrics.get("isolation_forest") else [])
    for i, r in enumerate(rows):
        t = r["test"]
        if kind == "roc":
            ax.plot(t["roc_curve"]["fpr"], t["roc_curve"]["tpr"], color=PALETTE[i % 8],
                    label=f"{r['name'].split(' (')[0]} (AUC {t['roc_auc']:.3f})")
        else:
            ax.plot(t["pr_curve"]["recall"], t["pr_curve"]["precision"], color=PALETTE[i % 8],
                    label=f"{r['name'].split(' (')[0]} (AP {t['pr_auc']:.3f})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    else:
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


CHARTS = {
    "1. Legitimate vs Fraudulent Distribution": lambda m, s: class_distribution(m, s),
    "2. Fraud Detection Trends Over Time": lambda m, s: trends(s),
    "3. Fraudulent Amount Distribution": lambda m, s: amount_distribution(m, True),
    "4. Legitimate Amount Distribution": lambda m, s: amount_distribution(m, False),
    "5. Model Accuracy Comparison": lambda m, s: model_comparison(m, "accuracy"),
    "6. Model Precision Comparison": lambda m, s: model_comparison(m, "precision"),
    "7. Model Recall Comparison": lambda m, s: model_comparison(m, "recall"),
    "8. Model F1-score Comparison": lambda m, s: model_comparison(m, "f1"),
    "9. Confusion Matrix (selected model)": lambda m, s: confusion_matrix_fig(m),
    "10. Precision-Recall Curve": lambda m, s: curve_fig(m, "pr"),
    "11. ROC Curve": lambda m, s: curve_fig(m, "roc"),
}
