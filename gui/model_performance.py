"""Model performance page: comparison table, confusion matrix, curves, reports and explanations."""
import tkinter as tk
from tkinter import ttk

from src.analytics import confusion_matrix_fig, curve_fig
from src.config import COLORS as C
from .styles import Card, Page, fill_tree, font, make_tree, show_canvas

EXPLAIN = """WHY THESE METRICS MATTER FOR FRAUD DETECTION

- Fraud is extremely rare (~0.17% of the dataset), so ACCURACY is misleading: a model that calls everything legitimate is ~99.8% accurate yet catches no fraud.
- RECALL (sensitivity): of all real frauds, how many did the model catch? Missed fraud = false negatives (financial loss).
- PRECISION: of all transactions the model flagged, how many were truly fraud? Low precision = many false positives (investigation cost, customer friction).
- FALSE POSITIVE RATE / SPECIFICITY: how many legitimate transactions were wrongly flagged.
- PR-AUC (average precision): threshold-free summary of the precision/recall trade-off; far more informative than ROC-AUC on imbalanced data.
- ROC-AUC: ranking quality over all thresholds; can look optimistic when negatives vastly outnumber positives.

HOW THIS APPLICATION EVALUATES
- Stratified 60/20/20 train / validation / test split; preprocessing and any resampling are fitted on training data only.
- Scores are calibrated (sigmoid) and the decision threshold is chosen on the VALIDATION split; all numbers in the table are from the untouched TEST split.
- The best supervised model is chosen with a balanced rule: 0.5*PR-AUC + 0.5*F1 on validation - not by accuracy.
- Isolation Forest is an unsupervised anomaly detector; it is listed for comparison but evaluated separately and never auto-selected.
- A chronological (time-based) check of the selected model is shown below when available.
- False positives and false negatives are shown in every confusion matrix. Results on this dataset do not guarantee performance on other portfolios.
"""


class ModelPerformancePage(Page):
    title = "Model Performance"
    subtitle = "Held-out test-set results for every trained model (nothing is hidden or estimated)."

    def build(self):
        self.info_lbl = tk.Label(self, bg=C["bg"], fg=C["text"], font=font(10, "bold"), anchor="w", justify="left", wraplength=1100)
        self.info_lbl.pack(fill="x", padx=24)
        cols = ["model", "accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "strategy"]
        card = Card(self, "Model comparison")
        card.pack(fill="x", padx=24, pady=8)
        fr, self.tree = make_tree(card.body, cols, {"model": 200, "strategy": 150}, height=8)
        fr.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.pick())
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self.tabs = {}
        for name in ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Classification Report", "Explanation"):
            f = tk.Frame(self.nb, bg=C["white"])
            self.nb.add(f, text=f"  {name}  ")
            self.tabs[name] = f
        self.report = tk.Text(self.tabs["Classification Report"], font=("Consolas", 10), bg=C["white"], relief="flat", wrap="word")
        self.report.pack(fill="both", expand=True, padx=10, pady=10)
        self.explain = tk.Text(self.tabs["Explanation"], font=font(10), bg=C["white"], relief="flat", wrap="word")
        self.explain.pack(fill="both", expand=True, padx=10, pady=10)
        self.explain.insert("1.0", EXPLAIN)
        self.explain.config(state="disabled")

    def on_show(self):
        m = self.ctx.metrics
        if not m:
            self.info_lbl.config(text="No trained model metrics found. Train the models from Settings -> Retrain Model.", fg=C["red"])
            fill_tree(self.tree, [])
            return
        best = next(r for r in m["models"] if r["name"] == m["selected_model"])
        thr = self.ctx.detector.threshold if self.ctx.detector else best["threshold"]
        self.info_lbl.config(fg=C["text"], text=(f"Best-performing model: {best['name']}   |   Selected classification threshold: "
                                                 f"{thr:.4f}   |   Trained: {m['trained_at'].replace('T', ' ')}"))
        rows, tags = [], []
        for r in m["models"] + ([m["isolation_forest"]] if m.get("isolation_forest") else []):
            t = r["test"]
            rows.append((r["name"], *(f"{t[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc")),
                         r["strategy"].split(" (")[0]))
            tags.append("ok" if r["name"] == m["selected_model"] else "medium" if "Isolation" in r["name"] else "ok")
        fill_tree(self.tree, rows, tags)
        for it in self.tree.get_children():
            if self.tree.item(it)["values"][0] == m["selected_model"]:
                self.tree.selection_set(it)
                break
        show_canvas(self.tabs["ROC Curve"], curve_fig(m, "roc"))
        show_canvas(self.tabs["Precision-Recall Curve"], curve_fig(m, "pr"))
        self.pick()

    def pick(self):
        m = self.ctx.metrics
        sel = self.tree.selection()
        if not m or not sel:
            return
        name = self.tree.item(sel[0])["values"][0]
        show_canvas(self.tabs["Confusion Matrix"], confusion_matrix_fig(m, name))
        r = next((x for x in m["models"] + ([m["isolation_forest"]] if m.get("isolation_forest") else []) if x["name"] == name), None)
        if not r:
            return
        t = r["test"]
        cm = t["confusion_matrix"]
        cr = t["classification_report"]
        lines = [f"{name}  (test set: {t['n_samples']:,} transactions, {t['n_fraud']:,} fraudulent)",
                 f"Threshold: {t['threshold']:.4f}   FPR: {t['false_positive_rate']:.5f}", "",
                 f"{'':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
        for k in ("Legitimate", "Fraud"):
            lines.append(f"{k:<14}{cr[k]['precision']:>10.4f}{cr[k]['recall']:>10.4f}{cr[k]['f1-score']:>10.4f}{int(cr[k]['support']):>10}")
        lines += ["", f"True negatives : {cm['tn']:,}", f"False positives: {cm['fp']:,}  (legitimate flagged)",
                  f"False negatives: {cm['fn']:,}  (fraud missed)", f"True positives : {cm['tp']:,}"]
        if "note" in r:
            lines += ["", r["note"]]
        tc = m.get("time_based_check")
        if tc and name == m["selected_model"]:
            tt = tc["test"]
            lines += ["", "Time-based (chronological) check of this model:",
                      f"  recall={tt['recall']:.4f} precision={tt['precision']:.4f} PR-AUC={tt['pr_auc']:.4f} "
                      f"(frauds in later test window: {tt['n_fraud']})"]
        if m.get("failures"):
            lines += ["", "Failures during training: " + "; ".join(f"{k}: {v}" for k, v in m["failures"].items())]
        self.report.delete("1.0", "end")
        self.report.insert("1.0", "\n".join(lines))
