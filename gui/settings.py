"""Settings and About pages."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import (APP_NAME, APP_VERSION, COLORS as C, DATASET_PATH, DB_PATH, MODEL_PATH, REPORTS_DIR)
from src.data_loader import DatasetError, download_from_openml
from src.database import DatabaseError
from src.train_model import train_all
from .styles import Card, Page, ScrollFrame, font, run_async


class SettingsPage(Page):
    title = "Settings"
    subtitle = "Model, threshold, privacy and data management."

    def build(self):
        self.scroll = ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        body = self.scroll.inner
        self.s_thr = tk.StringVar()
        self.s_ratio = tk.StringVar()
        self.s_beta = tk.StringVar()
        self.s_store = tk.BooleanVar()

        self.info_card = Card(body, "System information")
        self.info_card.pack(fill="x", padx=24, pady=8)
        self.info_lbl = tk.Label(self.info_card.body, bg=C["white"], fg=C["text"], font=font(9), justify="left", anchor="w", wraplength=1000)
        self.info_lbl.pack(fill="x")

        th = Card(body, "Prediction threshold & risk bands")
        th.pack(fill="x", padx=24, pady=8)
        g = tk.Frame(th.body, bg=C["white"])
        g.pack(fill="x")
        rows = [("Decision threshold override (0-1, blank = use trained threshold)", self.s_thr),
                ("Medium-risk band starts at this fraction of the threshold (0.05-0.95)", self.s_ratio),
                ("F-beta used to pick the threshold at the NEXT training (0.1-5; >1 favours recall)", self.s_beta)]
        for i, (t, v) in enumerate(rows):
            tk.Label(g, text=t, bg=C["white"], font=font(9)).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(g, textvariable=v, width=10).grid(row=i, column=1, padx=10)
        tk.Label(th.body, bg=C["white"], fg=C["muted"], font=font(8), wraplength=1000, justify="left", text=(
            "Risk bands are configurable score bands relative to the threshold; they are not independently validated fraud "
            "probabilities. Changing the threshold trades recall against false positives.")).pack(anchor="w", pady=(6, 0))

        pr = Card(body, "Privacy")
        pr.pack(fill="x", padx=24, pady=8)
        ttk.Checkbutton(pr.body, variable=self.s_store, text="Store all 28 anonymised feature values with manual predictions "
                        "(off by default for privacy and storage efficiency)").pack(anchor="w")
        tk.Label(pr.body, bg=C["white"], fg=C["muted"], font=font(8), wraplength=1000, justify="left", text=(
            "All data stays on this computer. The app never asks for card numbers, CVVs, PINs or passwords, never contacts banks "
            "and never blocks transactions. Use 'Clear history' below to delete everything stored.")).pack(anchor="w", pady=(4, 0))
        ttk.Button(pr.body, text="Save settings", style="Accent.TButton", command=self.save).pack(anchor="w", pady=(10, 0))

        dm = Card(body, "Data management")
        dm.pack(fill="x", padx=24, pady=8)
        r = tk.Frame(dm.body, bg=C["white"])
        r.pack(fill="x")
        self.train_btn = ttk.Button(r, text="Retrain model", style="Accent.TButton", command=self.retrain)
        self.train_btn.pack(side="left")
        self.dl_btn = ttk.Button(r, text="Download dataset from OpenML", style="Ghost.TButton", command=self.download)
        self.dl_btn.pack(side="left", padx=8)
        ttk.Button(r, text="Export history (CSV)", style="Ghost.TButton", command=self.export).pack(side="left")
        ttk.Button(r, text="Clear history", style="Danger.TButton", command=self.clear).pack(side="left", padx=8)
        self.log = tk.Text(dm.body, height=10, font=("Consolas", 9), bg="#0F1B2D", fg="#D6E4FF", relief="flat")
        self.log.pack(fill="x", pady=(10, 0))
        self.log.insert("1.0", "Training log will appear here.\n")
        self.log.config(state="disabled")

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def on_show(self):
        s, m, det = self.ctx.settings, self.ctx.metrics, self.ctx.detector
        ov = s.get("threshold_override")
        self.s_thr.set("" if ov in (None, "") else str(ov))
        self.s_ratio.set(str(s["risk_medium_ratio"]))
        self.s_beta.set(str(s["threshold_beta"]))
        self.s_store.set(bool(s["store_full_features"]))
        n_rec = self.ctx.db.stats()["total"]
        ds = (f"{m['dataset']['rows']:,} rows, {m['dataset']['fraudulent']:,} fraud ({m['dataset']['fraud_percentage']}%)" if m else "not analysed yet")
        self.info_lbl.config(text="\n".join([
            f"Application: {APP_NAME}  v{APP_VERSION}",
            f"Dataset: {DATASET_PATH}  [{'found' if DATASET_PATH.exists() else 'NOT FOUND'}]  - {ds}",
            f"Selected model: {det.model_name if det else 'none'}   |   trained threshold: {det.trained_threshold:.4f}" if det else "Selected model: none (train first)",
            f"Active threshold: {det.threshold:.4f}   |   model file: {MODEL_PATH.name} [{'found' if MODEL_PATH.exists() else 'missing'}]" if det else f"Model file: {MODEL_PATH} [missing]",
            f"Database: {DB_PATH}  ({n_rec:,} stored predictions)"]))

    def save(self):
        try:
            thr = self.s_thr.get().strip()
            thr = None if thr == "" else float(thr)
            ratio, beta = float(self.s_ratio.get()), float(self.s_beta.get())
        except ValueError:
            self.error("Invalid value", "Threshold, ratio and beta must be numbers.")
            return
        if thr is not None and not 0.0 < thr <= 1.0:
            self.error("Invalid threshold", "Threshold must be greater than 0 and at most 1 (or blank).")
            return
        if not 0.05 <= ratio <= 0.95 or not 0.1 <= beta <= 5:
            self.error("Invalid value", "Ratio must be 0.05-0.95 and beta 0.1-5.")
            return
        self.ctx.update_settings(threshold_override=thr, risk_medium_ratio=ratio, threshold_beta=beta,
                                 store_full_features=bool(self.s_store.get()))
        self.on_show()
        self.info("Saved", "Settings saved.")

    def retrain(self):
        if not DATASET_PATH.exists():
            from src.data_loader import MANUAL_INSTRUCTIONS
            self.error("Dataset missing", MANUAL_INSTRUCTIONS)
            return
        if not messagebox.askyesno("Retrain model", "Retraining trains 7 models on the full dataset and can take several minutes "
                                   "(Support Vector Machine, Gradient Boosting and Random Forest are the slowest).\n\n"
                                   "The existing model will be replaced. Continue?"):
            return
        self.train_btn.state(["disabled"])
        self._log("--- Training started ---")
        beta = float(self.ctx.settings.get("threshold_beta", 1.0))
        run_async(self, lambda emit: train_all(progress=emit, beta=beta), self._trained, self._train_failed,
                  on_msg=self._log)

    def _trained(self, metrics):
        self.train_btn.state(["!disabled"])
        self.ctx.reload_model()
        self.app.refresh_pages()
        self.info("Training complete", f"Selected model: {metrics['selected_model']}")

    def _train_failed(self, exc):
        self.train_btn.state(["!disabled"])
        self._log(f"ERROR: {exc}")
        self.error("Training failed", str(exc))

    def download(self):
        if not messagebox.askyesno("Download dataset", "This downloads the public credit-card fraud dataset from OpenML "
                                   "(internet needed, ~150 MB). No transaction data from this app is sent anywhere.\nContinue?"):
            return
        self.dl_btn.state(["disabled"])
        self._log("Downloading dataset from OpenML ...")
        run_async(self, lambda emit: download_from_openml(), lambda p: (self.dl_btn.state(["!disabled"]), self._log(f"Saved to {p}"), self.on_show()),
                  lambda e: (self.dl_btn.state(["!disabled"]), self._log("Download failed."), self.error("Download failed", str(e))))

    def export(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialdir=REPORTS_DIR, initialfile="history.csv", filetypes=[("CSV", "*.csv")])
        if p:
            try:
                self.info("Exported", f"{self.ctx.db.export_csv(p):,} record(s) saved.")
            except DatabaseError as exc:
                self.error("Export failed", str(exc))

    def clear(self):
        if messagebox.askyesno("Clear history", "Delete ALL stored predictions and reviews? This cannot be undone."):
            try:
                self.ctx.db.clear_all()
            except DatabaseError as exc:
                self.error("Clear failed", str(exc))
            self.app.refresh_pages()


class AboutPage(Page):
    title = "About"
    subtitle = f"{APP_NAME} - version {APP_VERSION}"

    def build(self):
        c = Card(self)
        c.pack(fill="x", padx=24, pady=8)
        tk.Label(c.body, bg=C["white"], fg=C["text"], font=font(10), justify="left", anchor="w", wraplength=1000, text=(
            "A desktop fraud-screening and decision-support tool built with Python, Tkinter, scikit-learn, SQLite, Matplotlib "
            "and ReportLab.\n\n"
            "DATASET: ULB Machine Learning Group, 'Credit Card Fraud Detection' (Kaggle, European cardholders, September 2013; "
            "284,807 transactions, 492 frauds). Features V1-V28 are anonymised PCA components.\n\n"
            "IMPORTANT: Predictions are automated risk assessments. A flagged transaction is NOT confirmed fraud and an unflagged "
            "one is NOT guaranteed legitimate. The tool never blocks transactions, contacts financial institutions or transmits "
            "data. Use it for screening and learning, not as the sole basis for financial decisions.\n\n"
            "Dal Pozzolo, Caelen, Johnson, Bontempi. 'Calibrating Probability with Undersampling for Unbalanced "
            "Classification.' IEEE CIDM, 2015.")).pack(anchor="w")
