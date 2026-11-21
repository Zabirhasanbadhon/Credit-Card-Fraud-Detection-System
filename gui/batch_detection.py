"""Batch CSV detection page."""
import tkinter as tk
from tkinter import filedialog, ttk

import pandas as pd

from src.batch_detection import CSVError, preview_csv, run_batch, validate_csv
from src.config import COLORS as C, REPORTS_DIR
from src.report_generator import ReportError, export_records_csv
from .styles import Card, Page, StatCard, fill_tree, font, make_tree, run_async

MAX_DISPLAY = 2000


class BatchDetectionPage(Page):
    title = "Batch CSV Detection"
    subtitle = ("Upload a CSV with Time, V1-V28 and Amount columns. A 'Class' column is optional and, if present, "
                "is never used for prediction.")

    def build(self):
        self.banner_holder = tk.Frame(self, bg=C["bg"])
        self.banner_holder.pack(fill="x")
        self.path = tk.StringVar()
        self.results: pd.DataFrame | None = None
        top = Card(self, "1. Select and validate file")
        top.pack(fill="x", padx=24, pady=(0, 8))
        row = tk.Frame(top.body, bg=C["white"])
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", style="Ghost.TButton", command=self.browse).pack(side="left", padx=6)
        ttk.Button(row, text="Validate & Preview", style="Ghost.TButton", command=self.validate).pack(side="left")
        self.status = tk.Label(top.body, text="No file selected.", bg=C["white"], fg=C["muted"], anchor="w",
                               justify="left", wraplength=1000, font=font(9))
        self.status.pack(fill="x", pady=(6, 0))
        pframe, self.preview = make_tree(top.body, ["Time", "V1", "V2", "V3", "V4", "Amount", "Class"], height=4)
        pframe.pack(fill="x", pady=(6, 0))

        run = Card(self, "2. Detect fraud")
        run.pack(fill="x", padx=24, pady=8)
        r2 = tk.Frame(run.body, bg=C["white"])
        r2.pack(fill="x")
        self.run_btn = ttk.Button(r2, text="Run Detection", style="Accent.TButton", command=self.run)
        self.run_btn.pack(side="left")
        self.save_hist = tk.BooleanVar(value=True)
        self.flag_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="Save to history", variable=self.save_hist).pack(side="left", padx=(16, 4))
        ttk.Checkbutton(r2, text="flagged transactions only", variable=self.flag_only).pack(side="left")
        ttk.Button(r2, text="Export results to CSV", style="Ghost.TButton", command=self.export).pack(side="right")
        self.bar = ttk.Progressbar(run.body, mode="indeterminate")
        self.msg = tk.Label(run.body, text="", bg=C["white"], fg=C["muted"], anchor="w", font=font(9))
        self.msg.pack(fill="x", pady=(6, 0))

        cards = tk.Frame(self, bg=C["bg"])
        cards.pack(fill="x", padx=24)
        self.stat = [StatCard(cards, t, c) for t, c in (("Transactions scored", C["blue"]), ("Legitimate", C["green"]),
                                                       ("Flagged", C["red"]), ("Flagged %", C["orange"]))]
        for i, s in enumerate(self.stat):
            s.grid(row=0, column=i, sticky="nsew", padx=6, pady=6)
            cards.columnconfigure(i, weight=1, uniform="b")
        res = Card(self, "Results (flagged transactions first; model predictions, not confirmed fraud)")
        res.pack(fill="both", expand=True, padx=24, pady=(8, 16))
        cols = ["row_number", "Time", "Amount", "prediction_label", "fraud_probability", "risk_category", "actual_class"]
        f, self.tree = make_tree(res.body, cols, {"prediction_label": 130, "risk_category": 110}, height=9)
        f.pack(fill="both", expand=True)

    def on_show(self):
        for w in self.banner_holder.winfo_children():
            w.destroy()
        self.model_banner(self.banner_holder)

    def browse(self):
        p = filedialog.askopenfilename(title="Select transactions CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.path.set(p)
            self.validate()

    def validate(self):
        if not self.path.get().strip():
            self.error("No file", "Please choose a CSV file first.")
            return False
        try:
            info = validate_csv(self.path.get())
            if not info["ok"]:
                self.status.config(fg=C["red"], text="Missing required columns: " + ", ".join(info["missing"]))
                fill_tree(self.preview, [])
                return False
            pv = preview_csv(self.path.get(), 5)
        except CSVError as exc:
            self.status.config(fg=C["red"], text=str(exc))
            return False
        note = " 'Class' column found: it will be kept separate and NOT used for prediction." if info["has_target"] else ""
        self.status.config(fg=C["green"], text=f"File looks valid ({len(info['columns'])} columns).{note}")
        fill_tree(self.preview, [tuple(r.get(c, "") for c in ["Time", "V1", "V2", "V3", "V4", "Amount", "Class"])
                                 for r in pv.to_dict("records")])
        return True

    def run(self):
        if self.ctx.detector is None:
            self.error("Model not available", self.ctx.model_error or "Train the model first.")
            return
        if not self.validate():
            return
        self.run_btn.state(["disabled"])
        self.bar.pack(fill="x", pady=(6, 0))
        self.bar.start(12)
        det, path = self.ctx.detector, self.path.get()
        run_async(self, lambda emit: run_batch(path, det, progress=lambda n: emit(f"{n:,} rows processed ...")),
                  self._done, self._failed, on_msg=lambda m: self.msg.config(text=m, fg=C["muted"]))

    def _finish_ui(self):
        self.bar.stop()
        self.bar.pack_forget()
        self.run_btn.state(["!disabled"])

    def _failed(self, exc):
        self._finish_ui()
        self.msg.config(text="Detection failed.", fg=C["red"])
        self.error("Batch detection failed", str(exc))

    def _done(self, out):
        self._finish_ui()
        df, s = out
        self.results = df
        self.stat[0].set(f"{s['total_scored']:,}")
        self.stat[1].set(f"{s['legitimate']:,}")
        self.stat[2].set(f"{s['flagged']:,}")
        self.stat[3].set(f"{s['flagged_percentage']}%")
        skipped = f" {s['skipped_invalid']:,} invalid row(s) skipped (e.g. rows {s['skipped_rows_sample']})." if s["skipped_invalid"] else ""
        saved = ""
        if self.save_hist.get() and len(df):
            try:
                sub = df[df["predicted_class"] == 1] if self.flag_only.get() else df
                recs = [{"amount": r.Amount, "time_feature": r.Time, "predicted_class": int(r.predicted_class),
                         "fraud_score": r.fraud_probability, "risk_category": r.risk_category,
                         "model_name": self.ctx.detector.model_name, "source": "csv"} for r in sub.itertuples()]
                saved = f" {self.ctx.db.add_predictions(recs):,} record(s) saved to history."
            except Exception as exc:  # noqa: BLE001
                self.error("Could not save history", str(exc))
        self.msg.config(fg=C["text"], text="Done." + skipped + saved)
        if not len(df):
            fill_tree(self.tree, [])
            return
        show = pd.concat([df[df["predicted_class"] == 1].sort_values("fraud_probability", ascending=False),
                          df[df["predicted_class"] == 0]]).head(MAX_DISPLAY)
        rows, tags = [], []
        for r in show.to_dict("records"):
            act = r.get("actual_class")
            rows.append((r["row_number"], f"{r['Time']:.0f}", f"{r['Amount']:.2f}", r["prediction_label"],
                         f"{r['fraud_probability']:.4f}", r["risk_category"], "" if act is None or pd.isna(act) else int(act)))
            tags.append("flag" if r["predicted_class"] == 1 else ("medium" if r["risk_category"] == "Medium Risk" else "ok"))
        fill_tree(self.tree, rows, tags)
        if len(df) > MAX_DISPLAY:
            self.msg.config(text=self.msg.cget("text") + f" Showing first {MAX_DISPLAY:,} rows; export for all.")
        self.app.refresh_pages()

    def export(self):
        if self.results is None or self.results.empty:
            self.error("Nothing to export", "Run a detection first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialdir=REPORTS_DIR, initialfile="batch_results.csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            try:
                self.results.to_csv(p, index=False)
                self.info("Exported", f"{len(self.results):,} rows saved to\n{p}")
            except OSError as exc:
                self.error("Export failed", str(exc))
