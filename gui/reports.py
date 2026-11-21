"""Report generation page (CSV / PDF)."""
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, ttk

from src.config import COLORS as C, REPORTS_DIR
from src.database import DatabaseError
from src.report_generator import ReportError, export_model_csv, export_pdf, export_records_csv
from .styles import Card, Page, font

KINDS = {"Full transaction analysis": "full", "Flagged transaction records only": "flagged", "Model performance report": "model"}


class ReportsPage(Page):
    title = "Reports"
    subtitle = "Export analysis results as CSV or PDF. Reports are generated locally."

    def build(self):
        card = Card(self, "Create a report")
        card.pack(fill="x", padx=24, pady=8)
        self.kind = tk.StringVar(value=list(KINDS)[0])
        self.fmt = tk.StringVar(value="PDF")
        tk.Label(card.body, text="Report type", bg=C["white"], fg=C["muted"], font=font(9)).pack(anchor="w")
        for k in KINDS:
            ttk.Radiobutton(card.body, text=k, value=k, variable=self.kind).pack(anchor="w", pady=1)
        tk.Label(card.body, text="Format", bg=C["white"], fg=C["muted"], font=font(9)).pack(anchor="w", pady=(10, 0))
        row = tk.Frame(card.body, bg=C["white"])
        row.pack(anchor="w")
        for f in ("PDF", "CSV"):
            ttk.Radiobutton(row, text=f, value=f, variable=self.fmt).pack(side="left", padx=(0, 12))
        ttk.Button(card.body, text="Generate report", style="Accent.TButton", command=self.generate).pack(anchor="w", pady=(14, 0))
        self.msg = tk.Label(card.body, bg=C["white"], fg=C["muted"], font=font(9), wraplength=900, justify="left")
        self.msg.pack(anchor="w", pady=(8, 0))
        tk.Label(self, bg=C["bg"], fg=C["muted"], font=font(9), justify="left", wraplength=1000, text=(
            "PDF reports include: application name, generation date, counts and flagged percentage, selected model, evaluation "
            "metrics, flagged records, a risk summary and a limitations disclaimer. Flagged records are model predictions, "
            "not confirmed fraud.")).pack(anchor="w", padx=24)

    def generate(self):
        kind, fmt = KINDS[self.kind.get()], self.fmt.get()
        stats, metrics = self.ctx.db.stats(), self.ctx.metrics
        if kind != "model" and stats["total"] == 0:
            self.error("No data", "There are no analysed transactions yet. Run a detection first.")
            return
        if kind == "model" and not metrics:
            self.error("No model metrics", "Train the models first (Settings -> Retrain Model).")
            return
        ext = ".pdf" if fmt == "PDF" else ".csv"
        path = filedialog.asksaveasfilename(defaultextension=ext, initialdir=REPORTS_DIR,
                                            initialfile=f"fraud_{kind}_{datetime.now():%Y%m%d_%H%M}{ext}",
                                            filetypes=[(fmt, f"*{ext}")])
        if not path:
            return
        try:
            rows = [] if kind == "model" else self.ctx.db.query(flagged_only=(kind == "flagged"), limit=10_000_000)
            if fmt == "PDF":
                export_pdf(path, kind, rows, stats, metrics)
            elif kind == "model":
                export_model_csv(metrics, path)
            else:
                export_records_csv(rows, path)
        except (ReportError, DatabaseError) as exc:
            self.error("Report failed", str(exc))
            return
        self.msg.config(fg=C["green"], text=f"Report saved: {path}")
        self.info("Report created", f"Saved to\n{path}")
