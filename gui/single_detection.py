"""Single transaction detection page (Time, Amount, V1-V28)."""
import tkinter as tk
from datetime import datetime
from tkinter import ttk

from src.config import COLORS as C, FEATURES, V_FEATURES
from src.data_loader import DatasetError, sample_real_rows
from src.validation import parse_single_input
from .styles import Card, Page, ScrollFrame, font

DISCLAIMER = ("This is an automated machine-learning risk assessment, not a confirmed financial investigation. "
              "Flagged transactions should be reviewed by a person.")


class Collapsible(tk.Frame):
    def __init__(self, parent, title, opened=False):
        super().__init__(parent, bg=C["white"])
        self.open = opened
        self.btn = tk.Button(self, text="", anchor="w", relief="flat", bg=C["bg"], fg=C["navy"],
                             font=font(10, "bold"), cursor="hand2", command=self.toggle, padx=10, pady=6)
        self.btn.pack(fill="x")
        self.title = title
        self.body = tk.Frame(self, bg=C["white"])
        self._render()

    def toggle(self):
        self.open = not self.open
        self._render()

    def _render(self):
        self.btn.config(text=("▼  " if self.open else "▶  ") + self.title)
        if self.open:
            self.body.pack(fill="x", pady=6)
        else:
            self.body.pack_forget()


class SingleDetectionPage(Page):
    title = "Single Transaction Detection"
    subtitle = "Enter one transaction's Time, Amount and the 28 anonymised PCA features, then run the model."

    def build(self):
        self.banner_holder = tk.Frame(self, bg=C["bg"])
        self.banner_holder.pack(fill="x")
        self.scroll = ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        body = self.scroll.inner
        self.vars = {n: tk.StringVar() for n in FEATURES}

        note = Card(body, "About the input fields")
        note.pack(fill="x", padx=24, pady=(0, 8))
        tk.Label(note.body, justify="left", wraplength=1000, bg=C["white"], fg=C["text"], font=font(9), text=(
            "Time = seconds since the first transaction in the dataset. Amount = transaction amount.\n"
            "V1-V28 are anonymised PCA components created by the dataset authors to protect privacy. They are NOT "
            "ordinary financial details and cannot be interpreted individually - enter values exactly as they appear "
            "in the dataset. Values are never modified; entries outside the training range only trigger a warning.\n"
            "Never enter card numbers, CVV codes, PINs or passwords.")).pack(anchor="w")

        core = Card(body, "Core fields")
        core.pack(fill="x", padx=24, pady=8)
        self._fields(core.body, ["Time", "Amount"], cols=2)

        groups = Card(body, "Anonymised features")
        groups.pack(fill="x", padx=24, pady=8)
        for title, names, opened in (("V1 - V10", V_FEATURES[:10], True), ("V11 - V20", V_FEATURES[10:20], False),
                                     ("V21 - V28", V_FEATURES[20:], False)):
            g = Collapsible(groups.body, title, opened)
            g.pack(fill="x", pady=2)
            self._fields(g.body, names, cols=5)

        bar = tk.Frame(body, bg=C["bg"])
        bar.pack(fill="x", padx=24, pady=8)
        ttk.Button(bar, text="Detect Fraud", style="Accent.TButton", command=self.detect).pack(side="left")
        ttk.Button(bar, text="Clear Form", style="Ghost.TButton", command=self.clear).pack(side="left", padx=8)
        ttk.Button(bar, text="Load real legitimate example", style="Ghost.TButton",
                   command=lambda: self.load_example(0)).pack(side="left", padx=(24, 4))
        ttk.Button(bar, text="Load real fraud-labelled example", style="Ghost.TButton",
                   command=lambda: self.load_example(1)).pack(side="left")
        self.result_card = Card(body, "Prediction result")
        self.result_card.pack(fill="x", padx=24, pady=(8, 24))
        tk.Label(self.result_card.body, text="No prediction yet.", bg=C["white"], fg=C["muted"]).pack(anchor="w")

    def _fields(self, parent, names, cols):
        for i, n in enumerate(names):
            r, c = divmod(i, cols)
            cell = tk.Frame(parent, bg=C["white"])
            cell.grid(row=r, column=c, padx=8, pady=4, sticky="w")
            tk.Label(cell, text=n, width=7, anchor="w", bg=C["white"], fg=C["muted"], font=font(9, "bold")).pack(side="left")
            ttk.Entry(cell, textvariable=self.vars[n], width=14).pack(side="left")

    def on_show(self):
        for w in self.banner_holder.winfo_children():
            w.destroy()
        self.model_banner(self.banner_holder)

    def clear(self):
        for v in self.vars.values():
            v.set("")

    def load_example(self, label):
        try:
            row = sample_real_rows(label).iloc[0]
        except DatasetError as exc:
            self.error("Dataset not available", str(exc))
            return
        for n in FEATURES:
            self.vars[n].set(repr(float(row[n])))

    def detect(self):
        det = self.ctx.detector
        if det is None:
            self.error("Model not available", self.ctx.model_error or "Train the model first.")
            return
        values, errors, warnings = parse_single_input({n: v.get() for n, v in self.vars.items()}, det.feature_ranges)
        if errors:
            self.error("Please fix the input", "\n".join(errors[:12]) + (f"\n... and {len(errors) - 12} more" if len(errors) > 12 else ""))
            return
        try:
            res = det.predict_single(values)
            self.ctx.db.add_predictions([self.ctx.history_record(res, "manual", values)])
        except Exception as exc:  # noqa: BLE001 - never crash the GUI
            self.error("Prediction failed", str(exc))
            return
        self.show_result(res, warnings)

    def show_result(self, res, warnings):
        for w in self.result_card.body.winfo_children():
            w.destroy()
        fraud = res["predicted_class"] == 1
        fg, bg = (C["red"], C["red_bg"]) if fraud else (C["green"], C["green_bg"])
        text = "⚠  Potential Fraud Detected - flagged for review" if fraud else "✔  No Fraud Detected"
        tk.Label(self.result_card.body, text=text, font=font(15, "bold"), fg=fg, bg=bg, padx=14, pady=10,
                 anchor="w").pack(fill="x")
        risk_color = {"High Risk": C["red"], "Medium Risk": C["orange"], "Low Risk": C["green"]}[res["risk_category"]]
        grid = tk.Frame(self.result_card.body, bg=C["white"])
        grid.pack(fill="x", pady=8)
        items = [("Classification", res["prediction_label"]),
                 ("Fraud probability (calibrated estimate)", f"{res['fraud_probability']:.4f}"),
                 ("Risk category", res["risk_category"]), ("Model", res["model_name"]),
                 ("Transaction amount", f"{res['amount']:,.2f}"), ("Time feature (s)", f"{res['time_feature']:,.0f}"),
                 ("Decision threshold", f"{res['threshold']:.4f}"),
                 ("Prediction time", res["predicted_at"].replace("T", " "))]
        for i, (k, v) in enumerate(items):
            tk.Label(grid, text=k, bg=C["white"], fg=C["muted"], font=font(9), anchor="w").grid(row=i, column=0, sticky="w", padx=(0, 20), pady=2)
            tk.Label(grid, text=v, bg=C["white"], fg=risk_color if k == "Risk category" else C["text"],
                     font=font(10, "bold"), anchor="w").grid(row=i, column=1, sticky="w")
        if res["risk_category"] == "Medium Risk" and not fraud:
            tk.Label(self.result_card.body, bg=C["orange_bg"], fg="#8A4B08", anchor="w", font=font(9),
                     text="Score is in the Medium Risk band but below the flag threshold - not flagged.", padx=8, pady=4).pack(fill="x", pady=2)
        for w in warnings[:6]:
            tk.Label(self.result_card.body, text="⚠ " + w, bg=C["orange_bg"], fg="#8A4B08", anchor="w", wraplength=950,
                     justify="left", font=font(9), padx=8, pady=3).pack(fill="x", pady=1)
        tk.Label(self.result_card.body, text=DISCLAIMER + " The result was saved to Transaction History.",
                 bg=C["white"], fg=C["muted"], font=font(8), wraplength=950, justify="left").pack(anchor="w", pady=(8, 0))
        self.scroll.canvas.update_idletasks()
        self.scroll.canvas.yview_moveto(1.0)
