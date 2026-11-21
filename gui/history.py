"""Transaction history and fraud review queue."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import COLORS as C, RISK_LEVELS, REPORTS_DIR, REVIEW_STATUSES
from src.database import DatabaseError
from .styles import Card, Page, fill_tree, fmt_score, font, make_tree, risk_tag

HCOLS = ["id", "predicted_at", "amount", "time_feature", "prediction", "fraud_score", "risk_category",
         "model_name", "source", "review_status"]


class HistoryPage(Page):
    title = "Transaction History & Review"
    subtitle = "All predictions saved locally. Review decisions are stored separately from the model prediction."

    def build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self.t_hist = tk.Frame(nb, bg=C["bg"])
        self.t_rev = tk.Frame(nb, bg=C["bg"])
        nb.add(self.t_hist, text="  All Predictions  ")
        nb.add(self.t_rev, text="  Fraud Review Queue  ")
        self._build_history()
        self._build_review()

    # ------------------------------------------------------------ history tab
    def _build_history(self):
        f = Card(self.t_hist)
        f.pack(fill="x", pady=8)
        r = tk.Frame(f.body, bg=C["white"])
        r.pack(fill="x")
        self.q_id, self.q_status, self.q_risk = tk.StringVar(), tk.StringVar(value="All"), tk.StringVar(value="All")
        self.q_date, self.q_sort, self.q_desc = tk.StringVar(), tk.StringVar(value="id"), tk.BooleanVar(value=True)

        def lab(t):
            tk.Label(r, text=t, bg=C["white"], fg=C["muted"], font=font(9)).pack(side="left", padx=(8, 2))
        lab("ID"); ttk.Entry(r, textvariable=self.q_id, width=7).pack(side="left")
        lab("Status"); ttk.Combobox(r, textvariable=self.q_status, width=15, state="readonly",
                                    values=["All", "Legitimate", "Potential Fraud"]).pack(side="left")
        lab("Risk"); ttk.Combobox(r, textvariable=self.q_risk, width=12, state="readonly",
                                  values=["All"] + RISK_LEVELS).pack(side="left")
        lab("Date (YYYY-MM-DD)"); ttk.Entry(r, textvariable=self.q_date, width=11).pack(side="left")
        lab("Sort"); ttk.Combobox(r, textvariable=self.q_sort, width=13, state="readonly",
                                  values=["id", "predicted_at", "amount", "fraud_score", "risk_category"]).pack(side="left")
        ttk.Checkbutton(r, text="newest/highest first", variable=self.q_desc).pack(side="left", padx=8)
        ttk.Button(r, text="Apply", style="Accent.TButton", command=self.load).pack(side="left", padx=4)
        ttk.Button(r, text="Reset", style="Ghost.TButton", command=self.reset).pack(side="left")
        card = Card(self.t_hist)
        card.pack(fill="both", expand=True, pady=(0, 8))
        fr, self.tree = make_tree(card.body, HCOLS, {"predicted_at": 140, "model_name": 130, "review_status": 120}, height=12)
        fr.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self.details())
        b = tk.Frame(self.t_hist, bg=C["bg"])
        b.pack(fill="x", pady=(0, 6))
        self.count = tk.Label(b, bg=C["bg"], fg=C["muted"], font=font(9))
        self.count.pack(side="left")
        ttk.Button(b, text="Export CSV", style="Ghost.TButton", command=self.export).pack(side="right")
        ttk.Button(b, text="Clear all history", style="Danger.TButton", command=self.clear_all).pack(side="right", padx=6)
        ttk.Button(b, text="Delete selected", style="Ghost.TButton", command=self.delete).pack(side="right")
        ttk.Button(b, text="View details", style="Ghost.TButton", command=self.details).pack(side="right", padx=6)

    def reset(self):
        self.q_id.set(""); self.q_status.set("All"); self.q_risk.set("All"); self.q_date.set("")
        self.q_sort.set("id"); self.q_desc.set(True)
        self.load()

    def _filters(self):
        cls = {"Legitimate": 0, "Potential Fraud": 1}.get(self.q_status.get())
        return dict(search_id=self.q_id.get(), predicted_class=cls, risk="" if self.q_risk.get() == "All" else self.q_risk.get(),
                    date=self.q_date.get(), sort_by=self.q_sort.get(), descending=self.q_desc.get())

    def load(self):
        try:
            rows = self.ctx.db.query(**self._filters(), limit=5000)
        except DatabaseError as exc:
            self.error("Filter error", str(exc))
            return
        self._rows = {r["id"]: r for r in rows}
        data, tags = [], []
        for r in rows:
            rev = r["review_status"] or ("-" if r["predicted_class"] == 0 else "Pending Review")
            data.append((r["id"], r["predicted_at"].replace("T", " "), f"{r['amount']:.2f}", f"{r['time_feature']:.0f}",
                         "Potential Fraud" if r["predicted_class"] else "Legitimate", fmt_score(r["fraud_score"]),
                         r["risk_category"], r["model_name"], r["source"], rev))
            tags.append(risk_tag(r["risk_category"], r["predicted_class"]))
        fill_tree(self.tree, data, tags)
        self.count.config(text=f"{len(rows):,} record(s) shown (max 5,000)")

    def _selected_id(self):
        sel = self.tree.selection()
        if not sel:
            self.info("Select a record", "Please select a row first.")
            return None
        return int(self.tree.item(sel[0])["values"][0])

    def details(self):
        pid = self._selected_id()
        if pid is None:
            return
        r = self.ctx.db.get(pid)
        if not r:
            return
        lines = [f"Prediction ID: {r['id']}", f"Predicted at: {r['predicted_at']}", f"Source: {r['source']}",
                 f"Model: {r['model_name']}", f"Amount: {r['amount']:.2f}", f"Time feature: {r['time_feature']}",
                 f"Model prediction: {'Potential Fraud (NOT confirmed)' if r['predicted_class'] else 'Legitimate'}",
                 f"Fraud probability (calibrated estimate): {fmt_score(r['fraud_score'])}", f"Risk category: {r['risk_category']}",
                 f"Review status: {r['review_status'] or 'n/a'}", f"Review note: {r['review_note'] or '-'}",
                 f"Full features stored: {'yes' if r['features_json'] else 'no (privacy default)'}"]
        messagebox.showinfo("Transaction details", "\n".join(lines))

    def delete(self):
        pid = self._selected_id()
        if pid is not None and messagebox.askyesno("Delete record", f"Delete prediction #{pid}? This cannot be undone."):
            try:
                self.ctx.db.delete(pid)
            except DatabaseError as exc:
                self.error("Delete failed", str(exc))
            self.app.refresh_pages()

    def clear_all(self):
        if messagebox.askyesno("Clear history", "Delete ALL stored predictions and reviews?\nThis cannot be undone."):
            try:
                self.ctx.db.clear_all()
            except DatabaseError as exc:
                self.error("Clear failed", str(exc))
            self.app.refresh_pages()

    def export(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialdir=REPORTS_DIR, initialfile="history.csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            try:
                n = self.ctx.db.export_csv(p, self.ctx.db.query(**self._filters(), limit=10_000_000))
                self.info("Exported", f"{n:,} record(s) saved to\n{p}")
            except DatabaseError as exc:
                self.error("Export failed", str(exc))

    # ------------------------------------------------------------ review tab
    def _build_review(self):
        f = Card(self.t_rev, "Flagged transactions awaiting human review")
        f.pack(fill="both", expand=True, pady=8)
        bar = tk.Frame(f.body, bg=C["white"])
        bar.pack(fill="x", pady=(0, 6))
        tk.Label(bar, text="Show status:", bg=C["white"], fg=C["muted"], font=font(9)).pack(side="left")
        self.r_filter = tk.StringVar(value="All")
        cb = ttk.Combobox(bar, textvariable=self.r_filter, state="readonly", width=16, values=["All"] + REVIEW_STATUSES)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_review())
        cols = ["id", "predicted_at", "amount", "fraud_score", "risk_category", "review_status", "review_note"]
        fr, self.rtree = make_tree(f.body, cols, {"predicted_at": 140, "review_note": 260, "review_status": 130}, height=9)
        fr.pack(fill="both", expand=True)
        self.rtree.bind("<<TreeviewSelect>>", self._pick)
        ed = tk.Frame(f.body, bg=C["white"])
        ed.pack(fill="x", pady=(8, 0))
        tk.Label(ed, text="Review status", bg=C["white"], fg=C["muted"], font=font(9)).grid(row=0, column=0, sticky="w")
        self.r_status = tk.StringVar(value=REVIEW_STATUSES[0])
        ttk.Combobox(ed, textvariable=self.r_status, state="readonly", values=REVIEW_STATUSES, width=18).grid(row=1, column=0, padx=(0, 12))
        tk.Label(ed, text="Review note", bg=C["white"], fg=C["muted"], font=font(9)).grid(row=0, column=1, sticky="w")
        self.r_note = tk.StringVar()
        ttk.Entry(ed, textvariable=self.r_note, width=70).grid(row=1, column=1, sticky="ew", padx=(0, 12))
        ttk.Button(ed, text="Save review", style="Accent.TButton", command=self.save_review).grid(row=1, column=2)
        ed.columnconfigure(1, weight=1)
        tk.Label(f.body, bg=C["white"], fg=C["muted"], font=font(8), justify="left", wraplength=1000,
                 text="A model flag is only a prediction. 'Confirmed Fraud' / 'False Positive' are YOUR review labels and "
                      "are stored separately - the original model prediction is never overwritten.").pack(anchor="w", pady=(6, 0))

    def load_review(self):
        status = "" if self.r_filter.get() == "All" else self.r_filter.get()
        rows = self.ctx.db.query(flagged_only=True, review_status=status, limit=5000)
        fill_tree(self.rtree, [(r["id"], r["predicted_at"].replace("T", " "), f"{r['amount']:.2f}", fmt_score(r["fraud_score"]),
                                r["risk_category"], r["review_status"], r["review_note"] or "") for r in rows],
                  ["flag" if r["risk_category"] == "High Risk" else "medium" for r in rows])

    def _pick(self, _):
        sel = self.rtree.selection()
        if sel:
            v = self.rtree.item(sel[0])["values"]
            self.r_status.set(v[5])
            self.r_note.set("" if len(v) < 7 else str(v[6]))

    def save_review(self):
        sel = self.rtree.selection()
        if not sel:
            self.info("Select a transaction", "Select a flagged transaction first.")
            return
        pid = int(self.rtree.item(sel[0])["values"][0])
        try:
            self.ctx.db.set_review(pid, self.r_status.get(), self.r_note.get())
        except DatabaseError as exc:
            self.error("Could not save review", str(exc))
            return
        self.load_review()
        self.load()

    def on_show(self):
        self.load()
        self.load_review()
