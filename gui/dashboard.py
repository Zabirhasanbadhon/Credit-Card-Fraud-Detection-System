"""Dashboard page: KPIs from real prediction history + trained-model metrics."""
import tkinter as tk

from src.config import COLORS as C
from .styles import Card, Page, StatCard, fill_tree, fmt_score, font, make_tree


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Overview of analysed transactions and the active fraud-detection model."

    def build(self):
        self.banner_holder = tk.Frame(self, bg=C["bg"])
        self.banner_holder.pack(fill="x")
        grid = tk.Frame(self, bg=C["bg"])
        grid.pack(fill="x", padx=24)
        spec = [("Transactions analysed", C["blue"]), ("Classified legitimate", C["green"]),
                ("Flagged as suspicious", C["red"]), ("Flagged rate", C["orange"]),
                ("Best model", C["navy"]), ("Model recall (test)", C["blue"]),
                ("Model precision (test)", C["blue"]), ("Dataset fraud rate", C["muted"])]
        self.cards = []
        for i, (t, col) in enumerate(spec):
            c = StatCard(grid, t, col)
            c.grid(row=i // 4, column=i % 4, sticky="nsew", padx=6, pady=6)
            self.cards.append(c)
        for j in range(4):
            grid.columnconfigure(j, weight=1, uniform="k")
        card = Card(self, "Recent flagged transactions (model predictions - not confirmed fraud)")
        card.pack(fill="both", expand=True, padx=24, pady=(8, 16))
        cols = ["id", "predicted_at", "amount", "fraud_probability", "risk_category", "review_status", "source"]
        frame, self.tree = make_tree(card.body, cols, {"predicted_at": 150, "review_status": 130}, height=8)
        frame.pack(fill="both", expand=True)

    def on_show(self):
        for w in self.banner_holder.winfo_children():
            w.destroy()
        self.model_banner(self.banner_holder)
        s, m = self.ctx.db.stats(), self.ctx.metrics
        self.cards[0].set(f"{s['total']:,}")
        self.cards[1].set(f"{s['legitimate']:,}")
        self.cards[2].set(f"{s['flagged']:,}")
        self.cards[3].set(f"{s['flagged_pct']}%")
        if m:
            best = next(r for r in m["models"] if r["name"] == m["selected_model"])
            self.cards[4].set(best["name"])
            self.cards[5].set(f"{best['test']['recall']:.3f}")
            self.cards[6].set(f"{best['test']['precision']:.3f}")
            self.cards[7].set(f"{m['dataset']['fraud_percentage']}%")
        else:
            for c in self.cards[4:]:
                c.set("n/a")
        rows = self.ctx.db.query(flagged_only=True, limit=15)
        fill_tree(self.tree, [(r["id"], r["predicted_at"], f"{r['amount']:.2f}", fmt_score(r["fraud_score"]),
                               r["risk_category"], r["review_status"], r["source"]) for r in rows],
                  ["flag"] * len(rows))
