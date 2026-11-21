"""Fraud analytics: the 11 required Matplotlib charts."""
import tkinter as tk
from tkinter import ttk

from src.analytics import CHARTS
from src.config import COLORS as C
from .styles import Card, Page, font, show_canvas


class AnalyticsPage(Page):
    title = "Fraud Analytics"
    subtitle = ("Dataset charts use the ACTUAL labels of the training dataset; history charts show MODEL PREDICTIONS "
                "from your analysed transactions.")

    def build(self):
        bar = tk.Frame(self, bg=C["bg"])
        bar.pack(fill="x", padx=24, pady=(0, 6))
        tk.Label(bar, text="Chart:", bg=C["bg"], fg=C["muted"], font=font(10)).pack(side="left")
        self.choice = tk.StringVar(value=list(CHARTS)[0])
        cb = ttk.Combobox(bar, textvariable=self.choice, values=list(CHARTS), state="readonly", width=48)
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw())
        ttk.Button(bar, text="Refresh", style="Ghost.TButton", command=self.draw).pack(side="left")
        card = Card(self)
        card.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self.area = card.body

    def draw(self):
        try:
            fig = CHARTS[self.choice.get()](self.ctx.metrics, self.ctx.db.stats())
            show_canvas(self.area, fig)
        except Exception as exc:  # noqa: BLE001
            self.error("Chart error", str(exc))

    def on_show(self):
        self.draw()
