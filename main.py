"""Credit Card Fraud Detection System - application entry point."""
import logging
import sys
import tkinter as tk
from tkinter import messagebox

from src.config import APP_NAME, APP_VERSION, ASSETS_DIR, COLORS as C, setup_logging

log = logging.getLogger("main")

PAGES = [("Dashboard", "gui.dashboard", "DashboardPage"),
         ("Single Transaction Detection", "gui.single_detection", "SingleDetectionPage"),
         ("Batch CSV Detection", "gui.batch_detection", "BatchDetectionPage"),
         ("Transaction History", "gui.history", "HistoryPage"),
         ("Fraud Analytics", "gui.analytics_page", "AnalyticsPage"),
         ("Model Performance", "gui.model_performance", "ModelPerformancePage"),
         ("Reports", "gui.reports", "ReportsPage"),
         ("Settings", "gui.settings", "SettingsPage"),
         ("About", "gui.settings", "AboutPage")]


class App(tk.Tk):
    def __init__(self, ctx):
        super().__init__()
        from gui.styles import apply_styles, font
        self.ctx = ctx
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1320x820")
        self.minsize(1050, 680)
        apply_styles(self)
        self._logo = None
        icon = ASSETS_DIR / "icon.png"
        try:
            from PIL import Image, ImageTk
            img = Image.open(icon)
            self.iconphoto(True, ImageTk.PhotoImage(img.resize((64, 64))))
            self._logo = ImageTk.PhotoImage(img.resize((44, 44)))
        except Exception as exc:  # noqa: BLE001 - the icon is optional
            log.warning("Icon not loaded: %s", exc)

        side = tk.Frame(self, bg=C["navy"], width=250)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        brand = tk.Frame(side, bg=C["navy"])
        brand.pack(fill="x", padx=16, pady=(20, 18))
        if self._logo:
            tk.Label(brand, image=self._logo, bg=C["navy"]).pack(side="left")
        tk.Label(brand, text="ZaBir\nFraudGuard", justify="left", font=font(12, "bold"), fg="white", bg=C["navy"]).pack(side="left", padx=10)
        self.nav = {}
        for name, _, _ in PAGES:
            b = tk.Label(side, text="   " + name, anchor="w", font=font(10), fg="#C9D6EA", bg=C["navy"], cursor="hand2", pady=10)
            b.pack(fill="x")
            b.bind("<Button-1>", lambda e, n=name: self.show(n))
            b.bind("<Enter>", lambda e, w=b: w.cget("bg") != C["blue"] and w.config(bg=C["navy_hover"]))
            b.bind("<Leave>", lambda e, w=b: w.cget("bg") != C["blue"] and w.config(bg=C["navy"]))
            self.nav[name] = b
        tk.Label(side, text="Decision-support tool only.\nFlags are not confirmed fraud.", fg="#8DA2C0", bg=C["navy"],
                 font=font(8), justify="left").pack(side="bottom", anchor="w", padx=16, pady=14)

        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)
        self.status = tk.Label(right, anchor="w", bg=C["navy_light"], fg="white", font=font(9), padx=12, pady=4)
        self.status.pack(side="bottom", fill="x")
        self.content = tk.Frame(right, bg=C["bg"])
        self.content.pack(fill="both", expand=True)
        self.pages, self.current = {}, None
        self.show("Dashboard")

    def report_callback_exception(self, exc, val, tb):
        """Never let a GUI callback crash the whole application."""
        log.error("Unhandled GUI error", exc_info=(exc, val, tb))
        messagebox.showerror("Unexpected error", f"{val}\n\nDetails were written to logs/app.log")

    def _page(self, name):
        if name not in self.pages:
            import importlib
            _, mod, cls = next(p for p in PAGES if p[0] == name)
            self.pages[name] = getattr(importlib.import_module(mod), cls)(self.content, self.ctx, self)
        return self.pages[name]

    def show(self, name):
        if self.current:
            self.pages[self.current].pack_forget()
            self.nav[self.current].config(bg=C["navy"], fg="#C9D6EA")
        page = self._page(name)
        page.pack(fill="both", expand=True)
        page.on_show()
        self.nav[name].config(bg=C["blue"], fg="white")
        self.current = name
        self.update_status()

    def refresh_pages(self):
        self.update_status()
        if self.current:
            self.pages[self.current].on_show()

    def update_status(self):
        d = self.ctx.detector
        self.status.config(text=(f"Model: {d.model_name}  |  threshold {d.threshold:.4f}  |  trained {d.bundle['trained_at']}"
                                 if d else "No trained model - open Settings to train (needs dataset/creditcard.csv)"))


def main():
    setup_logging()
    log.info("Starting %s v%s", APP_NAME, APP_VERSION)
    from src.app_context import AppContext
    from src.database import DatabaseError
    try:
        ctx = AppContext()
    except DatabaseError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Database error", str(exc))
        return 1
    App(ctx).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
