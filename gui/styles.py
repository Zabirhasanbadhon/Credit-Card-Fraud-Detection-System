"""Theme, shared widgets and helpers for the Tkinter GUI."""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.config import COLORS as C

FONT = "Segoe UI"


def font(size: int = 10, weight: str = "normal"):
    return (FONT, size, weight)


def apply_styles(root: tk.Misc) -> None:
    s = ttk.Style(root)
    s.theme_use("clam")
    root.configure(bg=C["bg"])
    s.configure("Treeview", background=C["white"], fieldbackground=C["white"], foreground=C["text"],
                rowheight=26, font=font(9), borderwidth=0)
    s.configure("Treeview.Heading", background=C["navy"], foreground="white", font=font(9, "bold"), relief="flat")
    s.map("Treeview.Heading", background=[("active", C["navy_hover"])])
    s.map("Treeview", background=[("selected", C["blue"])], foreground=[("selected", "white")])
    s.configure("TNotebook", background=C["bg"], borderwidth=0)
    s.configure("TNotebook.Tab", padding=(16, 7), font=font(9, "bold"))
    s.configure("TCombobox", padding=4)
    s.configure("TCheckbutton", background=C["white"], font=font(9))
    s.configure("TRadiobutton", background=C["white"], font=font(9))
    for name, bg, hover in (("Accent", C["blue"], C["blue_dark"]), ("Danger", C["red"], "#C94545"),
                            ("Ghost", C["white"], C["bg"])):
        fg = C["navy"] if name == "Ghost" else "white"
        s.configure(f"{name}.TButton", background=bg, foreground=fg, font=font(10, "bold"),
                    padding=(14, 7), borderwidth=1, bordercolor=C["border"] if name == "Ghost" else bg)
        s.map(f"{name}.TButton", background=[("active", hover), ("disabled", "#C5CFDD")])
    s.configure("Horizontal.TProgressbar", troughcolor=C["border"], background=C["blue"])


class Card(tk.Frame):
    """White rounded-look panel with an optional title; put content in `.body`."""

    def __init__(self, parent, title: str | None = None, **kw):
        super().__init__(parent, bg=C["white"], highlightthickness=1, highlightbackground=C["border"], **kw)
        if title:
            tk.Label(self, text=title, font=font(11, "bold"), bg=C["white"], fg=C["navy"]).pack(
                anchor="w", padx=14, pady=(12, 4))
        self.body = tk.Frame(self, bg=C["white"])
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 12))


class StatCard(tk.Frame):
    def __init__(self, parent, title: str, color: str = C["blue"]):
        super().__init__(parent, bg=C["white"], highlightthickness=1, highlightbackground=C["border"])
        tk.Frame(self, bg=color, width=5).pack(side="left", fill="y")
        inner = tk.Frame(self, bg=C["white"])
        inner.pack(side="left", fill="both", expand=True, padx=12, pady=10)
        tk.Label(inner, text=title, font=font(9), bg=C["white"], fg=C["muted"]).pack(anchor="w")
        self.value = tk.Label(inner, text="-", font=font(18, "bold"), bg=C["white"], fg=C["navy"])
        self.value.pack(anchor="w")

    def set(self, text: str) -> None:
        self.value.config(text=text)


class ScrollFrame(tk.Frame):
    """Vertically scrollable container; add children to `.inner`."""

    def __init__(self, parent, bg: str = C["bg"]):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.bind_all("<MouseWheel>", self._wheel, add="+")

    def _inside(self, widget) -> bool:
        while widget is not None:
            if widget is self:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _wheel(self, event) -> None:
        try:
            target = self.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            return
        if self.winfo_ismapped() and self._inside(target) and not isinstance(target, (ttk.Treeview, tk.Text)):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")


def make_tree(parent, columns: list[str], widths: dict | None = None, height: int = 10):
    """Treeview with scrollbars and row-colour tags: flag (red), medium (orange), ok (green)."""
    frame = tk.Frame(parent, bg=C["white"])
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
    vs = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vs.grid(row=0, column=1, sticky="ns")
    hs.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    for col in columns:
        tree.heading(col, text=col.replace("_", " ").title())
        tree.column(col, width=(widths or {}).get(col, 110), minwidth=60, anchor="center")
    tree.tag_configure("flag", background=C["red_bg"])
    tree.tag_configure("medium", background=C["orange_bg"])
    tree.tag_configure("ok", background=C["white"])
    return frame, tree


def fill_tree(tree: ttk.Treeview, rows: list[tuple], tags: list[str] | None = None) -> None:
    tree.delete(*tree.get_children())
    for i, row in enumerate(rows):
        tree.insert("", "end", values=row, tags=(tags[i],) if tags else ())


def show_canvas(parent: tk.Misc, fig) -> FigureCanvasTkAgg:
    """Replace whatever is in `parent` by a Matplotlib figure."""
    for child in parent.winfo_children():
        child.destroy()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


def run_async(widget: tk.Misc, work, done, error, on_msg=None) -> None:
    """Run `work(emit)` in a thread; callbacks run on the Tk thread (Tk is not thread-safe)."""
    q: queue.Queue = queue.Queue()

    def target():
        try:
            q.put(("ok", work(lambda m: q.put(("msg", m)))))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user via `error`
            logging.getLogger(__name__).exception("Background task failed")
            q.put(("err", exc))

    threading.Thread(target=target, daemon=True).start()

    def poll():
        finished = False
        while not finished:
            try:
                kind, val = q.get_nowait()
            except queue.Empty:
                break
            if kind == "msg":
                if on_msg:
                    on_msg(val)
            else:
                finished = True
                (done if kind == "ok" else error)(val)
        if not finished:
            widget.after(150, poll)

    widget.after(150, poll)


def fmt_score(x) -> str:
    return "n/a" if x is None else f"{float(x):.4f}"


def risk_tag(risk: str, predicted_class: int) -> str:
    if predicted_class == 1:
        return "flag"
    return "medium" if risk == "Medium Risk" else "ok"


class Page(tk.Frame):
    """Base class for navigation pages."""

    title = ""
    subtitle = ""

    def __init__(self, parent, ctx, app):
        super().__init__(parent, bg=C["bg"])
        self.ctx, self.app = ctx, app
        head = tk.Frame(self, bg=C["bg"])
        head.pack(fill="x", padx=24, pady=(18, 8))
        tk.Label(head, text=self.title, font=font(20, "bold"), bg=C["bg"], fg=C["navy"]).pack(anchor="w")
        if self.subtitle:
            tk.Label(head, text=self.subtitle, font=font(10), bg=C["bg"], fg=C["muted"], wraplength=1000,
                     justify="left").pack(anchor="w")
        self.build()

    def build(self) -> None:  # pragma: no cover - overridden
        pass

    def on_show(self) -> None:
        pass

    def model_banner(self, parent) -> tk.Widget | None:
        if self.ctx.detector is None:
            lab = tk.Label(parent, text="⚠  " + (self.ctx.model_error or "No trained model."), bg=C["orange_bg"],
                           fg="#8A4B08", font=font(9, "bold"), justify="left", wraplength=1000, padx=12, pady=8)
            lab.pack(fill="x", padx=24, pady=(0, 8))
            return lab
        return None

    @staticmethod
    def error(title: str, message: str) -> None:
        messagebox.showerror(title, message)

    @staticmethod
    def info(title: str, message: str) -> None:
        messagebox.showinfo(title, message)
