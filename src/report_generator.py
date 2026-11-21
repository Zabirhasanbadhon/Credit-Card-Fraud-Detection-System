"""CSV and PDF report generation (ReportLab)."""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import APP_NAME, APP_VERSION, LABELS

log = logging.getLogger(__name__)

DISCLAIMER = ("This report contains automated machine-learning risk assessments. A 'potential fraud' flag is NOT a "
              "confirmed fraud finding and must be verified by a qualified human reviewer. The model was trained on "
              "a public, anonymised European card dataset (2013) and may not generalise to other portfolios. "
              "Fraud probabilities are calibrated estimates on a validation split and may drift over time. "
              "False positives and false negatives are expected.")

RECORD_COLUMNS = ["id", "predicted_at", "amount", "time_feature", "prediction", "fraud_score",
                  "risk_category", "model_name", "source", "review_status"]


class ReportError(Exception):
    pass


def _prep(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        d["prediction"] = LABELS.get(int(r["predicted_class"]), "?")
        d["review_status"] = r.get("review_status") or ("Not applicable" if int(r["predicted_class"]) == 0 else "Pending Review")
        out.append(d)
    return out


def export_records_csv(rows: list[dict], path: Path | str) -> int:
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=RECORD_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(_prep(rows))
    except OSError as exc:
        raise ReportError(f"Could not write CSV report: {exc}") from exc
    return len(rows)


def model_table_rows(metrics: dict) -> list[list]:
    head = ["Model", "Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC-AUC", "PR-AUC"]
    rows = [head]
    for r in metrics["models"] + ([metrics["isolation_forest"]] if metrics.get("isolation_forest") else []):
        t = r["test"]
        rows.append([r["name"]] + [f"{t[k]:.4f}" for k in
                    ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc")])
    return rows


def export_model_csv(metrics: dict, path: Path | str) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(model_table_rows(metrics))
    except OSError as exc:
        raise ReportError(f"Could not write CSV report: {exc}") from exc


def _table(data, col_widths=None, header_color="#0B1F3A", font=7):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E1EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FB")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def export_pdf(path: Path | str, kind: str, rows: list[dict], stats: dict, metrics: dict | None,
               max_rows: int = 500) -> None:
    """kind: 'full' | 'flagged' | 'model'."""
    if kind == "model" and not metrics:
        raise ReportError("No trained model metrics available. Train the models first.")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#0B1F3A"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#1E5FBF"))
    body, small = ss["BodyText"], ParagraphStyle("sm", parent=ss["BodyText"], fontSize=7.5, leading=10)
    title = {"full": "Full Transaction Analysis Report", "flagged": "Flagged Transactions Report",
             "model": "Model Performance Report"}[kind]
    try:
        doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                                topMargin=14 * mm, bottomMargin=14 * mm, title=f"{APP_NAME} - {title}")
        s = [Paragraph(APP_NAME, h1), Paragraph(title, ss["Heading3"]),
             Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M} | Application v{APP_VERSION}", body), Spacer(1, 6)]
        if kind in ("full", "flagged"):
            s += [Paragraph("Summary (from prediction history)", h2), _table([
                ["Transactions analyzed", "Classified legitimate", "Potentially fraudulent", "Flagged %", "Selected model"],
                [f"{stats['total']:,}", f"{stats['legitimate']:,}", f"{stats['flagged']:,}", f"{stats['flagged_pct']}%",
                 metrics["selected_model"] if metrics else "n/a"]], font=9), Spacer(1, 6)]
            risk = stats["by_risk"]
            s += [Paragraph("Risk assessment summary", h2),
                  Paragraph(" | ".join(f"{k}: {risk.get(k, 0):,}" for k in ("High Risk", "Medium Risk", "Low Risk")), body),
                  Paragraph("Risk bands are configurable score bands relative to the decision threshold; "
                            "they are not independently validated fraud probabilities.", small), Spacer(1, 6)]
        if kind in ("full", "model") and metrics:
            s += [Paragraph("Model evaluation (held-out test set)", h2), _table(model_table_rows(metrics), font=8)]
            best = next(m for m in metrics["models"] if m["name"] == metrics["selected_model"])
            cm = best["test"]["confusion_matrix"]
            s += [Spacer(1, 4), Paragraph(
                f"Selected: <b>{best['name']}</b> at threshold {best['threshold']:.4f}. Test confusion matrix: "
                f"TN={cm['tn']:,} FP={cm['fp']:,} FN={cm['fn']:,} TP={cm['tp']:,}. "
                f"Selection rule: {metrics['selection_rule']}. The Isolation Forest row is an unsupervised model "
                "evaluated separately.", body), Spacer(1, 6)]
        if kind in ("full", "flagged"):
            data_rows = _prep([r for r in rows if kind == "full" or int(r["predicted_class"]) == 1])
            shown = data_rows[:max_rows]
            s.append(Paragraph(f"{'Flagged' if kind == 'flagged' else 'Transaction'} records "
                               f"({len(shown):,} of {len(data_rows):,} shown)", h2))
            table = [["ID", "Predicted at", "Amount", "Prediction", "Score", "Risk", "Review status", "Source"]]
            for r in shown:
                sc = r.get("fraud_score")
                table.append([r["id"], r["predicted_at"], f"{r['amount']:.2f}", r["prediction"],
                              f"{sc:.4f}" if sc is not None else "n/a", r["risk_category"], r["review_status"], r["source"]])
            s.append(_table(table) if len(table) > 1 else Paragraph("No records.", body))
        s += [Spacer(1, 10), Paragraph("Important limitations and disclaimer", h2), Paragraph(DISCLAIMER, body)]
        doc.build(s)
    except OSError as exc:
        raise ReportError(f"Could not write PDF: {exc}") from exc
    except Exception as exc:
        log.exception("PDF generation failed")
        raise ReportError(f"PDF generation failed: {exc}") from exc
