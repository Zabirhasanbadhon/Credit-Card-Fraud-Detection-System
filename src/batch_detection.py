"""Chunked CSV batch scoring. A 'Class' column, if present, is kept separate from model inputs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import FEATURES, TARGET
from .prediction import FraudDetector
from .validation import missing_columns

log = logging.getLogger(__name__)


class CSVError(Exception):
    """Raised for unreadable or structurally invalid CSV files."""


def read_header(path: Path | str) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise CSVError(f"File not found: {path}")
    try:
        return [str(c).strip() for c in pd.read_csv(path, nrows=0).columns]
    except Exception as exc:
        raise CSVError(f"Could not read CSV header: {exc}") from exc


def validate_csv(path: Path | str) -> dict:
    """Return {'columns', 'missing', 'has_target', 'ok'}; raises CSVError if unreadable."""
    cols = read_header(path)
    miss = missing_columns(cols)
    return {"columns": cols, "missing": miss, "has_target": TARGET in cols, "ok": not miss}


def preview_csv(path: Path | str, n: int = 20) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=n)
    except Exception as exc:
        raise CSVError(f"Could not preview CSV: {exc}") from exc


def run_batch(path: Path | str, detector: FraudDetector, chunk_size: int = 20000,
              progress: Callable[[int], None] | None = None) -> tuple[pd.DataFrame, dict]:
    """Score every valid row. Rows with non-numeric/missing feature values are skipped and counted."""
    info = validate_csv(path)
    if not info["ok"]:
        raise CSVError("CSV is missing required columns: " + ", ".join(info["missing"]))
    parts, skipped, offset = [], [], 0
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            row_numbers = pd.Series(range(offset + 1, offset + len(chunk) + 1), index=chunk.index)
            offset += len(chunk)
            feats = chunk[FEATURES].apply(pd.to_numeric, errors="coerce")
            bad = feats.isna().any(axis=1) | (feats["Amount"] < 0) | (feats["Time"] < 0)
            skipped.extend(row_numbers[bad].tolist())
            feats, rn = feats[~bad], row_numbers[~bad]
            if feats.empty:
                continue
            res = detector.predict_frame(feats)
            out = pd.concat([rn.rename("row_number"), feats[["Time", "Amount"]], res], axis=1)
            if TARGET in chunk.columns:   # kept ONLY for retrospective comparison, never a model input
                out["actual_class"] = pd.to_numeric(chunk.loc[feats.index, TARGET], errors="coerce")
            parts.append(out)
            if progress:
                progress(offset)
    except CSVError:
        raise
    except Exception as exc:
        log.exception("Batch detection failed")
        raise CSVError(f"Batch detection failed: {exc}") from exc
    results = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    total, flagged = len(results), int(results["predicted_class"].sum()) if len(results) else 0
    summary = {"total_rows_read": offset, "total_scored": total, "skipped_invalid": len(skipped),
               "skipped_rows_sample": skipped[:20], "legitimate": total - flagged, "flagged": flagged,
               "flagged_percentage": round(100.0 * flagged / total, 4) if total else 0.0,
               "has_target": info["has_target"]}
    return results, summary
