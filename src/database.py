"""SQLite prediction history and review workflow. All queries are parameterised."""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import DB_PATH, REVIEW_STATUSES

log = logging.getLogger(__name__)


class DatabaseError(Exception):
    """User-presentable database failure."""


SORTABLE = {"id": "p.id", "predicted_at": "p.predicted_at", "amount": "p.amount",
            "fraud_score": "p.fraud_score", "risk_category": "p.risk_category",
            "predicted_class": "p.predicted_class"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_datetime TEXT,                 -- only if the user supplied one; the dataset has no real dates
    amount REAL NOT NULL,
    time_feature REAL,
    predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
    fraud_score REAL,                 -- calibrated probability estimate from the model
    risk_category TEXT,
    model_name TEXT,
    predicted_at TEXT NOT NULL,
    source TEXT NOT NULL,             -- 'manual' or 'csv'
    features_json TEXT                -- NULL unless 'store full features' is enabled
);
CREATE TABLE IF NOT EXISTS reviews (  -- human review, stored SEPARATELY from the model prediction
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'Pending Review',
    note TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_class ON predictions(predicted_class);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(predicted_at);
"""

_FIELDS = ("tx_datetime", "amount", "time_feature", "predicted_class", "fraud_score",
           "risk_category", "model_name", "predicted_at", "source", "features_json")


class Database:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.executescript(SCHEMA)
        except (sqlite3.Error, OSError) as exc:
            log.exception("Database init failed")
            raise DatabaseError(f"Could not open database: {exc}") from exc

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            log.exception("Database error")
            raise DatabaseError(f"Database error: {exc}") from exc
        finally:
            conn.close()

    # ---------------------------------------------------------------- writes
    def add_predictions(self, records: list[dict]) -> int:
        """Insert prediction records; flagged ones also get a 'Pending Review' review row."""
        if not records:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            n = 0
            for r in records:
                row = {k: r.get(k) for k in _FIELDS}
                row["predicted_at"] = row["predicted_at"] or now
                if isinstance(row["features_json"], dict):
                    row["features_json"] = json.dumps(row["features_json"])
                cur = c.execute(f"INSERT INTO predictions ({','.join(_FIELDS)}) "
                                f"VALUES ({','.join('?' * len(_FIELDS))})", [row[k] for k in _FIELDS])
                if int(row["predicted_class"]) == 1:
                    c.execute("INSERT INTO reviews (prediction_id, status, updated_at) VALUES (?,?,?)",
                              (cur.lastrowid, "Pending Review", now))
                n += 1
        return n

    def set_review(self, prediction_id: int, status: str, note: str = "") -> None:
        if status not in REVIEW_STATUSES:
            raise DatabaseError(f"Invalid review status: {status}")
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("INSERT INTO reviews (prediction_id, status, note, updated_at) VALUES (?,?,?,?) "
                      "ON CONFLICT(prediction_id) DO UPDATE SET status=excluded.status, "
                      "note=excluded.note, updated_at=excluded.updated_at",
                      (prediction_id, status, note.strip(), now))

    def delete(self, prediction_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))

    def clear_all(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM reviews")
            c.execute("DELETE FROM predictions")
            c.execute("DELETE FROM sqlite_sequence WHERE name IN ('predictions')")

    # ---------------------------------------------------------------- reads
    def query(self, search_id: str = "", predicted_class: int | None = None, risk: str = "",
              date: str = "", review_status: str = "", flagged_only: bool = False,
              sort_by: str = "id", descending: bool = True, limit: int = 5000) -> list[dict]:
        sql = ("SELECT p.*, r.status AS review_status, r.note AS review_note, r.updated_at AS review_updated "
               "FROM predictions p LEFT JOIN reviews r ON r.prediction_id = p.id WHERE 1=1")
        params: list = []
        if str(search_id).strip():
            if not str(search_id).strip().isdigit():
                raise DatabaseError("Prediction ID must be a whole number.")
            sql += " AND p.id = ?"; params.append(int(search_id))
        if predicted_class in (0, 1):
            sql += " AND p.predicted_class = ?"; params.append(predicted_class)
        if flagged_only:
            sql += " AND p.predicted_class = 1"
        if risk:
            sql += " AND p.risk_category = ?"; params.append(risk)
        if date.strip():
            try:
                datetime.strptime(date.strip(), "%Y-%m-%d")
            except ValueError as exc:
                raise DatabaseError("Date must use the format YYYY-MM-DD.") from exc
            sql += " AND substr(p.predicted_at, 1, 10) = ?"; params.append(date.strip())
        if review_status:
            sql += " AND r.status = ?"; params.append(review_status)
        order = SORTABLE.get(sort_by, "p.id")          # whitelist -> no SQL injection via sort column
        sql += f" ORDER BY {order} {'DESC' if descending else 'ASC'} LIMIT ?"
        params.append(int(limit))
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def get(self, prediction_id: int) -> dict | None:
        rows = self.query(search_id=str(prediction_id))
        return rows[0] if rows else None

    def stats(self) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) n, COALESCE(SUM(predicted_class),0) f FROM predictions").fetchone()
            by_risk = {r["risk_category"]: r["n"] for r in c.execute(
                "SELECT risk_category, COUNT(*) n FROM predictions GROUP BY risk_category")}
            by_review = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM reviews GROUP BY status")}
            daily = [dict(r) for r in c.execute(
                "SELECT substr(predicted_at,1,10) day, COUNT(*) total, SUM(predicted_class) flagged "
                "FROM predictions GROUP BY day ORDER BY day")]
        total, flagged = int(row["n"]), int(row["f"])
        return {"total": total, "flagged": flagged, "legitimate": total - flagged,
                "flagged_pct": round(100.0 * flagged / total, 2) if total else 0.0,
                "by_risk": by_risk, "by_review": by_review, "daily": daily}

    def export_csv(self, path: Path | str, rows: list[dict] | None = None) -> int:
        rows = self.query(limit=10_000_000) if rows is None else rows
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                cols = [k for k in (rows[0].keys() if rows else
                        ["id", "amount", "predicted_class", "fraud_score", "risk_category"]) if k != "features_json"]
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        except OSError as exc:
            raise DatabaseError(f"Could not write CSV: {exc}") from exc
        return len(rows)
