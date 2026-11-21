from src import analytics
from src.database import Database
from src.report_generator import export_model_csv, export_pdf, export_records_csv


def _db(tmp_path):
    db = Database(tmp_path / "r.db")
    db.add_predictions([
        {"amount": 12.5, "time_feature": 1.0, "predicted_class": 0, "fraud_score": 0.01, "risk_category": "Low Risk",
         "model_name": "M", "source": "manual"},
        {"amount": 900.0, "time_feature": 2.0, "predicted_class": 1, "fraud_score": 0.93, "risk_category": "High Risk",
         "model_name": "M", "source": "csv"}])
    return db


def test_all_eleven_charts_build(trained, tmp_path):
    stats = _db(tmp_path).stats()
    assert len(analytics.CHARTS) == 11
    for name, build in analytics.CHARTS.items():
        fig = build(trained["metrics"], stats)
        fig.savefig(tmp_path / "c.png")            # forces a full render
    analytics.CHARTS["2. Fraud Detection Trends Over Time"](None, {"daily": []})   # empty states don't crash
    analytics.CHARTS["5. Model Accuracy Comparison"](None, stats)


def test_csv_and_pdf_exports(trained, tmp_path):
    db = _db(tmp_path)
    rows, stats, m = db.query(), db.stats(), trained["metrics"]
    assert export_records_csv(rows, tmp_path / "a.csv") == 2
    export_model_csv(m, tmp_path / "m.csv")
    for kind in ("full", "flagged", "model"):
        p = tmp_path / f"{kind}.pdf"
        export_pdf(p, kind, rows, stats, m)
        assert p.read_bytes()[:4] == b"%PDF" and p.stat().st_size > 1000
