import pytest

from src.database import Database, DatabaseError


def rec(cls=0, amount=10.0, risk="Low Risk", score=0.1):
    return {"amount": amount, "time_feature": 5.0, "predicted_class": cls, "fraud_score": score,
            "risk_category": risk, "model_name": "M", "source": "manual"}


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "t.db")


def test_add_query_stats(db):
    db.add_predictions([rec(0), rec(1, 99.0, "High Risk", 0.9), rec(1, 50.0, "High Risk", 0.8)])
    assert db.stats()["total"] == 3 and db.stats()["flagged"] == 2
    assert len(db.query(predicted_class=1)) == 2
    assert len(db.query(risk="Low Risk")) == 1
    assert db.query(sort_by="amount", descending=False)[0]["amount"] == 10.0
    assert db.query(search_id="1")[0]["id"] == 1


def test_review_is_separate_from_prediction(db):
    db.add_predictions([rec(1, 99.0, "High Risk", 0.9)])
    row = db.query()[0]
    assert row["review_status"] == "Pending Review" and row["predicted_class"] == 1
    db.set_review(row["id"], "False Positive", "customer confirmed")
    row = db.query()[0]
    assert row["review_status"] == "False Positive" and row["predicted_class"] == 1   # prediction unchanged
    with pytest.raises(DatabaseError):
        db.set_review(row["id"], "Bogus")


def test_delete_clear_export_and_safe_inputs(db, tmp_path):
    db.add_predictions([rec(1), rec(0)])
    db.delete(1)
    assert db.stats()["total"] == 1
    assert db.export_csv(tmp_path / "o.csv") == 1
    with pytest.raises(DatabaseError):
        db.query(search_id="1 OR 1=1")
    with pytest.raises(DatabaseError):
        db.query(date="not-a-date")
    assert db.query(sort_by="id; DROP TABLE predictions")        # unknown sort key falls back safely
    db.clear_all()
    assert db.stats()["total"] == 0


def test_full_features_not_stored_by_default(db):
    db.add_predictions([rec(0)])
    assert db.query()[0]["features_json"] is None
