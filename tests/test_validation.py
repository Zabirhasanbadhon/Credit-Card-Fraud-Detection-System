from src.config import FEATURES
from src.validation import missing_columns, parse_single_input


def _valid():
    return {f: "0.5" for f in FEATURES} | {"Time": "100", "Amount": "25.5"}


def test_valid_input_parses():
    values, errors, warnings = parse_single_input(_valid())
    assert not errors and values["Amount"] == 25.5 and len(values) == len(FEATURES)


def test_errors_for_empty_text_negative_and_nonfinite():
    raw = _valid() | {"V1": "", "V2": "abc", "Amount": "-5", "V3": "inf"}
    _, errors, _ = parse_single_input(raw)
    joined = " ".join(errors)
    assert "V1" in joined and "V2" in joined and "Amount" in joined and "V3" in joined


def test_out_of_range_is_warning_and_value_unchanged():
    values, errors, warnings = parse_single_input(_valid() | {"V5": "99"}, {"V5": [-1.0, 1.0]})
    assert not errors and values["V5"] == 99.0 and any("V5" in w for w in warnings)


def test_missing_columns():
    assert missing_columns(FEATURES) == []
    assert missing_columns(["Time", "Amount"]) != []
