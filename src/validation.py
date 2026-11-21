"""Input validation for single-transaction forms and CSV files. Values are never silently changed."""
from __future__ import annotations

import math

from .config import FEATURES


def parse_single_input(raw: dict[str, str], ranges: dict | None = None):
    """Validate form strings. Returns (values, errors, warnings).
    Errors block prediction; warnings (e.g. outside the training range) do not."""
    values, errors, warnings = {}, [], []
    for name in FEATURES:
        text = str(raw.get(name, "")).strip()
        if text == "":
            errors.append(f"{name}: required field is empty.")
            continue
        try:
            number = float(text)
        except ValueError:
            errors.append(f"{name}: '{text}' is not a valid number.")
            continue
        if not math.isfinite(number):
            errors.append(f"{name}: value must be a finite number.")
            continue
        if name == "Amount" and number < 0:
            errors.append("Amount: transaction amount cannot be negative.")
            continue
        if name == "Time" and number < 0:
            errors.append("Time: seconds since the first transaction cannot be negative.")
            continue
        values[name] = number
        if ranges and name in ranges:
            lo, hi = ranges[name]
            if number < lo or number > hi:
                warnings.append(f"{name}={number:g} is outside the training range [{lo:g}, {hi:g}]; "
                                "the prediction may be unreliable.")
    return values, errors, warnings


def missing_columns(columns) -> list[str]:
    """Required model features absent from a CSV header."""
    present = {str(c).strip() for c in columns}
    return [c for c in FEATURES if c not in present]
