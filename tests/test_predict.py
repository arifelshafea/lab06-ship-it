import json
from pathlib import Path

import pytest

from predict import THRESHOLD, predict_order, should_offer_voucher, validate_order


# Test that a valid order passes validation.
def test_valid_order(valid_order):
    result = validate_order(valid_order)

    assert result == valid_order


# Test that the model returns a probability between 0 and 1.
def test_probability_range(valid_order):
    result = predict_order(valid_order)

    assert 0 <= result["probability"] <= 1


# Test voucher decisions below, at, and above the saved threshold.
@pytest.mark.parametrize(
    "difference, expected",
    [
        (-0.01, False),  # Below threshold: no voucher
        (0.00, True),  # Exactly at threshold: offer voucher
        (0.01, True),  # Above threshold: offer voucher
    ],
)
def test_voucher_threshold_boundary(difference, expected):
    """Check the voucher decision around the saved threshold."""

    probability = THRESHOLD + difference

    assert should_offer_voucher(probability) is expected


# Test that invalid numeric values are rejected with clear error messages.
@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("distance_km", 0),
        ("distance_km", 31),
        ("items", 0),
        ("items", 21),
        ("items", 2.5),
        ("hour", 9),
        ("hour", 23),
        ("hour", 14.5),
    ],
)
def test_invalid_numeric_values(valid_order, field, invalid_value):
    """Reject numeric values outside their allowed limits."""

    # Copy the valid order so the original fixture stays unchanged.
    order = valid_order.copy()

    # Replace one field with an invalid value.
    order[field] = invalid_value

    # Confirm the error message identifies the invalid field.
    with pytest.raises(ValueError, match=field):
        validate_order(order)


# Test that every required model feature must be provided.
@pytest.mark.parametrize(
    "field",
    [
        "distance_km",
        "items",
        "hour",
        "weather",
        "day_type",
    ],
)
def test_missing_required_fields(valid_order, field):
    """Reject missing features with a clear error message."""

    # Copy the valid order.
    order = valid_order.copy()

    # Remove one required feature.
    del order[field]

    # Confirm that validation rejects the missing feature.
    with pytest.raises(ValueError, match=field):
        validate_order(order)


# Test that unknown categorical values are rejected.
@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("weather", "Snow"),
        ("weather", "storm"),
        ("day_type", "Weekday"),
        ("day_type", "holiday"),
    ],
)
def test_invalid_categories(valid_order, field, invalid_value):
    """Reject categories that the model was not trained to recognize."""

    # Copy the valid order.
    order = valid_order.copy()

    # Replace one category with an invalid value.
    order[field] = invalid_value

    # Confirm the error message names the invalid field.
    with pytest.raises(ValueError, match=field):
        validate_order(order)


# Test that a harder delivery has a higher late-delivery probability.
def test_harder_order_has_higher_probability():
    """Compare predictions for an easy and a difficult order."""

    # An easier delivery.
    easy_order = {
        "distance_km": 1.0,
        "items": 1,
        "hour": 14,
        "weather": "clear",
        "day_type": "weekday",
    }

    # A harder delivery.
    hard_order = {
        "distance_km": 25.0,
        "items": 15,
        "hour": 20,
        "weather": "snow",
        "day_type": "weekend",
    }

    # Calculate the late-delivery probabilities.
    easy_probability = predict_order(easy_order)["probability"]
    hard_probability = predict_order(hard_order)["probability"]

    # The harder order should have a higher probability.
    assert hard_probability > easy_probability


# Test that the saved model loads only once, not for every prediction.
def test_model_loads_once(valid_order, monkeypatch):
    """Count joblib.load calls when reloading predict.py."""

    import importlib

    import joblib

    import predict

    # Save the original joblib.load function.
    original_load = joblib.load

    # Count how many times the model is loaded.
    load_count = 0

    def counting_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    # Temporarily replace joblib.load with our counting function.
    with monkeypatch.context() as patch:
        patch.setattr(joblib, "load", counting_load)

        # Reload the module to simulate its first import.
        importlib.reload(predict)

        # Make two predictions using the same loaded model.
        predict.predict_order(valid_order)
        predict.predict_order(valid_order)

    # The model should have loaded exactly once.
    assert load_count == 1


# Test that required fields cannot contain None or empty strings.
@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("distance_km", None),
        ("items", ""),
        ("hour", None),
        ("weather", ""),
        ("day_type", None),
    ],
)
def test_empty_required_values(valid_order, field, invalid_value):
    """Reject empty required fields with a clear error message."""

    # Copy the valid order.
    order = valid_order.copy()

    # Replace one required value with None or an empty string.
    order[field] = invalid_value

    # Confirm the error identifies the incorrect field.
    with pytest.raises(ValueError, match=field):
        validate_order(order)


# Test that the voucher threshold matches the saved model information.
def test_threshold_matches_model_info():
    """Verify that the threshold comes from model_info.json."""

    # Locate the saved model information file.
    info_path = Path(__file__).resolve().parents[1] / "models" / "model_info.json"

    # Read the threshold stored with the model.
    with open(info_path, encoding="utf-8") as file:
        model_info = json.load(file)

    # Compare the saved threshold with the one used by predict.py.
    assert THRESHOLD == model_info["threshold"]
