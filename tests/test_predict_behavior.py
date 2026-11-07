import pytest

import predict
from predict import THRESHOLD, predict_order

ORDER = {
    "distance_km": 5.0,
    "items": 3,
    "hour": 18,
    "weather": "clear",
    "day_type": "weekday",
}


def test_result_has_expected_keys():
    result = predict_order(ORDER)
    assert set(result) == {"probability", "offer_voucher", "threshold"}


def test_probability_is_float_between_0_and_1():
    p = predict_order(ORDER)["probability"]
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_same_input_gives_same_output():
    assert predict_order(ORDER) == predict_order(ORDER)


def test_input_is_not_modified():
    order = dict(ORDER)
    predict_order(order)
    assert order == ORDER


def test_invalid_order_never_reaches_model():
    with pytest.raises(ValueError):
        predict_order({**ORDER, "hour": 3})


def test_threshold_is_a_valid_probability():
    assert 0.0 < THRESHOLD < 1.0


def test_offer_voucher_matches_probability():
    result = predict_order(ORDER)
    assert result["offer_voucher"] == (result["probability"] >= THRESHOLD)
    assert result["threshold"] == THRESHOLD


def test_model_info_features_match_expected_five():
    assert set(predict.FEATURES) == {
        "distance_km",
        "items",
        "hour",
        "weather",
        "day_type",
    }
