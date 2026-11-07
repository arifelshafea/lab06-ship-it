import pytest

from predict import predict_order

EASY = {
    "distance_km": 1.0,
    "items": 1,
    "hour": 15,
    "weather": "clear",
    "day_type": "weekday",
}


def prob(order):
    return predict_order(order)["probability"]


def test_farther_distance_not_less_risky():
    near = prob({**EASY, "distance_km": 1.0})
    far = prob({**EASY, "distance_km": 25.0})
    assert far >= near


def test_snow_not_less_risky_than_clear():
    assert prob({**EASY, "weather": "snow"}) >= prob({**EASY, "weather": "clear"})


@pytest.mark.parametrize("weather", ["clear", "rain", "snow"])
@pytest.mark.parametrize("day_type", ["weekday", "weekend"])
def test_every_category_combination_predicts(weather, day_type):
    p = prob({**EASY, "weather": weather, "day_type": day_type})
    assert 0.0 <= p <= 1.0
