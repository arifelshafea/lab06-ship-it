import pytest

from predict import validate_order

VALID = {
    "distance_km": 5.0,
    "items": 3,
    "hour": 18,
    "weather": "clear",
    "day_type": "weekday",
}


def make(**changes):
    return {**VALID, **changes}


def test_extra_keys_are_dropped():
    assert validate_order(make(extra="x")) == VALID


@pytest.mark.parametrize("bad", [None, [], "order", 42])
def test_non_dict_rejected(bad):
    with pytest.raises(ValueError):
        validate_order(bad)


@pytest.mark.parametrize("field", list(VALID))
@pytest.mark.parametrize("empty", [None, ""])
def test_missing_or_empty_field_rejected(field, empty):
    with pytest.raises(ValueError, match=field):
        validate_order(make(**{field: empty}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("distance_km", 0.1),
        ("distance_km", 30),
        ("items", 1),
        ("items", 20),
        ("hour", 10),
        ("hour", 22),
    ],
)
def test_boundary_values_accepted(field, value):
    validate_order(make(**{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("distance_km", 0.09),
        ("distance_km", 30.1),
        ("distance_km", -5),
        ("distance_km", "5"),
        ("distance_km", float("nan")),
        ("distance_km", float("inf")),
        ("items", True),
        ("hour", 18.0),
        ("weather", "Rain"),
        ("weather", "fog"),
    ],
)
def test_unusual_values_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        validate_order(make(**{field: value}))
