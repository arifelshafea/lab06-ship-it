import pytest


# Provide a valid sample order for our prediction tests.
@pytest.fixture
def valid_order():
    """Return an order containing all five required model features."""
    return {
        "distance_km": 5.0,
        "items": 3,
        "hour": 14,
        "weather": "clear",
        "day_type": "weekday",
    }
