# Import the libraries needed to load our saved model and model information.
import json
from pathlib import Path

import joblib
import pandas as pd

# Locate the models folder relative to this Python file.
MODEL_DIR = Path(__file__).resolve().parent / "models"


# Load the saved Lab 05 pipeline once when predict.py is imported.
MODEL = joblib.load(MODEL_DIR / "late_orders.joblib")


# Load the model information once when predict.py is imported.
with open(MODEL_DIR / "model_info.json", encoding="utf-8") as file:
    MODEL_INFO = json.load(file)


# Read the decision threshold and expected features from the saved information.
THRESHOLD = MODEL_INFO["threshold"]

NUMERIC_FEATURES = MODEL_INFO["numeric_features"]
CATEGORICAL_FEATURES = MODEL_INFO["categorical_features"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Check that an order contains valid values before making a prediction.
def validate_order(order):
    """Validate the five features expected by our saved model."""

    # Make sure the order is a dictionary.
    if not isinstance(order, dict):
        raise ValueError("Order must be a dictionary.")

    # Check that every required feature is present.
    for field in FEATURES:
        if field not in order or order[field] is None or order[field] == "":
            raise ValueError(f"{field}: This field is required.")

    # Check the distance.
    distance = order["distance_km"]
    if type(distance) not in (int, float) or not 0.1 <= distance <= 30:
        raise ValueError("distance_km: Must be a number between 0.1 and 30.")

    # Check the number of items.
    items = order["items"]
    if type(items) is not int or not 1 <= items <= 20:
        raise ValueError("items: Must be a whole number between 1 and 20.")

    # Check the order hour.
    hour = order["hour"]
    if type(hour) is not int or not 10 <= hour <= 22:
        raise ValueError("hour: Must be a whole number between 10 and 22.")

    # Check the weather category.
    if order["weather"] not in ("clear", "rain", "snow"):
        raise ValueError("weather: Must be clear, rain, or snow.")

    # Check the day type.
    if order["day_type"] not in ("weekday", "weekend"):
        raise ValueError("day_type: Must be weekday or weekend.")

    # Return only the features expected by the model.
    return {field: order[field] for field in FEATURES}


# Decide whether to offer a $4 voucher based on the saved threshold.
def should_offer_voucher(probability):
    """Return True when the probability reaches the saved threshold."""
    return probability >= THRESHOLD


# Validate an order, predict its late-delivery probability,
# and decide whether the customer should receive a $4 voucher.
def predict_order(order):
    """Return the prediction results for one order."""

    # Validate the order before sending it to the model.
    valid_order = validate_order(order)

    # Convert the validated order into a one-row DataFrame.

    order_df = pd.DataFrame([valid_order])

    # Predict the probability that the order will arrive late.
    probability = float(MODEL.predict_proba(order_df)[0][1])

    # Decide whether to offer the voucher.
    offer_voucher = should_offer_voucher(probability)

    # Return the results for the Gradio application.
    return {
        "probability": probability,
        "offer_voucher": offer_voucher,
        "threshold": THRESHOLD,
    }
