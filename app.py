# Import Gradio to build the web application.
import gradio as gr

# Import our prediction function and saved feature lists.
from predict import FEATURES, predict_order

# Provide the GPU placeholder required by Hugging Face ZeroGPU.
# Keep the application working locally when spaces is unavailable.
try:
    import spaces
except ImportError:
    pass
else:

    @spaces.GPU
    def gpu_placeholder():
        """Placeholder for ZeroGPU startup requirements."""
        pass


# Process one order submitted by the dispatcher.
def check_order(*values):
    """Validate an order and return the prediction results."""

    # Match each submitted value to its feature name.
    order = dict(zip(FEATURES, values, strict=True))

    try:
        # Validate the order and calculate its prediction.
        result = predict_order(order)

    except ValueError as error:
        # Show invalid input as a clear error in Gradio.
        raise gr.Error(str(error)) from error

    # Calculate both outcome probabilities.
    late_probability = result["probability"]
    on_time_probability = 1 - late_probability

    # Explain the decision in plain language.
    if result["offer_voucher"]:
        action = "Offer the customer a $4 voucher."
    else:
        action = "No voucher needed."

    # Return probabilities, action, and threshold information.
    return (
        {
            "On time": on_time_probability,
            "Late": late_probability,
        },
        f"{action} (Decision threshold: {result['threshold']:.0%})",
    )


# Build the Gradio web page for QuickBite dispatchers.
with gr.Blocks(title="QuickBite Late Order Check") as demo:
    gr.Markdown("# QuickBite Late Order Check")
    gr.Markdown("Enter the order details to estimate the chance of a late delivery.")

    # Define the settings for each supported input field.
    field_settings = {
        "distance_km": {"label": "Distance (km)", "value": 5.0},
        "items": {"label": "Number of Items", "value": 3, "precision": 0},
        "hour": {"label": "Order Hour (10-22)", "value": 14, "precision": 0},
        "weather": {
            "label": "Weather",
            "choices": ["clear", "rain", "snow"],
            "value": "clear",
        },
        "day_type": {
            "label": "Day Type",
            "choices": ["weekday", "weekend"],
            "value": "weekday",
        },
    }

    # Create only the fields listed in the saved model information.
    input_controls = {}

    for field in FEATURES:
        settings = field_settings[field]

        if field in ("weather", "day_type"):
            input_controls[field] = gr.Dropdown(**settings)
        else:
            input_controls[field] = gr.Number(**settings)

    # Display the prediction and recommended action.
    probabilities = gr.Label(label="Delivery Probabilities")
    action = gr.Textbox(label="Recommended Action", interactive=False)

    # Run the prediction when the dispatcher clicks the button.
    check_button = gr.Button("Check Order")
    check_button.click(
        fn=check_order,
        inputs=[input_controls[field] for field in FEATURES],
        outputs=[probabilities, action],
    )


# Launch the application when this file is run directly.
if __name__ == "__main__":
    demo.launch()
