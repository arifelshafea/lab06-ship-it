---
title: QuickBite Late Order Check
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
pinned: false
short_description: Will this order arrive late?
---

# QuickBite Late Order Check

## What it does

This Gradio application predicts the probability that a QuickBite food delivery will arrive late using the machine learning model developed in Lab 05.

Dispatchers enter order details and receive the probabilities of on-time and late delivery. If the predicted late-delivery probability is at least the saved threshold of 65%, the application recommends offering the customer a $4 voucher.

Space URL: To be added after deployment.

## How to run it locally

Clone the GitHub repository and open the project folder.

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate lab06-ship-it
```

Start the Gradio application:

```bash
gradio app.py
```

Open the local URL displayed in the terminal, usually http://127.0.0.1:7860.

## Decisions

The application uses the five features listed in `model_info.json`. Each input is validated before being passed to the model.

| Field | Allowed values | Reason |
|---|---|---|
| distance_km | 0.1 to 30 | Keeps the distance within the model's supported training range. |
| items | Whole numbers from 1 to 20 | Prevents impossible or unrealistic item counts. |
| hour | Whole numbers from 10 to 22 | Ensures the order hour is within the supported range. |
| weather | clear, rain, snow | Rejects unknown weather categories that could produce misleading predictions. |
| day_type | weekday, weekend | Prevents unsupported day categories from reaching the model. |

**Feature selection**

The model uses only information available when an order is placed.

The `prep_minutes` feature was excluded during Lab 05 because actual preparation time may not be known when the order is placed. This helps prevent potential data leakage.

The `courier_months` and `vehicle` features were also excluded because courier information may not be available before a courier is assigned.

The application reads its expected features and decision threshold from `model_info.json`.

Invalid inputs raise clear error messages identifying the incorrect field and the allowed values.

**Weather category validation experiment**

To demonstrate why input validation is important, I compared two otherwise identical orders using the underlying saved model:

- `snow` produced a late-delivery probability of **60.23%**.
- `Snow` produced a late-delivery probability of **49.82%**.

The difference was **10.41 percentage points**, even though only the capitalization changed.

This happens because the model's OneHotEncoder uses `handle_unknown="ignore"`. An unknown category can therefore be processed without raising an error, potentially producing a misleading prediction.

Our `validate_order()` function prevents this problem by accepting only `clear`, `rain`, and `snow`. The application rejects `Snow` with a clear error message before it reaches the model.

## Result

The model was trained and evaluated in Lab 05 using a stratified train/test split.

The saved test-set results were:

| Metric | Result |
|---|---|
| ROC-AUC | 0.7636 |
| Precision | 0.4492 |
| Recall | 0.4530 |
| F1-score | 0.4511 |
| Selected threshold | 0.65 |
| Cost without flagging orders | $1,755 |
| Cost when flagging every order | $3,000 |
| Cost using the model | $1,432 |

The model reduced the estimated test-set cost by **$323** compared with flagging no orders, representing approximately **18.4% cost savings**.

The selected threshold is stored in `models/model_info.json` and is used for every prediction.

## Tests

The automated test suite verifies:

- Valid orders are accepted.
- Predicted probabilities are between 0 and 1.
- A harder order receives a higher late-delivery probability than an easier order in the selected comparison.
- Voucher decisions work below, at, and above the saved threshold.
- The threshold matches `model_info.json`.
- Invalid numeric values are rejected.
- Missing required fields are rejected.
- Empty required values are rejected.
- Unknown categorical values are rejected.
- The saved model loads only once.
- The installed scikit-learn version matches the saved model and requirements.
- The Gradio version and application file match the README settings.

From the project root, run:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

All three commands must pass before submission.

## Limitations

The model was trained on the synthetic QuickBite delivery-order data generated in Lab 05. Its results may not generalize to real-world food delivery operations.

The model cannot account for unexpected events such as road closures, traffic accidents, sudden congestion, or other delivery disruptions.

The model uses only five order-time features. It does not consider courier-specific information or actual preparation time.

Predicted probabilities are estimates, not guarantees that an order will arrive on time or late.

Dispatchers should use the prediction as decision support when considering a $4 voucher. It should not be used as a guarantee of delivery time or as the sole basis for operational decisions.
