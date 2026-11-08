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

Space URL: https://huggingface.co/spaces/SaraAly/lab06-ship-it

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
- Boundary values at the edge of each allowed range are accepted.
- Non-dictionary input, extra keys, NaN, infinity, strings, booleans and whole-number floats are handled correctly.
- Prediction output has the expected keys and is deterministic.
- Greater distance and snow weather do not reduce the late-delivery probability.
- The saved model files exist and `model_info.json` contains the required keys.

The full suite now contains 81 tests, and all pass.

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

Running the tests shows NumPy deprecation warnings that come from the libraries when the saved model is loaded. They do not affect results, but pinning the NumPy version in `requirements.txt` would protect the application from breaking when libraries update.

`validate_order()` currently rejects whole-number floats such as `items: 3.0` and NumPy integers such as `np.int64`, because it checks `type(...) is int`. Gradio number inputs can return floats, so accepting whole-number values and converting them to `int` would be safer. Extra keys in an order are silently ignored.


## Team Contributions

**Arif Elshafea**
- **Arif Elshafea — Model Integration, Application Development, and Testing**
- Developed the original Lab 05 machine learning model used in Lab 06.
- Integrated the saved model and metadata into the Lab 06 repository.
- Implemented `predict.py`, including input validation, probability prediction, and the $4 voucher decision.
- Developed the Gradio web interface in `app.py`.
- Created automated prediction, validation, and deployment tests.
- Configured the Conda environment, requirements files, pytest, and Ruff.
- Prepared the initial README model card and documented the model's limitations.
- Verified that all 32 automated tests and Ruff checks passed, including from a fresh GitHub clone.

- **AI Use:** Used ChatGPT to help troubleshoot Python coding errors, clarify technical questions, organize my work, and review the language and formatting of the documentation. All code changes and results were reviewed and tested.

**Sara Aly**
- Created and configured the public Hugging Face Gradio Space.
- Uploaded the application files and saved model to Hugging Face.
- Tested the deployed application and verified that it generates predictions successfully.
- Ran the automated tests locally and confirmed that all 32 tests passed.
- Verified that both Ruff checks passed locally.

**Devashree Paranjape**
- Wrote additional automated tests for model testing in four new files: `test_validation.py`, `test_predict_behavior.py`, `test_model_behavior.py` and `test_artifacts.py`.
- Added tests for exact boundary values (for example distance 0.1 and 30, items 1 and 20, hours 10 and 22) and for unusual input types, including NaN, infinity, strings, booleans, whole-number floats and wrong-case categories.
- Added tests for the prediction output (required keys, float probability between 0 and 1, deterministic results, input not modified) and for model behavior (greater distance and snow do not lower the late probability, and every weather and day-type combination produces a valid prediction).
- Added tests that check that the saved model files exist and that `model_info.json` contains the keys `predict.py` needs.
- Reviewed the existing tests to avoid duplicating them, ran the full suite locally (81 tests passed) and recorded the library warnings.
- Reviewed `predict.py` and suggested improvements (see Limitations).