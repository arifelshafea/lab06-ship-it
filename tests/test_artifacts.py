import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def test_model_files_exist():
    assert (MODEL_DIR / "late_orders.joblib").is_file()
    assert (MODEL_DIR / "model_info.json").is_file()


def test_model_info_has_required_keys():
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    for key in [
        "threshold",
        "numeric_features",
        "categorical_features",
        "scikit_learn_version",
    ]:
        assert key in info
