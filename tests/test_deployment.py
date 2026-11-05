# Import libraries to check package versions and project files.
import json
from importlib.metadata import version
from pathlib import Path

# Locate the main Lab 06 project folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]


# Test that scikit-learn versions match across the project.
def test_scikit_learn_version():
    """Check that the installed version matches the saved model and requirements."""

    # Read the saved model information.
    info_path = PROJECT_DIR / "models" / "model_info.json"

    with open(info_path, encoding="utf-8") as file:
        model_info = json.load(file)

    # Read the Space requirements.
    requirements = (PROJECT_DIR / "requirements.txt").read_text(encoding="utf-8")

    # Get the version used to train the model.
    saved_version = model_info["scikit_learn_version"]

    # Verify that all three versions agree.
    assert version("scikit-learn") == saved_version
    assert f"scikit-learn=={saved_version}" in requirements

    # Gradio must not appear in the Space requirements.
    assert "gradio" not in requirements.lower()


# Test the settings used by Hugging Face Spaces.
def test_space_settings():
    """Check the Gradio version and application file in README.md."""

    # Read the README file.
    readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")

    # Extract the settings block at the beginning.
    assert readme.startswith("---")
    settings = readme.split("---", 2)[1]

    # Check the installed Gradio version against the README.
    assert f"sdk_version: {version('gradio')}" in settings

    # Check that the README identifies the correct application file.
    assert "app_file: app.py" in settings
    assert (PROJECT_DIR / "app.py").is_file()
