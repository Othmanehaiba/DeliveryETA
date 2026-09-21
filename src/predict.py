"""Model loading and prediction helpers."""

from pathlib import Path
from typing import Any

import joblib


def load_model(model_path: str | Path) -> Any:
    """Load a joblib model from disk."""
    return joblib.load(model_path)


def predict(model: Any, features: Any) -> Any:
    """Generate predictions for the provided features."""
    return model.predict(features)
