"""FastAPI service for delivery-time predictions."""

from typing import Any

from fastapi import FastAPI


app = FastAPI(title="Food Delivery Time Prediction API")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Food Delivery Time Prediction API"}


@app.post("/predict")
def make_prediction(features: dict[str, Any]) -> dict[str, Any]:
    """Return a placeholder prediction until a trained model is configured."""
    return {"prediction": None, "features": features}
