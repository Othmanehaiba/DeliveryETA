#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${1:-.}"

mkdir -p \
  "$PROJECT_ROOT/data/raw" \
  "$PROJECT_ROOT/data/processed" \
  "$PROJECT_ROOT/notebooks" \
  "$PROJECT_ROOT/src" \
  "$PROJECT_ROOT/models" \
  "$PROJECT_ROOT/app" \
  "$PROJECT_ROOT/api" \
  "$PROJECT_ROOT/dags"

cat > "$PROJECT_ROOT/README.md" <<'EOF'
# Food Delivery Time Prediction

Projet de Machine Learning pour prédire le temps de livraison des commandes alimentaires.

## Étapes du projet

1. Collecter et stocker les données brutes dans `data/raw/`.
2. Nettoyer et préparer les données dans `data/processed/`.
3. Explorer les données et créer de nouvelles features dans `notebooks/`.
4. Entraîner et évaluer un modèle de prédiction.
5. Sauvegarder le modèle dans `models/`.
6. Exposer les prédictions via Streamlit et FastAPI.
7. Automatiser le réentraînement avec Airflow.
EOF

cat > "$PROJECT_ROOT/requirements.txt" <<'EOF'
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
fastapi
uvicorn
mlflow
EOF

cat > "$PROJECT_ROOT/src/__init__.py" <<'EOF'
"""Core package for food delivery time prediction."""
EOF

cat > "$PROJECT_ROOT/src/utils.py" <<'EOF'
"""Utility functions for the delivery-time prediction project."""

from math import asin, cos, radians, sin, sqrt


def haversine_distance(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
    earth_radius_km: float = 6371.0,
) -> float:
    """Return the great-circle distance between two GPS coordinates in km."""
    delta_latitude = radians(latitude_2 - latitude_1)
    delta_longitude = radians(longitude_2 - longitude_1)
    latitude_1 = radians(latitude_1)
    latitude_2 = radians(latitude_2)

    haversine = (
        sin(delta_latitude / 2) ** 2
        + cos(latitude_1) * cos(latitude_2) * sin(delta_longitude / 2) ** 2
    )
    return 2 * earth_radius_km * asin(sqrt(haversine))
EOF

cat > "$PROJECT_ROOT/src/data_processing.py" <<'EOF'
"""Data loading and cleaning helpers."""

import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a minimally cleaned copy of the input data."""
    cleaned_data = data.copy()
    cleaned_data = cleaned_data.drop_duplicates().dropna(how="all")
    return cleaned_data
EOF

cat > "$PROJECT_ROOT/src/features.py" <<'EOF'
"""Feature engineering helpers."""

import pandas as pd

from .utils import haversine_distance


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model features from a cleaned delivery dataset."""
    features = data.copy()
    coordinate_columns = {
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    }
    if coordinate_columns.issubset(features.columns):
        features["distance_km"] = features.apply(
            lambda row: haversine_distance(
                row["restaurant_latitude"],
                row["restaurant_longitude"],
                row["delivery_latitude"],
                row["delivery_longitude"],
            ),
            axis=1,
        )
    return features
EOF

cat > "$PROJECT_ROOT/src/predict.py" <<'EOF'
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
EOF

cat > "$PROJECT_ROOT/notebooks/01_data_cleaning.ipynb" <<'EOF'
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {"language": "markdown"},
      "source": ["# Data Cleaning\n", "Load and clean the raw delivery dataset."]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {"language": "python"},
      "outputs": [],
      "source": ["import pandas as pd\n", "\n", "# Add the raw data loading and cleaning steps here."]
    }
  ],
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
  "nbformat": 4,
  "nbformat_minor": 5
}
EOF

cat > "$PROJECT_ROOT/notebooks/02_eda_feature_engineering.ipynb" <<'EOF'
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {"language": "markdown"},
      "source": ["# EDA and Feature Engineering\n", "Explore distributions and create predictive features."]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {"language": "python"},
      "outputs": [],
      "source": ["import pandas as pd\n", "\n", "# Add exploratory analysis and feature engineering here."]
    }
  ],
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
  "nbformat": 4,
  "nbformat_minor": 5
}
EOF

cat > "$PROJECT_ROOT/notebooks/03_modeling_evaluation.ipynb" <<'EOF'
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {"language": "markdown"},
      "source": ["# Modeling and Evaluation\n", "Train, evaluate, and persist a delivery-time model."]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {"language": "python"},
      "outputs": [],
      "source": ["from sklearn.model_selection import train_test_split\n", "\n", "# Add model training and evaluation here."]
    }
  ],
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
  "nbformat": 4,
  "nbformat_minor": 5
}
EOF

cat > "$PROJECT_ROOT/app/streamlit_app.py" <<'EOF'
"""Streamlit interface for delivery-time predictions."""

import streamlit as st


st.set_page_config(page_title="Food Delivery Time Prediction")
st.title("Food Delivery Time Prediction")
st.sidebar.header("Prediction inputs")
st.write("Enter delivery features to generate a prediction.")
EOF

cat > "$PROJECT_ROOT/api/main.py" <<'EOF'
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
EOF

cat > "$PROJECT_ROOT/dags/ml_retraining_dag.py" <<'EOF'
"""Basic Airflow DAG for periodic model retraining."""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def retrain_model() -> None:
    """Placeholder for the model training pipeline."""
    print("Run data preparation, training, evaluation, and model registration.")


with DAG(
    dag_id="ml_retraining_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )
EOF

printf 'Project structure created in %s\n' "$PROJECT_ROOT"