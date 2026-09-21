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
