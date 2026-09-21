"""Data loading and cleaning helpers."""

import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a minimally cleaned copy of the input data."""
    cleaned_data = data.copy()
    cleaned_data = cleaned_data.drop_duplicates().dropna(how="all")
    return cleaned_data
