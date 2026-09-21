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
