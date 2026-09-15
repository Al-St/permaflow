"""Geodesic helpers used by clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

EARTH_RADIUS_KM = 6371.0


def to_radians(coords: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert a (lat, lon) degree frame/array to a radians numpy array."""
    return np.radians(np.asarray(coords, dtype=float))


def haversine_matrix(left_rad: np.ndarray, right_rad: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance matrix in kilometres."""
    return pairwise_distances(left_rad, right_rad, metric="haversine") * EARTH_RADIUS_KM


def zone_mask(lat: pd.Series, lon: pd.Series, min_lat: float = 65.0,
              min_lon: float = 50.0, max_lat: float = 90.0,
              max_lon: float = 360.0) -> pd.Series:
    """Permafrost-zone filter (thesis rule): keep a point if it lies in the
    arctic band **or** east of ``min_lon`` — an OR, not a rectangle.  A
    rectangle would drop high-arctic western sites (e.g. ГМО СП Баренцбург at
    14°E); the OR keeps them while still rejecting GNSS glitches far to the
    south-west.
    """
    lat_ok = (lat >= min_lat) & (lat <= max_lat)
    lon_ok = (lon >= min_lon) & (lon <= max_lon)
    return lat_ok | lon_ok
