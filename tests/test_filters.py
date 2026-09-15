"""Санити-фильтры сырых данных: зона интереса и будущие даты."""

import pandas as pd

from permaflow.data_io import drop_future_dates
from permaflow.geo import zone_mask


def test_zone_mask_is_or_not_a_rectangle():
    lat = pd.Series([66.0, 60.0, 70.0, 60.0])
    lon = pd.Series([40.0, 60.0, 14.0, 40.0])
    mask = zone_mask(lat, lon, min_lat=65, min_lon=50, max_lat=90, max_lon=360)
    # (66,40) — арктический пояс; (60,60) — восточнее 50°; (70,14) —
    # высокоширотный запад (Баренцбург, выпадал из прямоугольника); (60,40) — мимо
    assert mask.tolist() == [True, True, True, False]


def test_drop_future_dates():
    df = pd.DataFrame({"DateTime": pd.to_datetime(
        ["2024-06-01", "2033-01-01", "2025-01-01"])})
    kept = drop_future_dates(df, max_date="2030-01-01")
    assert kept["DateTime"].dt.year.tolist() == [2024, 2025]
