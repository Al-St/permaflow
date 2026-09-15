"""Loading of raw telemetry and the registries (base stations, devices)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Датчики T1..T32 соответствуют глубинам 0.0–25.0 м; карта глубин живёт в
# prepare.py — там же ряды переводятся в широкую таблицу глубин.


def load_base_stations(path: str | Path) -> pd.DataFrame:
    """Load the base-station registry (base_stations.csv).

    Ожидаемые колонки: base station id (5-значный индекс ВМО), base station
    name (английская транслитерация), latitude, longitude, timezone.
    """
    bs = pd.read_csv(path, dtype={"base station id": str})
    return bs.rename(columns={"latitude": "lat", "longitude": "lon"})


def load_devices(path: str | Path, accuracy_split_m: float = 1000.0) -> pd.DataFrame:
    """Load the device registry (devices.csv) and derive coordinate classes.

    Ожидаемые колонки: device id, coordinate accuracy (m), MinPts.
    Devices without GNSS are tower-positioned (LBS) with kilometre-scale
    coordinate errors; GNSS devices report tens of metres.  The class drives
    the device's spatial DBSCAN radius; MinPts is passed through from the
    registry (used as a fallback when empty).
    """
    dv = pd.read_csv(path, dtype={"device id": str})
    dv = dv.rename(columns={"device id": "IMEI"})
    dv = dv.drop_duplicates("IMEI").reset_index(drop=True)
    error = pd.to_numeric(dv["coordinate accuracy (m)"], errors="coerce")
    dv["coord_class"] = np.where(error > accuracy_split_m, "lbs", "gnss")
    dv["MinPts"] = pd.to_numeric(dv["MinPts"], errors="coerce")
    return dv


def load_telemetry(path: str | Path) -> pd.DataFrame:
    """Load raw device messages.

    Returns a frame with at least ``IMEI`` (str), ``DateTime`` (UTC-naive),
    ``GNSSLatitude``/``GNSSLongitude`` (float) and temperature columns.
    Depending on the export, the device identifier column is called ``IMEI``
    or ``M_ID`` — both are normalised to ``IMEI``.
    """
    df = pd.read_csv(path, low_memory=False)
    if "IMEI" not in df.columns and "M_ID" in df.columns:
        df = df.rename(columns={"M_ID": "IMEI"})
    df["IMEI"] = df["IMEI"].astype(str)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce", utc=True)
    df = df.dropna(subset=["DateTime"])
    df["DateTime"] = df["DateTime"].dt.tz_localize(None)
    return df


def drop_future_dates(df: pd.DataFrame, max_date: str = "2030-01-01") -> pd.DataFrame:
    """Remove obviously wrong future timestamps (device clock glitches)."""
    return df[df["DateTime"] <= pd.Timestamp(max_date)].copy()
