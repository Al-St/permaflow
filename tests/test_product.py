"""Контракт продукта RES_clustered.csv — входа этапа фильтрации."""

import numpy as np
import pandas as pd

from permaflow.pipeline import _wells_report, clustered_product


def test_clustered_product_contract():
    clean = pd.DataFrame({
        "IMEI": ["a", "b"],
        "DateTime": pd.to_datetime(["2024-01-01 00:10", "2024-01-02 06:10"]),
        "cluster": [7, 9],
        "GNSSLatitude": [66.5, 66.6],
        "GNSSLongitude": [66.6, 66.7],
        "well": ["TB-24959-01", "TB-24959-01"],
        "T1": [1.0, 1.1],
        "T32": [-2.0, -2.1],
        "T33": [np.nan, np.nan],  # пустые хвостовые колонки выгрузки
    })
    product = clustered_product(clean)
    # точно оговорённый набор и порядок колонок; IMEI и T33+ не входят
    assert product.columns.tolist() == [
        "well id", "DateTime", "cluster", "GNSSLatitude", "GNSSLongitude",
        "T1", "T32"]
    assert set(product["well id"]) == {"TB-24959-01"}
    assert product["cluster"].tolist() == [7, 9]


def test_wells_report_counts_calendar_days():
    wells = pd.DataFrame([{
        "Base station name": "M-2 Test", "base station id": "24959",
        "TB id": "TB-24959-01", "latitude": 66.5, "longitude": 66.6,
        "distance to the base station (km)": 1.2, "clusters merged": 1,
        "messages": 3}])
    clean = pd.DataFrame({
        "well": ["TB-24959-01"] * 3,
        "DateTime": pd.to_datetime(
            ["2024-01-01 06:00", "2024-01-01 18:00", "2024-01-04 06:00"])})
    clusters = pd.DataFrame({"well": ["TB-24959-01"], "messages": [3]})
    row = _wells_report(wells, clean, clusters).iloc[0]
    assert row["number of measurements"] == 3
    assert row["number of measurement days"] == 2        # 1 и 4 января
    assert row["number of missing measurements (in days)"] == 2  # охват 4 дня
    assert str(row["start date of measurements"]) == "2024-01-01"
