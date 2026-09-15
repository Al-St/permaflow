"""Тесты ST-DBSCAN, привязки к БПН и сборки скважин."""

import numpy as np
import pandas as pd
import pytest

from permaflow.clustering import (assign_base_stations, build_wells,
                                  cluster_by_device, cluster_table, st_dbscan,
                                  unattached_clusters)


def make_devices():
    return pd.DataFrame([
        {"IMEI": "gnss_dev", "coordinate accuracy (m)": 30.0, "MinPts": 5,
         "coord_class": "gnss"},
        {"IMEI": "lbs_dev", "coordinate accuracy (m)": 85000.0, "MinPts": 3,
         "coord_class": "lbs"},
    ])


def make_raw():
    """GNSS-устройство на скв.1 (зима), то же место через год (скв.2?), и LBS рядом."""
    t0 = np.datetime64("2024-01-01")
    gnss1 = pd.DataFrame({
        "IMEI": "gnss_dev", "DateTime": t0 + np.arange(0, 60) * np.timedelta64(6, "h"),
        "GNSSLatitude": 66.5 + np.random.default_rng(0).normal(0, 2e-4, 60),
        "GNSSLongitude": 66.6 + np.random.default_rng(0).normal(0, 2e-4, 60)})
    gnss2 = pd.DataFrame({
        "IMEI": "gnss_dev", "DateTime": t0 + np.arange(400, 460) * np.timedelta64(6, "h"),
        "GNSSLatitude": 66.5005 + np.random.default_rng(1).normal(0, 2e-4, 60),
        "GNSSLongitude": 66.6005 + np.random.default_rng(1).normal(0, 2e-4, 60)})
    lbs = pd.DataFrame({
        "IMEI": "lbs_dev", "DateTime": t0 + np.arange(0, 40) * np.timedelta64(3, "h"),
        # вышка смещена на ~5 км от пункта: медиана облака не совпадает со скважиной
        "GNSSLatitude": 66.545 + np.random.default_rng(2).normal(0, 0.03, 40),
        "GNSSLongitude": 66.645 + np.random.default_rng(2).normal(0, 0.03, 40)})
    return pd.concat([gnss1, gnss2, lbs], ignore_index=True)


def make_base_stations():
    return pd.DataFrame([
        {"base station id": "24951", "base station name": "M-2 Test",
         "lat": 66.5, "lon": 66.6},
    ])


CLASS_PARAMS = {"gnss": {"eps_rad": 9.3e-06, "eps_t_days": 30.0, "min_samples": 5},
                "lbs": {"eps_rad": 0.003, "eps_t_days": 30.0, "min_samples": 3}}


def test_st_dbscan_splits_same_place_by_time_gap():
    t0 = np.datetime64("2024-01-01")
    times = np.concatenate([t0 + np.arange(0, 30, 1, dtype="timedelta64[D]"),
                            t0 + np.arange(300, 330, 1, dtype="timedelta64[D]")])
    coords = np.radians(np.column_stack([np.full(60, 60.0), np.full(60, 100.0)]))
    labels = st_dbscan(coords, times, eps_rad=0.001, eps_t_days=30, min_samples=5)
    # периоды разнесены на ~9 месяцев: одно место, два кластера
    assert len(set(labels)) == 2


def test_st_dbscan_min_samples_counts_the_point_itself():
    # 5 точек в одном месте и времени: у каждой 4 соседа + она сама = min_samples.
    # Канонический DBSCAN считает окрестность вместе с самой точкой.
    t0 = np.datetime64("2024-01-01")
    times = t0 + np.arange(5) * np.timedelta64(1, "h")
    coords = np.radians(np.column_stack([np.full(5, 60.0), np.full(5, 100.0)]))
    labels = st_dbscan(coords, times, eps_rad=0.001, eps_t_days=1, min_samples=5)
    assert len(set(labels) - {-1}) == 1


def test_st_dbscan_sparse_points_are_noise():
    # точки дальше eps_t друг от друга: ни одной плотной окрестности — весь шум
    t0 = np.datetime64("2024-01-01")
    times = t0 + np.arange(5) * np.timedelta64(40, "D")
    coords = np.radians(np.column_stack([np.full(5, 60.0), np.full(5, 100.0)]))
    labels = st_dbscan(coords, times, eps_rad=0.001, eps_t_days=30, min_samples=2)
    assert (labels == -1).all()


def test_cluster_by_device_separates_ids():
    labels = cluster_by_device(make_raw(), make_devices(), CLASS_PARAMS)
    df = make_raw().assign(cluster=labels)
    # GNSS-устройство: два периода на ~60 м друг от друга -> 2 кластера
    assert df[df["IMEI"] == "gnss_dev"]["cluster"].nunique() == 2
    # LBS-устройство: своё облако -> отдельный кластер, не смешивается с GNSS
    assert df[df["IMEI"] == "lbs_dev"]["cluster"].nunique() == 1
    assert set(df[df["IMEI"] == "gnss_dev"]["cluster"]).isdisjoint(
        set(df[df["IMEI"] == "lbs_dev"]["cluster"]))


def test_wells_merge_mutually_reachable_clusters():
    raw = make_raw()
    labels = cluster_by_device(raw, make_devices(), CLASS_PARAMS)
    clusters = cluster_table(raw, labels)
    eps_map = {"gnss_dev": 0.008, "lbs_dev": 0.008}  # ~50 км: взаимная близость
    clusters["eps_m"] = clusters["IMEI"].map(eps_map) * 6371000
    clusters = assign_base_stations(clusters, make_base_stations(), max_attach_km=50)
    clusters, wells = build_wells(clusters)
    assert len(wells) == 1                      # все кластеры — одна скважина
    assert wells.iloc[0]["TB id"] == "TB-24951-01"
    assert (clusters["well"] == "TB-24951-01").all()
    assert "distance to the base station (km)" in clusters.columns


def test_wells_split_on_one_sided_reachability():
    raw = make_raw()
    labels = cluster_by_device(raw, make_devices(), CLASS_PARAMS)
    clusters = cluster_table(raw, labels)
    # LBS-кластеру "до скважины" ~2 км, но GNSS-скважина с eps 30 м до него
    # не дотягивается -> взаимной близости нет -> две разные скважины
    eps_map = {"gnss_dev": 30.0, "lbs_dev": 20000.0}
    clusters["eps_m"] = clusters["IMEI"].map(eps_map)
    clusters = assign_base_stations(clusters, make_base_stations(), max_attach_km=50)
    clusters, wells = build_wells(clusters)
    assert len(wells) == 3                      # скв1, скв2, LBS-облако
    assert set(wells["TB id"]) == {"TB-24951-01", "TB-24951-02", "TB-24951-03"}


def test_registry_minpts_nan_falls_back_to_class():
    # пустой MinPts в реестре (или устройство без строки) не должен ронять
    # пайплайн: берётся значение класса точности
    devices = make_devices()
    devices.loc[devices["IMEI"] == "gnss_dev", "MinPts"] = np.nan
    labels = cluster_by_device(make_raw(), devices, CLASS_PARAMS)
    assert (labels >= 0).any()


def test_unattached_cluster_keeps_nearest_station():
    clusters = pd.DataFrame({
        "cluster": [1], "IMEI": ["gnss_dev"], "lat": [70.0], "lon": [100.0],
        "messages": [500],
        "start": [pd.Timestamp("2024-01-01")],
        "end": [pd.Timestamp("2024-12-01")], "eps_m": [60.0]})
    bs = pd.DataFrame([{"base station id": "24959",
                        "base station name": "OGMS Yakutsk",
                        "lat": 62.015, "lon": 129.668}])
    out = assign_base_stations(clusters, bs, max_attach_km=50)
    assert out["base station id"].isna().all()      # ~1300 км — без привязки
    assert out["nearest base station"].iloc[0] == "OGMS Yakutsk"
    assert out["nearest distance (km)"].iloc[0] > 50
    un = unattached_clusters(out)
    assert len(un) == 1
    assert {"nearest base station", "nearest distance (km)"} <= set(un.columns)
