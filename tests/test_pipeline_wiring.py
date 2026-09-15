"""Тесты обвязки пайплайна: загрузчики, конфиг, экспорт и этап 2 целиком."""

import numpy as np
import pandas as pd
import pytest

from permaflow.config import OutliersConfig, load_config
from permaflow.data_io import (load_base_stations, load_devices,
                               load_telemetry)
from permaflow.pipeline import (export_product, filter_outliers,
                                run_filtering)
from permaflow.prepare import DEPTH_COLUMNS

WELL = "TB-99999-01"


# --- загрузка реестров и телеметрии -------------------------------------------

def test_load_telemetry_normalises_mid_and_drops_bad_dates(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({
        "M_ID": ["111", "111", "222"],
        "DateTime": ["2024-01-01 00:00", "не дата", "2024-06-01 12:00"],
        "GNSSLatitude": [66.5, 66.5, 70.0],
        "GNSSLongitude": [66.6, 66.6, 100.0],
        "T1": [-1.0, -1.1, -2.0],
    }).to_csv(src, index=False)
    df = load_telemetry(src)
    # выгрузка с M_ID нормализуется к IMEI; сообщение с битой датой удалено
    assert "IMEI" in df.columns and "M_ID" not in df.columns
    assert df["IMEI"].tolist() == ["111", "222"]
    assert df["DateTime"].dt.tz is None


def test_load_devices_splits_classes_by_accuracy(tmp_path):
    src = tmp_path / "dev.csv"
    pd.DataFrame({
        "device id": ["a", "b", "a", "c"],
        "coordinate accuracy (m)": [30, 85000, 30, 500],
        "MinPts": [120, 13, 120, ""],
    }).to_csv(src, index=False)
    dv = load_devices(src, accuracy_split_m=1000)
    assert dv["coord_class"].tolist() == ["gnss", "lbs", "gnss"]  # дубль удалён
    assert dv["MinPts"].isna().tolist() == [False, False, True]


def test_load_base_stations_renames_coords(tmp_path):
    src = tmp_path / "bs.csv"
    pd.DataFrame({
        "base station id": ["01234"], "base station name": ["OGMS Yakutsk"],
        "latitude": [62.015], "longitude": [129.668], "timezone": [3],
    }).to_csv(src, index=False)
    bs = load_base_stations(src)
    assert bs.columns.tolist() == ["base station id", "base station name",
                                   "lat", "lon", "timezone"]
    # 5-значный код ВМО читается строкой: ведущий ноль не теряется
    assert bs["base station id"].tolist() == ["01234"]


# --- конфигурация ---------------------------------------------------------------

def test_load_config_maps_yaml_to_dataclasses(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(f"""
raw_data: {tmp_path}/raw.csv
base_stations: {tmp_path}/bs.csv
devices: {tmp_path}/dev.csv
output_dir: {tmp_path}/out
filter_input: "{tmp_path}/RES_clustered.csv"
clustering:
  eps_t_days: 15
outliers:
  observation_hours: [0, 12]
""", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.clustering.eps_t_days == 15
    assert cfg.clustering.min_samples_gnss == 120     # незаданное — дефолт
    assert cfg.filter_input == tmp_path / "RES_clustered.csv"
    assert cfg.outliers.observation_hours == (0, 12)  # список -> кортеж


def test_load_config_filter_input_null_is_none(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(f"""
raw_data: {tmp_path}/raw.csv
base_stations: {tmp_path}/bs.csv
devices: {tmp_path}/dev.csv
output_dir: {tmp_path}/out
""", encoding="utf-8")
    assert load_config(cfg_file).filter_input is None


def test_load_config_rejects_unknown_key(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(f"""
raw_data: {tmp_path}/raw.csv
base_stations: {tmp_path}/bs.csv
devices: {tmp_path}/dev.csv
output_dir: {tmp_path}/out
clustering:
  eps_t_dayz: 15
""", encoding="utf-8")
    # опечатка в ключе должна вскрываться сразу, а не в рантайме этапа
    with pytest.raises(TypeError):
        load_config(cfg_file)


# --- экспорт и фильтрация -------------------------------------------------------

def test_export_product_date_time_format(tmp_path):
    wide = pd.DataFrame({
        "site": [WELL] * 2,
        "DateTime": pd.to_datetime(["2024-01-01 06:00", "2024-01-01 12:00"]),
        0.0: [-1.0, -1.1], 5.0: [-3.0, -3.1]})
    export_product(wide, tmp_path / "out.csv")
    df = pd.read_csv(tmp_path / "out.csv")
    assert df.columns.tolist() == ["well id", "Date", "Time", "0.0", "5.0"]
    assert df["Date"].tolist() == ["2024-01-01", "2024-01-01"]
    assert df["Time"].tolist() == ["06:00:00", "12:00:00"]


def test_filter_outliers_removes_spike_and_reports_rate():
    idx = pd.date_range("2024-01-01", periods=200, freq="6h")
    rng = np.random.default_rng(3)
    shallow = (-2.0 + 3 * np.sin(2 * np.pi * np.arange(200) / 1460)
               + rng.laplace(0, 0.05, 200))
    deep = np.full(200, -3.0) + rng.laplace(0, 0.02, 200)
    shallow[100] = 40.0                      # одиночный сбой на 0.0 м
    wide = pd.DataFrame({"site": WELL, "DateTime": idx,
                         0.0: shallow, 5.0: deep})
    # контракт RES_clustered: все 32 датчика; пустые горизонты — NaN
    for z in DEPTH_COLUMNS:
        wide[z] = wide.get(z, np.nan)
    clean, removal = filter_outliers(wide, OutliersConfig())
    assert clean["site"].unique().tolist() == [WELL]
    assert np.isnan(clean.loc[clean["DateTime"] == idx[100], 0.0].iloc[0])
    assert clean.loc[clean["DateTime"] == idx[100], 5.0].iloc[0] == pytest.approx(deep[100])
    assert removal.loc[WELL, 0.0] > 0.0      # доля удалённых, %
    assert removal.loc[WELL, 5.0] == 0.0
    assert removal.index.name == "скважина"


# --- этап 2 целиком (run_filtering) ----------------------------------------------

def _make_clustered_product(path):
    """Мини-выгрузка: одна скважина, один кластер, сбой на 0.0 м."""
    hours = pd.date_range("2024-01-01", periods=320, freq="3h")
    rng = np.random.default_rng(7)
    t = np.arange(320)
    t1 = (-5.0 + 3 * np.sin(2 * np.pi * t / 2920)      # годовой ход, 3 ч/сообщение
          + rng.laplace(0, 0.05, 320))
    t19 = np.full(320, -3.0) + rng.laplace(0, 0.02, 320)
    t1[150] = 40.0
    data = {
        "well id": WELL,
        "DateTime": hours.strftime("%Y-%m-%d %H:%M:%S"),
        "cluster": 1,
        "GNSSLatitude": 66.5,
        "GNSSLongitude": 66.6,
        "T1": t1,
        "T19": t19,
    }
    # контракт продукта: колонки T1..T32 на месте (пустые горизонты — NaN)
    data.update({f"T{i}": np.nan for i in range(1, 33) if f"T{i}" not in data})
    pd.DataFrame(data).to_csv(path, index=False)


def _write_filter_config(tmp_path, out_dir, product):
    cfg_file = tmp_path / "filter.yaml"
    cfg_file.write_text(f"""
raw_data: {tmp_path}/raw.csv
base_stations: {tmp_path}/bs.csv
devices: {tmp_path}/dev.csv
output_dir: "{out_dir}"
filter_input: "{product}"
regularization:
  default_freq: 6h
  device_warmup_hours: 12
outliers:
  example_series:
    - well: "{WELL}"
      depths: [0.0, 5.0]
gap_fill:
  diurnal_upto_m: 1.0
  diurnal_limit: 2
  seasonal_limit: 40
  season_free_from_m: 12.0
  deep_limit: 120
""", encoding="utf-8")
    return cfg_file


def test_run_filtering_creates_missing_output_dir(tmp_path):
    # регрессия: этап 2 заявлен перезапускаемым независимо от этапа 1 —
    # выходная директория создаётся сама, а не падает FileNotFoundError
    product = tmp_path / "RES_clustered.csv"
    _make_clustered_product(product)
    out_dir = tmp_path / "fresh" / "out"               # ещё не существует
    cfg = load_config(_write_filter_config(tmp_path, out_dir, product))
    clean = run_filtering(cfg)
    assert clean["site"].eq(WELL).all()
    assert (out_dir / "RES_clean.csv").exists()
    assert (out_dir / "removal_rates.csv").exists()
    assert list((out_dir / "figures").glob("series_*.png"))


def test_run_filtering_product_contract_and_stale_figures(tmp_path):
    product = tmp_path / "RES_clustered.csv"
    _make_clustered_product(product)
    out_dir = tmp_path / "out"
    (out_dir / "figures").mkdir(parents=True)
    stale = out_dir / "figures" / "series_OLD.png"
    stale.write_bytes(b"old")
    cfg = load_config(_write_filter_config(tmp_path, out_dir, product))
    run_filtering(cfg)

    assert not stale.exists()                          # устаревшие примеры убраны

    df = pd.read_csv(out_dir / "RES_clean.csv")
    assert df.columns[:5].tolist() == ["well id", "Date", "Time",
                                       "GNSSLatitude", "GNSSLongitude"]
    assert "0.0" in df.columns and "5.0" in df.columns  # T1/T19 -> глубины
    # сбой удалён и залит интерполяцией: ряд без 40-градусных отсчётов
    assert df["0.0"].dropna().abs().max() < 15.0

    removal = pd.read_csv(out_dir / "removal_rates.csv", index_col=0)
    assert removal.loc[WELL, "0.0"] > 0.0
    assert removal.loc[WELL, "5.0"] == 0.0
