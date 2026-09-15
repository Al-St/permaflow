"""Двухэтапный пайплайн: сырая телеметрия -> продукт для научной работы.

Этап 1 — кластеризация и регуляризация (``--stage cluster``)
    Загрузка сообщений и реестров, зона интереса, ручные исключения;
    ST-DBSCAN по устройствам -> кластеры = периоды прибора в одном месте;
    привязка кластеров к ближайшему БПН, сборка скважин
    ``TB-<код БПН>-<NN>``; регуляризация на единую сетку времени.  Продукты:
    ``RES_clustered.csv`` (поток сообщений, вход этапа 2), ``RES_regularized.csv``
    (регулярные ряды — на этом можно остановиться), ``wells_summary.csv``,
    ``wells.csv``, ``clusters.csv``, интерактивная карта привязки.

Этап 2 — фильтрация выбросов и восстановление пропусков (``--stage filter``)
    Читает ``RES_clustered.csv``, повторяет регуляризацию и очищает ряды:
    порог по уровню + скользящий и глобальный пороги Лапласа по вейвлет-невязке,
    выравнивание ступеней при смене поколений приборов, восстановление
    пропусков с лимитом по классу глубины (от 12 ч на суточных горизонтах
    до 30 суток на сезон-свободных).  Продукты: ``RES_clean.csv``,
    ``removal_rates.csv``, примеры «до/после».

Run: ``python -m permaflow --config configs/permaflow.yaml [--stage ...]``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import (assign_base_stations, build_wells, cluster_by_device,
                         cluster_table, unattached_clusters)
from .config import PipelineConfig, load_config
from .data_io import (drop_future_dates, load_base_stations, load_devices,
                      load_telemetry)
from .geo import zone_mask
from .outliers import (align_level_shifts, fill_short_gaps, level_outlier_mask,
                       pointwise_outlier_mask, rolling_outlier_mask,
                       split_by_observation_hour, wavelet_detrend)
from .prepare import (DEPTH_COLUMNS, drop_device_warmup, regularize_grid,
                      wide_temperature_table)
from .reporting import plot_site_map_interactive

logger = logging.getLogger(__name__)


# --- этап 1: кластеризация ---------------------------------------------------

def run_clustering(cfg: PipelineConfig) -> pd.DataFrame:
    out_dir: Path = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    logger.info("Загрузка телеметрии: %s", cfg.raw_data)
    raw = load_telemetry(cfg.raw_data)
    base_stations = load_base_stations(cfg.base_stations)
    devices = load_devices(cfg.devices, cfg.clustering.accuracy_split_m)
    logger.info("Сообщений: %d | БПН: %d | устройств: %d",
                len(raw), len(base_stations), len(devices))

    # --- зона интереса (правило магистерской: арктический пояс ИЛИ восточнее) -
    zone = cfg.clustering.zone
    raw = raw[zone_mask(raw["GNSSLatitude"], raw["GNSSLongitude"],
                        zone["min_lat"], zone["min_lon"],
                        zone["max_lat"], zone["max_lon"])]
    raw = raw.dropna(subset=["GNSSLatitude", "GNSSLongitude"])
    raw = drop_future_dates(raw, cfg.max_valid_date)

    # --- ручные исключения (чужие SIM и т.п.) ---------------------------------
    for exc in cfg.exclusions:
        cond = raw["IMEI"] == str(exc["imei"])
        if exc.get("start"):
            cond &= raw["DateTime"] >= pd.Timestamp(exc["start"])
        if exc.get("end"):
            cond &= raw["DateTime"] <= pd.Timestamp(exc["end"])
        logger.info("Ручное исключение: %d сообщений (IMEI %s) — %s",
                    int(cond.sum()), exc["imei"], exc.get("reason", ""))
        raw = raw[~cond]
    logger.info("После фильтров: %d сообщений", len(raw))

    # --- ST-DBSCAN по устройствам ---------------------------------------------
    labels = cluster_by_device(raw, devices, cfg.clustering.class_params())
    marked_all = raw.copy()
    marked_all["cluster"] = labels
    n_noise = int((labels == -1).sum())
    logger.info("Кластеров: %d | шум: %d сообщений (%.1f%%)",
                len(set(labels) - {-1}), n_noise, 100 * n_noise / max(len(labels), 1))

    # --- привязка к БПН и сборка скважин ---------------------------------------
    clusters = cluster_table(raw, labels)
    eps_by_class = {cls: params["eps_rad"] * 6371000.0 for cls, params in
                    cfg.clustering.class_params().items()}
    clusters["eps_m"] = clusters["IMEI"].map(_class_of_imei(devices)).map(
        lambda cls: eps_by_class.get(cls, np.nan))
    clusters = assign_base_stations(clusters, base_stations,
                                    cfg.clustering.max_attach_km)
    clusters, wells = build_wells(clusters)
    unattached = unattached_clusters(clusters)
    logger.info("Скважин: %d (БПН %d) | кластеров без привязки: %d (%d сообщений)",
                len(wells),
                wells["base station id"].nunique() if len(wells) else 0,
                len(unattached), int(unattached["messages"].sum()) if len(unattached) else 0)

    # --- разметка сообщений скважинами ----------------------------------------
    cluster_well = clusters.set_index("cluster")["well"]
    marked_all["well"] = marked_all["cluster"].map(cluster_well)
    clean = marked_all[marked_all["well"].notna()].copy()

    # --- регуляризация: единая сетка времени -----------------------------------
    product = clustered_product(clean)
    product.to_csv(out_dir / "RES_clustered.csv", index=False)
    wide = regularize(load_clustered_product_from(product), cfg.regularization)
    export_product(wide.drop(columns="cluster", errors="ignore"),
                   out_dir / "RES_regularized.csv")

    # --- сводные таблицы и карта ------------------------------------------------
    wells_report = _wells_report(wells, clean, clusters)
    wells_report.to_csv(out_dir / "wells_summary.csv", index=False)
    wells.to_csv(out_dir / "wells.csv", index=False)
    clusters.to_csv(out_dir / "clusters.csv", index=False)

    named = _well_map_points(wells_report)
    bp_points = _bp_map_points(base_stations, named)
    plot_site_map_interactive(named, bp_points, unattached,
                              _noise_points(marked_all),
                              fig_dir / "site_map_interactive.html")
    logger.info("Интерактивная карта: %s", fig_dir / "site_map_interactive.html")

    logger.info("Этап 1 готов. Результаты: %s", out_dir)
    return clean


def clustered_product(clean: pd.DataFrame) -> pd.DataFrame:
    """Поток сообщений, размеченный скважинами — вход этапа фильтрации.

    Формат: ``well id, DateTime, cluster, GNSSLatitude, GNSSLongitude,
    T1..T32``.  ``cluster`` (период устройства на скважине) нужен этапу 2
    для отбраковки переходного процесса прибора и выравнивания ступеней при
    смене поколений.  IMEI в продукт не входит — привязка приборов к
    скважинам живёт в wells.csv и clusters.csv; пустые служебные колонки
    выгрузки (T33 и далее) в продукт не идут.
    """
    export = clean.rename(columns={"well": "well id"})
    sensors = [f"T{i}" for i in range(1, 33) if f"T{i}" in export.columns]
    return export[["well id", "DateTime", "cluster", "GNSSLatitude",
                   "GNSSLongitude"] + sensors]


def _class_of_imei(devices: pd.DataFrame):
    class_map = devices.set_index("IMEI")["coord_class"]
    return lambda imei: class_map.get(imei, "gnss")


def _wells_report(wells: pd.DataFrame, clean: pd.DataFrame,
                  clusters: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица скважин.

    ``clean`` — поток сообщений, размеченный скважинами: дни измерений и
    пропуски считаются по датам сообщений, число измерений — по кластерам.
    """
    measurements = clusters.groupby("well")["messages"].sum()
    spans = clean.groupby("well")["DateTime"].agg(["min", "max"])
    days = clean.groupby("well")["DateTime"].apply(lambda s: s.dt.normalize().nunique())
    rows = []
    for _, well in wells.iterrows():
        well_id = well["TB id"]
        if well_id not in spans.index:
            continue
        days_with_data = int(days[well_id])
        span_days = int((spans.loc[well_id, "max"] - spans.loc[well_id, "min"]).days) + 1
        rows.append({
            "Base station name": well["Base station name"],
            "TB id": well_id,
            "latitude": well["latitude"],
            "longitude": well["longitude"],
            "distance to the base station": well["distance to the base station (km)"],
            "start date of measurements": spans.loc[well_id, "min"].date(),
            "number of measurements": int(measurements.get(well_id, 0)),
            "number of measurement days": days_with_data,
            "number of missing measurements (in days)": int(max(span_days - days_with_data, 0)),
        })
    return pd.DataFrame(rows)


def _well_map_points(wells_report: pd.DataFrame) -> pd.DataFrame:
    """Скважины в колонках, которых ждёт слой карты."""
    return wells_report.rename(columns={"latitude": "lat", "longitude": "lon"})[
        ["TB id", "Base station name", "lat", "lon",
         "distance to the base station", "number of measurements"]]


def _bp_map_points(base_stations: pd.DataFrame, named: pd.DataFrame) -> pd.DataFrame:
    """Base stations that have selected wells (for the map layer)."""
    bp = named.groupby("Base station name", as_index=False).size() \
        .rename(columns={"size": "n_boreholes"})
    bs = base_stations.dropna(subset=["lat", "lon"]).drop_duplicates("base station name")
    bp = bp.merge(bs[["base station name", "base station id", "lat", "lon"]],
                  left_on="Base station name", right_on="base station name", how="left")
    return bp.drop(columns="base station name").dropna(subset=["lat", "lon"])


def _noise_points(marked_all: pd.DataFrame, limit: int = 3000) -> pd.DataFrame:
    """Серые точки шума; не более ``limit`` случайных, чтобы не раздувать HTML."""
    noise = marked_all[marked_all["cluster"] == -1]
    if len(noise) > limit:
        noise = noise.sample(limit, random_state=0)
    return noise[["GNSSLatitude", "GNSSLongitude"]].rename(
        columns={"GNSSLatitude": "lat", "GNSSLongitude": "lon"})


# --- регуляризация (общая для обоих этапов) -----------------------------------

def load_clustered_product(path) -> pd.DataFrame:
    """Чтение продукта кластеризации: длинный формат сообщений.

    Колонки: ``well id, DateTime, cluster, GNSSLatitude, GNSSLongitude,
    T1..T32``.  Внутреннее имя скважины — ``site``.
    """
    long_df = pd.read_csv(path)
    long_df["DateTime"] = pd.to_datetime(long_df["DateTime"])
    return long_df.rename(columns={"well id": "site"})


def load_clustered_product_from(df: pd.DataFrame) -> pd.DataFrame:
    """То же без дисковой пересылки — внутри одного прогона."""
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    return out.rename(columns={"well id": "site"})


def regularize(long_df: pd.DataFrame, rc) -> pd.DataFrame:
    """Регуляризация: единая сетка времени и широкая таблица глубин."""
    warmed = drop_device_warmup(long_df, rc.device_warmup_hours)
    regular = regularize_grid(warmed, rc.default_freq)
    wide = wide_temperature_table(regular)
    logger.info("Регулярная таблица: %d строк, %d скважин",
                len(wide), wide["site"].nunique())
    return wide


def export_product(wide: pd.DataFrame, path: Path) -> None:
    """Экспорт регулярной таблицы в формате продукта: well id, Date, Time,
    GNSS, глубины."""
    export = wide.rename(columns={"site": "well id"})
    timestamps = pd.to_datetime(export.pop("DateTime"))
    export.insert(1, "Date", timestamps.dt.date)
    export.insert(2, "Time", timestamps.dt.time)
    export.to_csv(path, index=False)


# --- этап 2: фильтрация выбросов и восстановление пропусков -------------------

def run_filtering(cfg: PipelineConfig) -> pd.DataFrame:
    out_dir: Path = cfg.output_dir
    # этап 2 можно запускать независимо (свой filter_input) — выходная
    # директория может ещё не существовать
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for stale in fig_dir.glob("series_*.png"):   # карта этапа 1 не трогается
        stale.unlink()

    product = cfg.filter_input or (out_dir / "RES_clustered.csv")
    logger.info("Этап 2, вход: %s", product)
    wide = regularize(load_clustered_product(product), cfg.regularization)

    clean, removal = filter_outliers(wide, cfg.outliers)

    # идентификатор кластера — признак «смены прибора» для выравнивания
    # ступеней; align_level_shifts удаляет колонку после использования
    if "cluster" in wide.columns:
        clean = clean.merge(wide[["site", "DateTime", "cluster"]],
                            on=["site", "DateTime"], how="left")

    ls = cfg.level_shift
    clean = align_level_shifts(clean, depths_from=ls.depths_from_m,
                               min_step=ls.min_step_c, window_days=ls.window_days,
                               max_shifts=ls.max_shifts,
                               season_free_from=ls.season_free_from_m)

    clean = recover_gaps(clean, cfg.gap_fill)

    # координаты сообщений переносятся в продукт из регулярной таблицы
    # как есть, фильтрация их не трогает
    gnss = (wide[["site", "DateTime", "GNSSLatitude", "GNSSLongitude"]]
            .rename(columns={"site": "well id"}))
    gnss["DateTime"] = pd.to_datetime(gnss["DateTime"])

    export = clean.rename(columns={"site": "well id"})
    export["DateTime"] = pd.to_datetime(export["DateTime"])
    export = export.merge(gnss, on=["well id", "DateTime"], how="left")
    depth_cols = [c for c in export.columns if isinstance(c, float)]
    export = export[["well id", "DateTime", "GNSSLatitude", "GNSSLongitude"]
                    + depth_cols]
    export_product(export, out_dir / "RES_clean.csv")
    removal.to_csv(out_dir / "removal_rates.csv")

    for example in cfg.outliers.example_series:
        well = example["well"]
        depths = list(example.get("depths", (0.0, 0.4, 1.6, 10.0)))
        site_part = wide[wide["site"] == well].set_index("DateTime")[DEPTH_COLUMNS]
        clean_part = clean[clean["site"] == well].set_index("DateTime")[DEPTH_COLUMNS]
        if site_part.empty:
            logger.warning("Пример недоступен: скважины %s нет в данных", well)
            continue
        _plot_series(site_part, clean_part, well, depths,
                     fig_dir / f"series_{str(well).replace(' ', '_')}.png")

    logger.info("Этап 2 готов. Результаты: %s", out_dir)
    return clean


def recover_gaps(clean: pd.DataFrame, gf) -> pd.DataFrame:
    """Восстановление пропусков после фильтрации — по классам глубин.

    Допустимая длина непрерывного пропуска и метод зависят от масштаба
    изменчивости ряда на глубине (класс — по глубине колонки):

    * **суточный ход** (``z <= gf.diurnal_upto_m``): не более
      ``diurnal_limit`` интервалов сетки, линейно по времени — мост
      длиннее нескольких часов выдумывает суточную волну;
    * **годовой ход без суточного** (до ``gf.season_free_from_m``): не
      более ``seasonal_limit`` интервалов, интерполяция Akima — гладкая
      кривая встраивается в сезонный ход лучше прямой;
    * **сезон-свободные горизонты** (глубже): не более ``deep_limit``
      интервалов, мост по средним нескольких точек у границ пропуска —
      у границ ряд ступенчат и зашумлён, и прямая между одиночными
      точками переносила бы этот шум на весь мост.

    Пропуски длиннее лимита своего класса остаются NaN целиком; краевые
    пропуски не достраиваются — их восстановление выходит за рамки
    фильтрации.
    """
    parts = []
    for well, part in clean.groupby("site"):
        part = part.sort_values("DateTime").set_index("DateTime")
        depths = [c for c in part.columns if isinstance(c, float)]
        filled = part[depths].copy()
        for col in depths:
            z = float(col)
            limit = gf.limit_for_depth(z)
            if z >= gf.season_free_from_m:
                filled[col] = _fill_deep_by_boundaries(part[col], limit)
                continue
            method = "time" if z <= gf.diurnal_upto_m else "akima"
            # пропуски длиннее лимита класса возвращаются в NaN
            # (интерполяция без limit заполняет их с обоих концов)
            isna = part[col].isna()
            run_len = isna.groupby((isna != isna.shift()).cumsum()).transform("sum")
            interp = part[col].interpolate(method=method, limit_area="inside")
            filled[col] = interp.mask(isna & (run_len > limit))
        filled["site"] = well
        parts.append(filled.reset_index())
    return pd.concat(parts, ignore_index=True)


def _fill_deep_by_boundaries(series: pd.Series, limit: int,
                             edge: int = 4) -> pd.Series:
    """Мост по средним у границ пропуска (сезон-свободные горизонты).

    Уровни у границ — среднее ``edge`` соседних точек с каждой стороны
    (усреднение гасит шум и локальные ступени), мост — линейный ход по
    времени от левого уровня к правому.  Пропуски длиннее ``limit`` и
    краевые (нет валидных точек с обеих сторон) остаются NaN.
    """
    values = series.to_numpy(dtype=float)
    idx = series.index
    out = values.copy()
    n = len(values)
    i = 0
    while i < n:
        if not np.isnan(values[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(values[j]):
            j += 1
        run = j - i
        if 0 < i and j < n and run <= limit:
            left = float(np.nanmean(values[max(0, i - edge):i]))
            right = float(np.nanmean(values[j:min(n, j + edge)]))
            span = (idx[j] - idx[i - 1]).total_seconds()
            if span > 0:
                frac = (idx[i:j] - idx[i - 1]).total_seconds() / span
                out[i:j] = left + (right - left) * frac
        i = j
    return pd.Series(out, index=idx, name=series.name)


def _samples_per_day(hour_part: pd.DataFrame) -> int:
    idx = hour_part.index
    if len(idx) < 2:
        return 1
    return max(1, int(round(pd.Timedelta("1D") / np.median(np.diff(idx.values)))))


def filter_outliers(wide: pd.DataFrame, oc) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust three-stage filter per well and observation hour.

    Возвращает ``(очищенная таблица, доли удалённого по скважинам и глубинам
    в процентах)``.  Аномалии не сглаживаются: помеченные значения удаляются
    целиком, остальные сохраняют исходный уровень.
    """
    cleaned_parts, rates = [], []
    for well, part in wide.groupby("site"):
        series = part.set_index("DateTime")[DEPTH_COLUMNS]
        per_hour, masks = [], []
        for _, hour_part in split_by_observation_hour(series, oc.observation_hours):
            filled = fill_short_gaps(hour_part, oc.gap_fill_limit)
            spd = _samples_per_day(hour_part)
            detrended = filled.apply(lambda col: wavelet_detrend(col, oc.wavelet, oc.wavelet_level))
            mask = level_outlier_mask(filled, samples_per_day=spd,
                                      window_days=oc.window_days, alpha=oc.alpha,
                                      floor_c=oc.level_floor_c)
            mask = mask | rolling_outlier_mask(detrended,
                                        samples_per_day=spd,
                                        window_days=oc.window_days, alpha=oc.alpha,
                                        min_deviation=oc.min_deviation)
            mask = mask | pointwise_outlier_mask(detrended, alpha=oc.alpha,
                                                 min_deviation=oc.min_deviation)
            per_hour.append(hour_part.where(~mask))
            masks.append(mask)
        site_mask = pd.concat(masks).sort_index()
        site_clean = pd.concat(per_hour).sort_index()
        site_clean["site"] = well
        cleaned_parts.append(site_clean.reset_index())
        rates.append(site_mask.mean().to_frame(well).T * 100)
    clean = pd.concat(cleaned_parts, ignore_index=True)
    removal = pd.concat(rates).fillna(0).round(1)
    removal.index.name = "скважина"
    return clean, removal


def _plot_series(before: pd.DataFrame, after: pd.DataFrame, well: str,
                 depths: list[float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(depths), 1, figsize=(12, 2.2 * len(depths)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, z in zip(axes, depths):
        if z not in before.columns:
            continue
        ax.plot(before.index, before[z], ".", ms=2, color="0.6", label="исходный ряд")
        ax.plot(after.index, after[z], ".", ms=2, color="b", label="после фильтрации")
        ax.set_title(f"Скважина {well} — глубина {z} м", fontsize=9)
        ax.grid(True, alpha=0.3)
    # легенда над первым сабплотом, чтобы не перекрывать данные
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               ncol=2, fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- оркестрация ---------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> pd.DataFrame:
    """Оба этапа подряд: кластеризация с регуляризацией, затем фильтрация."""
    run_clustering(cfg)
    return run_filtering(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/permaflow.yaml")
    parser.add_argument("--stage", choices=["cluster", "filter", "all"],
                        default="all",
                        help="cluster — этап 1 (кластеризация и регуляризация), "
                             "filter — этап 2 (фильтрация выбросов), "
                             "all — оба этапа подряд")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.stage == "cluster":
        run_clustering(cfg)
    elif args.stage == "filter":
        run_filtering(cfg)
    else:
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
