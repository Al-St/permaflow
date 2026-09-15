"""Pipeline configuration (loaded from a YAML file)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ClusteringConfig:
    # Раздельная кластеризация по классам точности координат: выше порога —
    # LBS (вышки связи, км), ниже — GNSS (метры).
    accuracy_split_m: float = 1000.0
    # Пространственные радиусы DBSCAN по классам (радианы дуги): GNSS ~60 м,
    # LBS ~19 км (значения из магистерской).
    eps_rad_gnss: float = 9.3e-06
    eps_rad_lbs: float = 0.003
    # Временной радиус ST-DBSCAN: сообщения устройства, разнесённые сильнее,
    # не попадают в одну кластерную окрестность.
    eps_t_days: float = 30.0
    # MinPts по умолчанию по классам; для конкретного устройства берётся
    # значение MinPts из реестра devices, если оно заполнено.
    min_samples_gnss: int = 120
    min_samples_lbs: int = 13
    # Максимальное расстояние кластера до ближайшего БПН, км; дальше кластер
    # остаётся без привязки (UNIDENTIFIED, показывается на карте красным).
    max_attach_km: float = 50.0
    zone: dict = field(default_factory=lambda: {"min_lat": 65.0, "min_lon": 50.0,
                                                "max_lat": 90.0, "max_lon": 360.0})

    def class_params(self) -> dict:
        return {
            "gnss": {"eps_rad": self.eps_rad_gnss,
                     "eps_t_days": self.eps_t_days,
                     "min_samples": self.min_samples_gnss},
            "lbs": {"eps_rad": self.eps_rad_lbs,
                    "eps_t_days": self.eps_t_days,
                    "min_samples": self.min_samples_lbs},
        }


@dataclass
class RegularizationConfig:
    default_freq: str = "6h"          # единая сетка и единые сроки для всех
    # Переходный процесс нового прибора: часы после первого сообщения
    # кластера на скважине отбрасываются (дрейф датчиков после установки).
    device_warmup_hours: float = 12.0


@dataclass
class OutliersConfig:
    window_days: int = 30       # ширина скользящего окна, сутки
    alpha: float = 0.01         # уровень квантильного коридора Лапласа
    wavelet: str = "bior3.1"
    wavelet_level: int = 1
    gap_fill_limit: int = 12
    observation_hours: tuple[int, ...] | None = None  # None = все сроки
    min_deviation: float = 0.1  # нижний порог "грубого сбоя" невязки, °C
                                # (точность датчиков ~0.1 °C; отсекает только
                                #  заведомые артефакты, но не шаги квантования)
    level_floor_c: float = 25.0 # пол порога по уровню, °C: серийный сбой должен
                                # отклоняться от уровня месяца сильнее любого
                                # реального синоптического спада на 0.0 м
    example_series: list = field(default_factory=list)  # [{well, depths}]


@dataclass
class LevelShiftConfig:
    # Выравнивание сдвигов уровня между поколениями приборов на глубоких
    # горизонтах: естественная скорость изменения температуры там мала,
    # поэтому устойчивый сдвиг медианы > min_step_c — приборный артефакт.
    depths_from_m: float = 5.0
    min_step_c: float = 0.2
    window_days: int = 14
    max_shifts: int = 5
    # Сезон-свободные горизонты (годовой ход < min_step_c): только по ним
    # подтверждается и оценивается ступень — через пропуск связи длиной в
    # долю сезона естественный ход на мелких горизонтах неотличим от
    # приборной ступени.
    season_free_from_m: float = 12.0


@dataclass
class GapFillConfig:
    # Восстановление пропусков после фильтрации. Допустимая длина
    # непрерывного пропуска зависит от масштаба изменчивости ряда на
    # глубине; лимиты заданы интервалами сетки (при 6 ч: 2 = 12 ч,
    # 40 = 10 сут, 120 = 30 сут).  Пропуски длиннее лимита своего класса
    # остаются NaN — их восстановление не входит в задачу фильтрации.
    #
    # Суточный ход (глубины до diurnal_upto_m): мост длиннее пары
    # интервалов выдумывает суточную волну — допускаются только
    # единичные пропуски, интерполяция линейная по времени.
    diurnal_upto_m: float = 1.0
    diurnal_limit: int = 2
    # Годовой ход без суточного (до season_free_from_m): Akima следует за
    # плавным сезонным ходом лучше прямой.
    seasonal_limit: int = 40
    # Сезон-свободные горизонты (глубже season_free_from_m, годовой ход
    # < 0.2 °C — как в level_shift): уровень меняется шагами и шумом,
    # поэтому мост строится по средним нескольких точек у границ пропуска.
    season_free_from_m: float = 12.0
    deep_limit: int = 120

    def limit_for_depth(self, depth: float) -> int:
        """Допустимая длина пропуска (интервалов сетки) для глубины."""
        if depth <= self.diurnal_upto_m:
            return self.diurnal_limit
        if depth < self.season_free_from_m:
            return self.seasonal_limit
        return self.deep_limit


@dataclass
class PipelineConfig:
    raw_data: Path
    base_stations: Path
    devices: Path
    output_dir: Path
    # Отсечение заведомо неверных будущих дат (сбой часов прибора).
    max_valid_date: str = "2030-01-01"
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    # Ручные исключения: [{imei, reason, start?, end?}]
    exclusions: list = field(default_factory=list)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    # Вход этапа фильтрации; None = output_dir/RES_clustered.csv (продукт
    # этапа кластеризации этого же прогона или прошлого запуска).
    filter_input: Path | None = None
    outliers: OutliersConfig = field(default_factory=OutliersConfig)
    level_shift: LevelShiftConfig = field(default_factory=LevelShiftConfig)
    gap_fill: GapFillConfig = field(default_factory=GapFillConfig)


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    outliers = OutliersConfig(**(raw.get("outliers") or {}))
    if outliers.observation_hours is not None:
        outliers.observation_hours = tuple(outliers.observation_hours)
    for example in outliers.example_series:
        example["depths"] = tuple(example.get("depths", (0.0, 0.4, 1.6, 10.0)))
    return PipelineConfig(
        raw_data=Path(raw["raw_data"]),
        base_stations=Path(raw["base_stations"]),
        devices=Path(raw["devices"]),
        output_dir=Path(raw["output_dir"]),
        max_valid_date=str(raw.get("max_valid_date", "2030-01-01")),
        clustering=ClusteringConfig(**(raw.get("clustering") or {})),
        exclusions=list(raw.get("exclusions") or []),
        regularization=RegularizationConfig(**(raw.get("regularization") or {})),
        filter_input=(Path(raw["filter_input"])
                      if raw.get("filter_input") else None),
        outliers=outliers,
        level_shift=LevelShiftConfig(**(raw.get("level_shift") or {})),
        gap_fill=GapFillConfig(**(raw.get("gap_fill") or {})),
    )
