"""Регуляризация кластеризованного продукта: единая сетка времени.

Продукт кластеризации — поток сообщений с номерами скважин; для работы с
рядами им нужна единая сетка времени и широкий формат глубин.  Колонка
``cluster`` (период прибора на скважине) сохраняется: по ней этап
фильтрации отбрасывает переходный процесс после установки прибора и
выравнивает ступени при смене поколений.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Датчики T1..T32 соответствуют глубинам 0.0–25.0 м (карта из реестра
# термокос).
DEPTH_MAP: dict[str, float] = {
    "T1": 0.0, "T2": 0.2, "T3": 0.4, "T4": 0.5, "T5": 0.6, "T6": 0.8,
    "T7": 1.0, "T8": 1.2, "T9": 1.5, "T10": 1.6, "T11": 2.0, "T12": 2.4,
    "T13": 2.5, "T14": 3.0, "T15": 3.2, "T16": 3.5, "T17": 4.0, "T18": 4.5,
    "T19": 5.0, "T20": 6.0, "T21": 7.0, "T22": 8.0, "T23": 9.0, "T24": 10.0,
    "T25": 12.0, "T26": 14.0, "T27": 16.0, "T28": 18.0, "T29": 20.0,
    "T30": 22.0, "T31": 24.0, "T32": 25.0,
}

DEPTH_COLUMNS: list[float] = sorted(DEPTH_MAP.values())


def drop_device_warmup(df: pd.DataFrame, hours: float = 12.0) -> pd.DataFrame:
    """Drop the first `hours` of each cluster's messages at a well.

    Сразу после установки термокосы датчики «выходят на режим»: первые часы
    показания дрейфуют (у нового прибора Якутска -8.55 -> -1.5 °C за три
    часа). Такие сообщения осредняются в интервалы сетки и искажают ряд.
    Кластер — это период работы одного устройства в одном месте, поэтому
    переходный процесс отбрасывается в начале каждого кластера.
    """
    group_cols = ["site"] + (["cluster"] if "cluster" in df.columns else [])
    parts = []
    for key, part in df.groupby(group_cols):
        start = part["DateTime"].min()
        parts.append(part[part["DateTime"] >= start + pd.Timedelta(hours=hours)])
    return pd.concat(parts, ignore_index=True)


def regularize_grid(df: pd.DataFrame, default_freq: str = "6h") -> pd.DataFrame:
    """Bin every well's messages onto a shared uniform time grid by averaging.

    Все скважины приводятся к одной сетке и одним срокам: сообщения
    усредняются внутри интервалов ``default_freq`` (стандартные сроки
    00/06/12/18 UTC для 6h).  Одиночные выбросы внутри интервала остаются
    во весь рост — усреднение применяется только к нескольким сообщениям
    одного интервала.  Пустые интервалы остаются явными NaN.

    Идентификатор кластера (периода прибора) переносится первым сообщением
    интервала: осреднять его нельзя, а вниз по пайплайну он нужен как
    признак «смены прибора» для выравнивания уровня.
    """
    pieces = []
    for site, part in df.groupby("site"):
        part = part.sort_values("DateTime").set_index("DateTime")
        part = part[~part.index.duplicated(keep="first")]
        numeric = part.drop(columns=["cluster"], errors="ignore") \
                      .select_dtypes(include=[np.number]).resample(default_freq).mean()
        if "cluster" in part.columns:
            numeric["cluster"] = part["cluster"].resample(default_freq).first()
        numeric["site"] = site
        pieces.append(numeric.reset_index().rename(columns={"index": "DateTime"}))
    return pd.concat(pieces, ignore_index=True)


def wide_temperature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rename T1..T32 sensor columns to depths (metres) and order them."""
    out = df.rename(columns=DEPTH_MAP)
    cols = [c for c in ("site", "DateTime", "cluster", "GNSSLatitude",
                        "GNSSLongitude") if c in out.columns]
    return out[cols + [z for z in DEPTH_COLUMNS if z in out.columns]]
