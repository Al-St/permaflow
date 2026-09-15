"""Outlier detection in ground-temperature series.

Temperature series of permafrost boreholes are non-stationary (seasonal cycle
on top of a trend) and heavy-tailed even after detrending, so classical 3-sigma
rules fail.  The residual gates (developed for the master's thesis):

1. **Wavelet detrending** — remove the seasonal component and trend with a
   one-level discrete wavelet decomposition (bior3.1): the residual contains
   only the "weather-scale" variability.
2. **Robust standardisation** — centre by the median and scale by the median
   absolute deviation (MAD) of the residual.
3. **Laplace quantile gate** — the residual distribution is well approximated
   by a Laplace law; outliers are values outside the central
   ``1 - alpha`` quantile interval of that law.
4. **Rolling application** — steps 2-3 run in a centred moving window (with
   explicit edge windows) so slow drifts cannot hide inside the thresholds.

Serial glitches that the detrend absorbs into its approximation (sensor
saturation rails, stuck channels) do not show up in the residual and are
caught by a separate level gate; persistent steps from device replacement
are handled by level-shift alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from scipy.stats import laplace


def fill_short_gaps(df: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    """Bridge short gaps with Akima interpolation (for the wavelet transform).

    ``limit_direction="both"`` fills up to ``limit`` consecutive NaN from
    each end of a gap, so runs of up to ``2 * limit`` samples are bridged
    entirely; longer runs keep their NaN core.  The wavelet decomposition
    needs a gap-free series, but only short gaps may be bridged: long data
    losses stay NaN and are excluded from statistics.  The filled values
    are used for detrending only — outlier masks keep the original NaN
    positions, and the filtered product is cut from the raw series.
    Columns with fewer than two observations cannot be interpolated and are
    returned unchanged.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].notna().sum() >= 2:
            out[col] = out[col].interpolate(method="akima", limit_direction="both",
                                            limit=limit)
    return out


def wavelet_detrend(series: pd.Series, wavelet: str = "bior3.1",
                    level: int = 1) -> pd.Series:
    """Subtract the low-frequency wavelet approximation from the series.

    NaN positions cannot pass through the transform, so they are temporarily
    bridged by linear interpolation and restored afterwards.
    """
    valid = series.notna()
    filled = series.interpolate(method="linear", limit_direction="both")
    coeffs = pywt.wavedec(filled, wavelet=wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    approx = pywt.waverec(coeffs, wavelet=wavelet)[:len(series)]
    resid = series - pd.Series(approx, index=series.index, name=series.name)
    return resid.where(valid)


def laplace_bounds(values: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """Central quantile interval of the Laplace law fitted via MAD."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        # degenerate window (e.g. flat sensor output): MAD carries no scale
        # information, fall back to the mean absolute deviation
        mad = np.mean(np.abs(values - median))
    scale = mad / np.log(2)  # MAD -> Laplace scale b
    if scale == 0:
        # constant window: the only admissible value is the median itself
        return float(median), float(median)
    left = laplace.ppf(alpha / 2, loc=median, scale=scale)
    right = laplace.ppf(1 - alpha / 2, loc=median, scale=scale)
    return float(left), float(right)


def pointwise_outlier_mask(df: pd.DataFrame, alpha: float = 0.01,
                           min_deviation: float = 0.0) -> pd.DataFrame:
    """Global (whole-series) robust outlier mask, same shape as ``df``.

    A point is flagged only when it is beyond the Laplace quantile interval
    **and** deviates from the series median by more than ``min_deviation``
    (°C).  The second condition matters for flat deep layers, where sensors
    report quantized values and the fitted scale collapses to hundredths of a
    degree: without the floor the gate flags the sensor's own quantization
    steps instead of gross glitches.
    """
    out = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        values = df[col].dropna().to_numpy()
        if values.size < 3:
            continue
        left, right = laplace_bounds(values, alpha)
        median = np.median(values)
        deviates = (df[col] - median).abs() > min_deviation
        out[col] = ((df[col] < left) | (df[col] > right)) & deviates
    return out


def rolling_outlier_mask(df: pd.DataFrame, samples_per_day: int,
                         window_days: int = 30, alpha: float = 0.01,
                         min_valid: int | None = None,
                         min_deviation: float = 0.0) -> pd.DataFrame:
    """Rolling robust outlier mask with explicit windows at both edges.

    ``pandas`` rolling with ``closed='both'`` covers the centred part; the
    first and last ``window_days`` are scored against statistics of the
    nearest full window (as in the original thesis implementation).

    The gate needs ``min_valid`` (by default a third of the window, but at
    least 8) valid points instead of the full window: sites transmit with
    long gaps, and requiring the full window silently disabled filtering in
    every gappy stretch — exactly where glitches concentrate.

    ``min_deviation`` (°C) is the same gross-glitch floor as in the global
    gate: on flat quantized deep layers the fitted scale collapses and the
    quantile test alone would flag the sensor's own quantization steps.
    """
    window = samples_per_day * window_days
    if window < 8:
        raise ValueError("Rolling window is too short: increase window_days")
    # pandas rolling with an even window and closed="both" produces an
    # asymmetric window whose midpoint is the right neighbour: every flag
    # would land one position off.  An odd window is strictly centred.
    if window % 2 == 0:
        window += 1
    if min_valid is None:
        min_valid = max(8, window // 3)

    def _gate(chunk: np.ndarray) -> float:
        center = chunk[len(chunk) // 2]
        if np.isnan(center):
            return 0.0
        finite = chunk[~np.isnan(chunk)]
        if finite.size < min_valid:
            return 0.0
        left, right = laplace_bounds(finite, alpha)
        if abs(center - np.median(finite)) <= min_deviation:
            return 0.0
        return float(center < left or center > right)

    out = (df.rolling(window, center=True, min_periods=min_valid, closed="both")
           .apply(_gate, raw=True).fillna(0.0).astype(bool))

    # At the series edges pandas windows are left/right-clipped (not centred),
    # so the midpoint trick above evaluates the wrong position.  Re-score the
    # edge zones explicitly against the statistics of the nearest full window.
    for col in df.columns:
        head = df[col].iloc[:window]
        tail = df[col].iloc[-window:]
        for segment in (head, tail):
            values = segment.dropna().to_numpy()
            if values.size < 3:
                continue
            left, right = laplace_bounds(values, alpha)
            median = np.median(values)
            deviates = (segment - median).abs() > min_deviation
            beyond = (segment < left) | (segment > right)
            out.loc[segment.index, col] = (beyond & deviates).fillna(False)
    return out


def level_outlier_mask(df: pd.DataFrame, samples_per_day: int,
                       window_days: int = 30, alpha: float = 0.01,
                       floor_c: float = 25.0) -> pd.DataFrame:
    """Гейт по уровню: значение само по себе далеко от уровня ряда.

    Ловит серийные сбои (насыщение датчика на рейке +41.9 °C, «залипание»
    канала на константе до -60 °C неделями), которые вейвлет-детренд
    поглощает в аппроксимацию — невязка таких точек мала, и пороги по
    невязке их пропускают. Три проверки покрывают разную долю сбойных
    точек в окне:

    - **локальная**: отклонение от скользящей медианы (справляется, пока
      сбой занимает меньше половины окна);
    - **глобальная**: отклонение от медианы всего ряда (ловит эпизоды
      длиннее окна, когда локальная медиана сама отравлена);
    - **серии одинаковых значений**: подряд идущие повторяющиеся показания,
      далёкие от уровня ряда, — сигнатура «залипшего» канала (реальные
      плоские глубокие ряды повторяют значения, но на уровне самого ряда).

    Порог локальной и глобальной проверок — ``max(floor_c, адаптивная
    граница Лапласа)``: пол ``floor_c`` (25 °C) отсекает всё физически
    возможное (даже в Якутии спад температуры грунта на 0.0 м не превосходит
    ~20 °C от уровня месяца), адаптивная граница подстраивается под окна с
    большой естественной изменчивостью (сезонный ход на мелких глубинах).
    Проверка серий одинаковых значений использует только фиксированный пол
    ``floor_c``: «залипший» канал отличается от уровня ряда на десятки
    градусов, и локальная ширина окна никак не меняет эту дистанцию.
    """
    window = samples_per_day * window_days
    if window % 2 == 0:
        window += 1
    min_valid = max(8, window // 3)
    out = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        valid = values[~np.isnan(values)]
        if valid.size < 3:
            continue
        med = float(np.median(valid))
        q25, q75 = np.quantile(valid, [0.25, 0.75])
        scale = (q75 - q25) / (2.0 * np.log(2.0))   # IQR -> масштаб Лапласа
        bound = max(floor_c, float(laplace.ppf(1 - alpha / 2,
                                               scale=max(scale, 1e-9))))
        global_mask = np.abs(values - med) > bound
        stuck_mask = _stuck_run_mask(values, med, floor=floor_c)
        out[col] = global_mask | stuck_mask

    def _gate(chunk: np.ndarray) -> float:
        center = chunk[len(chunk) // 2]
        if np.isnan(center):
            return 0.0
        finite = chunk[~np.isnan(chunk)]
        if finite.size < min_valid:
            return 0.0
        med = np.median(finite)
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        scale = (q75 - q25) / (2.0 * np.log(2.0))
        bound = max(floor_c, laplace.ppf(1 - alpha / 2, scale=max(scale, 1e-9)))
        return float(abs(center - med) > bound)

    local = (df.rolling(window, center=True, min_periods=min_valid, closed="both")
             .apply(_gate, raw=True).fillna(0.0).astype(bool))
    return out | local


def _stuck_run_mask(values: np.ndarray, level: float, floor: float,
                    tol: float = 0.01, min_run: int = 2) -> np.ndarray:
    """Помечает серии подряд повторяющихся значений вдали от уровня ряда."""
    flags = np.zeros(len(values), dtype=bool)
    rounded = np.round(values, 2)
    n = len(values)
    i = 0
    while i < n:
        if np.isnan(values[i]):
            i += 1
            continue
        j = i
        while (j + 1 < n and not np.isnan(values[j + 1])
               and abs(rounded[j + 1] - rounded[i]) <= tol):
            j += 1
        if j - i + 1 >= min_run and abs(values[i] - level) > floor:
            flags[i:j + 1] = True
        i = j + 1
    return flags


def split_by_observation_hour(df: pd.DataFrame,
                              hours: tuple[int, ...] | None = None):
    """Yield ``(hour, subframe)`` slices of a DateTime-indexed frame.

    Filtering is applied per observation term: the diurnal wave survives in
    shallow layers down to ~1 m, so mixing terms would inflate the variance
    and mask real anomalies.  ``hours=None`` uses every hour present.
    """
    if hours is None:
        hours = tuple(sorted(df.index.hour.unique()))
    for hour in hours:
        subset = df[df.index.hour == hour]
        if not subset.empty:
            yield hour, subset


def align_level_shifts(clean: pd.DataFrame, depths_from: float = 5.0,
                       min_step: float = 0.2, window_days: int = 14,
                       max_shifts: int = 5, edge: int = 32,
                       season_free_from: float = 12.0) -> pd.DataFrame:
    """Выравнивание сдвигов уровня между поколениями приборов.

    При замене термокосы новый прибор даёт другой уровень (калибровка) —
    один и тот же сдвиг на всех датчиках. На глубинах >= `depths_from`
    температура меняется медленно, ступень не может быть естественной —
    точечные фильтры её не ловят, поэтому отдельная стадия.

    Подтверждение и оценка — только по сезон-свободным горизонтам
    (>= `season_free_from`, где годовой ход < min_step): оценка по мелким
    горизонтам через пропуск связи длительностью в долю сезона неизбежно
    включает естественный ход (июнь->декабрь на 5 м — это +1.4 °C, что
    неотличимо от ступени прибора). Единая константа (медиана по
    сезон-свободным глубинам) вычитается из всех горизонтов >= depths_from
    после границы: калибровка едина для всех датчиков кабеля. Скважины без
    сезон-свободных горизонтов не выравниваются — ступень там подтвердить
    нечем.

    Границы — только смены кластеров (периодов прибора, колонка
    ``cluster``): внутри одного кластера одно устройство стоит в одном
    месте, и устойчивый скачок там — прогрев в начале записи или
    естественный ход, а не смена прибора. Медианные окна ``edge``
    наблюдений по обе стороны границы устойчивы к переходному прогреву
    нового прибора (первые ~3 суток) и к выбросам; кластеры сравниваются
    по соседним валидным наблюдениям, поэтому граница видна и внутри
    пропуска связи.
    """
    fixed_parts = []
    for well, part in clean.groupby("site"):
        part = part.sort_values("DateTime").set_index("DateTime").copy()
        has_cluster = "cluster" in part.columns
        depth_cols = [c for c in part.columns
                      if isinstance(c, float) and c >= depths_from]
        if has_cluster and depth_cols:
            free_cols = [c for c in depth_cols if c >= season_free_from]
            boundaries = _cluster_boundaries(part, depth_cols)
            applied_at = []
            for b in boundaries:
                if len(applied_at) >= max_shifts:
                    break
                if any(abs(b - t0) <= pd.Timedelta(days=window_days)
                       for t0 in applied_at):
                    continue
                values = [_median_step(part[d], b, edge) for d in free_cols]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                common = float(np.median(values))
                if abs(common) < min_step:
                    continue
                for d in depth_cols:
                    series = part[d].dropna()
                    series.loc[series.index >= b] = \
                        series.loc[series.index >= b] - common
                    part[d] = series
                applied_at.append(b)
        if has_cluster:
            part = part.drop(columns="cluster")
        fixed_parts.append(part.reset_index())
    return pd.concat(fixed_parts, ignore_index=True)


def _cluster_boundaries(part: pd.DataFrame, depth_cols: list[float]) -> list:
    """Моменты смены кластера (первый срок нового кластера)."""
    rows = part[depth_cols].notna().any(axis=1) & part["cluster"].notna()
    clusters = part.loc[rows, "cluster"]
    change = clusters[clusters != clusters.shift()]
    return list(change.index[1:])       # начало записи — не граница


def _median_step(series: pd.Series, boundary: pd.Timestamp,
                 edge: int) -> float | None:
    """Разность медиан по ``edge`` наблюдениям по обе стороны границы."""
    series = series.dropna()
    pos = series.index.searchsorted(boundary)
    head = series.iloc[max(0, pos - edge):pos]
    tail = series.iloc[pos:pos + edge]
    if len(head) < edge // 4 or len(tail) < edge // 4:
        return None
    return float(tail.median() - head.median())
