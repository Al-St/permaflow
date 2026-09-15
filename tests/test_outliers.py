"""Unit tests for the robust outlier filter (synthetic signals)."""

import numpy as np
import pandas as pd
import pytest

from permaflow.config import GapFillConfig
from permaflow.outliers import (align_level_shifts, fill_short_gaps,
                                laplace_bounds, level_outlier_mask,
                                pointwise_outlier_mask, rolling_outlier_mask,
                                wavelet_detrend)
from permaflow.pipeline import recover_gaps
from permaflow.prepare import (drop_device_warmup, regularize_grid,
                               wide_temperature_table)


def make_series(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 365)
    trend = 0.002 * t
    noise = rng.laplace(0, 0.15, n)
    return pd.Series(seasonal + trend + noise, index=pd.date_range("2023-01-01", periods=n, freq="D"))


def test_wavelet_detrend_removes_seasonality():
    s = make_series()
    resid = wavelet_detrend(s)
    assert resid.std() < s.std() / 5
    # mean of the residual stays close to zero
    assert abs(resid.mean()) < 0.5


def test_laplace_bounds_symmetric_and_finite():
    rng = np.random.default_rng(1)
    values = rng.laplace(0, 1, 5000)
    left, right = laplace_bounds(values, alpha=0.01)
    assert np.isfinite(left) and np.isfinite(right)
    assert abs(left + right) < 0.3  # roughly symmetric around 0
    assert (values < left).sum() / values.size < 0.02


def test_pointwise_mask_catches_spike():
    s = make_series()
    s.iloc[500] = 42.0  # physically impossible value
    resid = wavelet_detrend(s)  # mask is applied to the detrended series
    mask = pointwise_outlier_mask(resid.to_frame("z"), alpha=0.01)
    assert bool(mask.iloc[500, 0])


def test_rolling_mask_catches_burst_and_keeps_edges():
    s = make_series()
    s.iloc[500:505] = 25.0  # sensor glitch burst
    df = s.to_frame("z")
    mask = rolling_outlier_mask(df, samples_per_day=1, window_days=60, alpha=0.01)
    assert bool(mask.iloc[501, 0])
    assert bool(mask.index.equals(df.index))


def test_rolling_mask_flags_exactly_the_spike():
    # regression: even window + closed="both" shifted the tested centre by
    # one position, so isolated spikes survived whenever the neighbour was
    # NaN; the odd window must flag the spike itself
    idx = pd.date_range("2023-01-01", periods=100, freq="D")
    s = pd.Series(np.zeros(100), index=idx)
    s.iloc[10:13] = np.nan
    s.iloc[11] = 50.0  # spike between two NaNs
    mask = rolling_outlier_mask(s.to_frame("z"), samples_per_day=1,
                                window_days=30, alpha=0.01)
    assert bool(mask.iloc[11, 0])
    assert mask["z"].sum() == 1


def test_rolling_mask_works_inside_long_gaps():
    # regression: full-window requirement silently disabled filtering in
    # gappy stretches — a spike there must still be flagged
    s = make_series(n=300)
    s.iloc[130:146] = np.nan          # long transmission gap
    s.iloc[138] = 60.0                # glitch inside the gap
    mask = rolling_outlier_mask(s.to_frame("z"), samples_per_day=1,
                                window_days=30, alpha=0.01)
    assert bool(mask.iloc[138, 0])
    # and the pre-existing behaviour outside gaps is preserved
    s2 = make_series()
    s2.iloc[500:505] = 25.0
    mask2 = rolling_outlier_mask(s2.to_frame("z"), samples_per_day=1,
                                 window_days=60, alpha=0.01)
    assert bool(mask2.iloc[501, 0])


def test_fill_short_gaps_bridges_only_short_gaps():
    idx = pd.date_range("2023-01-01", periods=50, freq="D")
    s = pd.Series(np.linspace(0, 1, 50), index=idx)
    s.iloc[10:13] = np.nan   # short gap -> fully bridged
    s.iloc[30:47] = np.nan   # long gap -> only window edges filled, core stays NaN
    filled = fill_short_gaps(s.to_frame("z"), limit=3)["z"]
    assert filled.iloc[10:13].notna().all()
    assert filled.iloc[36:41].isna().all()  # core of the long gap untouched


def test_global_stage_catches_spike_inside_contaminated_window():
    # regression: when a whole month transmits garbage, the local window's
    # median/MAD are poisoned and the rolling gate misses even a gross spike;
    # the second (whole-series) stage must catch it — exactly the pipeline's
    # two-stage design
    s = make_series(n=400)
    s.iloc[200:216] = [25.0, -25.0] * 8   # corrupted transmission episode
    s.iloc[207] = 45.0                    # gross spike inside the episode
    det = wavelet_detrend(s)
    rolling = rolling_outlier_mask(det.to_frame("z"), samples_per_day=1,
                                   window_days=30, alpha=0.01)
    combined = rolling["z"] | pointwise_outlier_mask(det.to_frame("z"), 0.01)["z"]
    assert not bool(rolling.iloc[207, 0])
    assert bool(combined.iloc[207])
    # plausible values outside the episode stay untouched
    assert not bool(combined.iloc[100])


def test_laplace_bounds_degenerate_window():
    # constant window: MAD = 0 must not produce NaN bounds
    values = np.zeros(20)
    assert laplace_bounds(values, alpha=0.01) == (0.0, 0.0)
    values[3] = 5.0
    left, right = laplace_bounds(values, alpha=0.01)
    assert right < 5.0  # the single deviating value falls outside


def test_quantized_flat_series_is_not_mass_flagged():
    # regression (АМСГ-2 Нюрба, 14 м): deep sensors report quantized values
    # (step ~0.005 °C); without an absolute floor the fitted scale collapses
    # and the gates flag the sensor's own quantization steps
    idx = pd.date_range("2023-01-01", periods=800, freq="D")
    rng = np.random.default_rng(0)
    base = -0.48 + 0.05 * np.sin(2 * np.pi * np.arange(800) / 365)
    quantized = np.round((base + rng.normal(0, 0.002, 800)) / 0.005) * 0.005
    s = pd.Series(quantized, index=idx)
    s.iloc[400] = 5.0  # a real gross glitch
    det = wavelet_detrend(s)
    frame = det.to_frame("z")
    global_mask = pointwise_outlier_mask(frame, alpha=0.01, min_deviation=0.5)
    rolling_mask = rolling_outlier_mask(frame, samples_per_day=1,
                                        window_days=30, alpha=0.01,
                                        min_deviation=0.5)
    combined = global_mask["z"] | rolling_mask["z"]
    assert combined.mean() < 0.02          # quantization steps survive
    assert bool(combined.iloc[400])        # the gross glitch does not


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_alpha_controls_strictness(alpha):
    rng = np.random.default_rng(3)
    s = pd.Series(rng.laplace(0, 1, 3000),
                  index=pd.date_range("2023-01-01", periods=3000, freq="D"))
    mask = pointwise_outlier_mask(s.to_frame("z"), alpha=alpha)
    assert mask["z"].mean() <= alpha * 1.5


def test_permaflow_imports():
    import permaflow  # noqa: F401


def test_align_level_shifts_fixes_device_step():
    # смена прибора (граница кластеров): старый уровень -2.58, новый -2.34
    idx = pd.date_range("2024-01-01", periods=360, freq="6h")
    base = -2.46 + 0.05 * np.sin(2 * np.pi * np.arange(360) / 1460)
    values = np.where(np.arange(360) < 180, base - 0.12, base + 0.12)
    df = pd.DataFrame({"DateTime": idx, 10.0: values, "site": "TB-0000-00",
                       "cluster": np.where(np.arange(360) < 180, 1, 2)})

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)

    first = aligned[aligned["DateTime"] < idx[180]][10.0]
    second = aligned[aligned["DateTime"] >= idx[180]][10.0]
    assert abs(first.mean() - second.mean()) < 0.08  # уровни выровнены
    # реальные колебания не тронуты: разброс внутри сегментов сохранился
    assert second.std() > 0.003
    assert "cluster" not in aligned.columns          # служебная колонка удалена


def test_align_level_shifts_negative_step_converges():
    # regression: вычитался модуль сдвига, итерации расходились (31x за 5
    # проходов, значения до -444.7 C в продукте); знаковый сдвиг должен
    # выровнять ряд за один проход без расходимости
    idx = pd.date_range("2024-01-01", periods=1460, freq="6h")
    base = -2.0 + 0.05 * np.sin(2 * np.pi * np.arange(1460) / 1460)
    values = np.where(np.arange(1460) < 730, base + 0.5, base - 0.5)
    df = pd.DataFrame({"DateTime": idx, 10.0: values, "site": "TB-0000-00",
                       "cluster": np.where(np.arange(1460) < 730, 1, 2)})

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)

    first = aligned[aligned["DateTime"] < idx[730]][10.0]
    second = aligned[aligned["DateTime"] >= idx[730]][10.0]
    assert abs(first.mean() - second.mean()) < 0.1
    assert aligned[10.0].min() > -10.0        # нет расходимости вниз


def test_align_level_shifts_ignores_seasonal_ramp():
    # защита от сезонного хода: у синусоиды с большим наклоном сдвиг медиан
    # до 0.72 C (больше min_step), но внутри одного кластера (без смены
    # прибора) выравнивание не применяется в принципе
    idx = pd.date_range("2024-01-01", periods=2920, freq="6h")
    values = -2.0 + 3.0 * np.sin(2 * np.pi * np.arange(2920) / 1460)
    df = pd.DataFrame({"DateTime": idx, 10.0: values, "site": "TB-0000-00",
                       "cluster": 1})

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)

    assert np.allclose(aligned[10.0].dropna(), values, atol=0.05)


def test_align_level_shifts_ignores_warmup_without_device_change():
    # regression (20 скважин: сдвиг к уровню первых 0-9 суток): переходный
    # прогрев прибора в начале записи (один кластер, смены прибора не было)
    # не должен выравниваться — ряд остаётся как есть
    idx = pd.date_range("2023-10-08", periods=1200, freq="6h")
    rng = np.random.default_rng(4)
    values = -5.1 + rng.normal(0, 0.02, 1200)
    values[:12] = np.linspace(-0.09, -5.1, 12)      # прогрев ~3 суток
    df = pd.DataFrame({"DateTime": idx, 10.0: values, "site": "TB-0000-00",
                       "cluster": 1})

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)

    assert np.allclose(aligned[10.0].dropna(), values)


def test_align_level_shifts_warmup_survives_real_device_change():
    # смена прибора с переходным прогревом у нового: величина сдвига
    # оценивается по медианам 32 наблюдений (8 суток), поэтому прогрев
    # первых 3 суток не смещает оценку — уровни выравниваются корректно
    idx = pd.date_range("2023-01-01", periods=1000, freq="6h")
    rng = np.random.default_rng(5)
    values = -5.0 + rng.normal(0, 0.02, 1000)
    values[500:] = values[500:] - 0.8               # новый прибор ниже на 0.8
    values[500:512] = np.linspace(-4.2, -5.8, 12)   # прогрев нового прибора
    df = pd.DataFrame({"DateTime": idx, 10.0: values, "site": "TB-0000-00",
                       "cluster": np.where(np.arange(1000) < 500, 1, 2)})

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)

    first = aligned[aligned["DateTime"] < idx[500]][10.0]
    second = aligned[aligned["DateTime"] >= idx[530]][10.0]   # после прогрева
    assert abs(first.mean() - second.mean()) < 0.15


def test_align_level_shifts_works_on_short_series():
    # детекция по границам кластеров не требует длинных медианных окон:
    # ступень выравнивается и на коротком ряду (14 суток)
    idx = pd.date_range("2024-01-01", periods=56, freq="6h")
    values = np.where(np.arange(56) < 28, -2.5, -1.5)
    df = pd.DataFrame({"DateTime": idx, 10.0: values, "site": "TB-0000-00",
                       "cluster": np.where(np.arange(56) < 28, 1, 2)})

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)

    first = aligned[aligned["DateTime"] < idx[28]][10.0]
    second = aligned[aligned["DateTime"] >= idx[28]][10.0]
    assert abs(first.mean() - second.mean()) < 0.1


def test_align_level_shifts_sees_step_across_gap():
    # границы кластеров сравниваются по соседним валидным наблюдениям:
    # ступень смены прибора внутри длинного пропуска выравнивается
    idx = pd.date_range("2024-01-01", periods=1460, freq="6h")
    base = -2.0 + 0.05 * np.sin(2 * np.pi * np.arange(1460) / 1460)
    values = np.where(np.arange(1460) < 730, base - 0.3, base + 0.3)
    values[700:900] = np.nan                 # 50 суток без связи вокруг ступени
    df = pd.DataFrame({"DateTime": idx, 10.0: values, "site": "TB-0000-00",
                       "cluster": np.where(np.arange(1460) < 730, 1, 2)})

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)

    tail = aligned[aligned["DateTime"] >= idx[900]][10.0]
    head = aligned[aligned["DateTime"] < idx[700]][10.0]
    assert abs(head.mean() - tail.mean()) < 0.1


def test_align_level_shifts_seasonal_gap_is_not_a_step():
    # regression (TB-24959-01: сдвиги -1.37..-0.21 на 5-9 м были сезонным
    # ходом июнь->декабрь через 203-дневный пропуск): смена кластера через
    # длинный пропуск при отсутствии приборной ступени не выравнивается —
    # подтверждение идёт по сезон-свободным горизонтам (>= 12 м), где
    # естественный ход меньше порога
    idx = pd.date_range("2023-12-01", periods=1460, freq="6h")
    t = np.arange(1460)
    noise = np.random.default_rng(6).normal(0, 0.02, 1460)
    df = pd.DataFrame({"DateTime": idx, "site": "TB-0000-00"})
    for z, amp in [(5.0, 3.0), (6.0, 2.0), (7.0, 1.2), (12.0, 0.05),
                   (14.0, 0.03), (25.0, 0.02)]:
        df[z] = -2.0 + amp * np.sin(2 * np.pi * (t + 60) / 1460) + noise
        df.loc[300:1000, z] = np.nan              # ~104 суток пропуска
    df["cluster"] = np.where(np.arange(1460) < 650, 1, 2)

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=12.0)

    for z in [5.0, 6.0, 7.0, 12.0, 14.0, 25.0]:
        assert np.allclose(aligned[z].dropna(), df[z].dropna(), atol=1e-9), z


def test_align_level_shifts_deep_step_applies_common_offset():
    # приборная ступень подтверждается на сезон-свободных горизонтах и
    # применяется единой константой на все глубокие горизонты (калибровка
    # едина для датчиков кабеля), а не по-глубинно
    idx = pd.date_range("2023-12-01", periods=1460, freq="6h")
    t = np.arange(1460)
    noise = np.random.default_rng(7).normal(0, 0.02, 1460)
    df = pd.DataFrame({"DateTime": idx, "site": "TB-0000-00"})
    for z, amp in [(5.0, 3.0), (6.0, 2.0), (12.0, 0.05), (14.0, 0.03)]:
        values = -2.0 + amp * np.sin(2 * np.pi * (t + 60) / 1460) + noise
        df[z] = np.where(t < 730, values, values + 0.5)   # новый прибор +0.5
    df["cluster"] = np.where(t < 730, 1, 2)

    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=12.0)

    # выровненный ряд совпадает с эталоном «ступени не было» на всех глубинах
    for z, amp in [(5.0, 3.0), (6.0, 2.0), (12.0, 0.05), (14.0, 0.03)]:
        base = -2.0 + amp * np.sin(2 * np.pi * (t + 60) / 1460) + noise
        assert (aligned[z] - base).abs().max() < 0.1, z


def test_align_level_shifts_keeps_real_shallow_variability():
    # на глубинах < depths_from выравнивание не применяется
    idx = pd.date_range("2024-01-01", periods=360, freq="6h")
    df = pd.DataFrame({"DateTime": idx, "site": "TB-0000-00",
                       "cluster": np.where(np.arange(360) < 180, 1, 2)})
    df[0.0] = np.where(np.arange(360) < 180, -5.0, 5.0)  # реальный сезонный ход
    aligned = align_level_shifts(df, depths_from=5.0, min_step=0.2,
                                 season_free_from=5.0)
    assert abs(aligned[0.0].mean()) < 0.5  # мелкие глубины не выравниваются


def test_level_gate_catches_saturation_burst():
    # regression (TB-24691-01, 0.0 м; TB-20046-01, 4.0 м: сбои +41.91 C
    # выживали): серию насыщения детренд поглощает в аппроксимацию, порог
    # по уровню должен помечать её по отклонению от скользящей медианы
    idx = pd.date_range("2024-01-01", periods=1000, freq="6h")
    rng = np.random.default_rng(0)
    values = -0.5 + 0.1 * np.sin(2 * np.pi * np.arange(1000) / 1460) \
        + rng.normal(0, 0.02, 1000)
    values[500:515] = 41.91                  # насыщение датчика, ~4 суток
    mask = level_outlier_mask(pd.DataFrame({"z": values}, index=idx),
                              samples_per_day=4, window_days=30)
    assert bool(mask["z"].iloc[500:515].all())
    assert mask["z"].sum() <= 15 + 2         # остальной ряд не тронут


def test_level_gate_keeps_genuine_cold_snap():
    # реальный синоптический спад на 0.0 м (на 20 °C за пару суток от уровня
    # месяца) не помечается: пол 25 °C отсекает всё физически возможное
    idx = pd.date_range("2024-01-01", periods=1000, freq="6h")
    rng = np.random.default_rng(1)
    values = -15.0 + rng.normal(0, 2.0, 1000)
    values[500:510] = -35.0                  # холодный всплеск
    mask = level_outlier_mask(pd.DataFrame({"z": values}, index=idx),
                              samples_per_day=4, window_days=30)
    assert not bool(mask["z"].iloc[500:510].any())
    assert mask["z"].sum() <= 1000 * 0.005   # фон не выкашивается


def test_level_gate_catches_stuck_channel():
    # regression (TB-20046-01, 5-25 м, июнь-июль 2025): канал «залип» на
    # константе ~-59.8 около месяца — больше половины скользящего окна,
    # локальная медиана отравлена; глобальная проверка должна ловить серию
    idx = pd.date_range("2024-01-01", periods=1000, freq="6h")
    values = -9.0 + np.random.default_rng(2).normal(0, 0.05, 1000)
    values[400:470] = -59.79                  # stuck channel, ~17 суток
    mask = level_outlier_mask(pd.DataFrame({"z": values}, index=idx),
                              samples_per_day=4, window_days=30)
    assert bool(mask["z"].iloc[400:470].all())
    assert not bool(mask["z"].iloc[600:800].any())   # фон не тронут


def test_level_gate_catches_cold_rail_on_winter_series():
    # regression (TB-24790-01, 0.0 м): рейка -41.91 на зимнем ряду —
    # отклонение от медианы ~27 C ловит серия одинаковых значений
    idx = pd.date_range("2024-10-01", periods=800, freq="6h")
    rng = np.random.default_rng(3)
    values = -15.0 + rng.normal(0, 4.0, 800)
    values[300:325] = -41.91                  # холодная рейка датчика
    mask = level_outlier_mask(pd.DataFrame({"z": values}, index=idx),
                              samples_per_day=4, window_days=30)
    assert bool(mask["z"].iloc[300:325].all())
    assert mask["z"].sum() <= 25 + 800 * 0.01


def test_recover_gaps_limits_by_depth_class():
    # лимит пропуска зависит от класса глубины: суточный ход (<= 1 м) —
    # 2 интервала, годовой ход (1–12 м) — 40, сезон-свободные — 120
    idx = pd.date_range("2024-01-01", periods=200, freq="6h")

    def col(gap):
        values = np.linspace(-1, 1, 200)
        values[60:60 + gap] = np.nan
        return values

    clean = pd.DataFrame({"DateTime": idx, "site": "TB-0000-00",
                          0.4: col(3),       # суточный класс: 3 > лимита 2
                          5.0: col(41),      # годовой ход: 41 > лимита 40
                          14.0: col(41)})    # сезон-свободный: 41 <= 120

    filled = recover_gaps(clean, GapFillConfig()).set_index("DateTime")

    assert filled[0.4].iloc[60:63].isna().all()      # длиннее лимита класса
    assert filled[5.0].iloc[60:101].isna().all()
    assert filled[14.0].iloc[60:101].notna().all()   # внутри лимита класса


def test_recover_gaps_diurnal_short_gap_filled():
    # на суточных глубинах допустимы единичные пропуски: 2 интервала
    # (ровно лимит) восстанавливаются, 3 — уже нет
    idx = pd.date_range("2024-01-01", periods=100, freq="6h")
    values = np.linspace(-1, 1, 100)
    values[40:42] = np.nan                     # 2 интервала
    values[60:63] = np.nan                     # 3 интервала
    clean = pd.DataFrame({"DateTime": idx, 0.0: values, "site": "TB-0000-00"})

    filled = recover_gaps(clean, GapFillConfig()).set_index("DateTime")[0.0]
    assert filled.iloc[40:42].notna().all()
    assert filled.iloc[60:63].isna().all()


def test_recover_gaps_deep_uses_boundary_means():
    # сезон-свободный горизонт: мост по средним 4 точек у границ, а не по
    # одиночным (зашумлённым) граничным точкам; ход между уровнями линейный
    idx = pd.date_range("2024-01-01", periods=300, freq="6h")
    values = np.full(300, -3.0)
    values[150:] = -2.0                         # ступень уровня внутри пропуска
    values[149] = -1.0                          # шум слева от границы
    values[150:200] = np.nan                    # пропуск 50 интервалов <= 120
    values[200] = -6.0                          # шум справа от границы
    clean = pd.DataFrame({"DateTime": idx, 14.0: values, "site": "TB-0000-00"})

    filled = recover_gaps(clean, GapFillConfig()).set_index("DateTime")[14.0]

    left = np.mean([-3.0, -3.0, -3.0, -1.0])    # 4 точки до пропуска
    right = np.mean([-6.0, -2.0, -2.0, -2.0])   # 4 точки после
    frac = np.arange(1, 51) / 51                # линейный ход по времени
    assert np.allclose(filled.iloc[150:200], left + (right - left) * frac)


def test_recover_gaps_never_extrapolates():
    # краевые пропуски (до первого и после последнего наблюдения) не
    # заполняются — интерполяции там не из чего строиться
    idx = pd.date_range("2024-01-01", periods=100, freq="6h")
    values = np.linspace(-1, 1, 100)
    values[:10] = np.nan
    values[-10:] = np.nan
    clean = pd.DataFrame({"DateTime": idx, 0.4: values, 14.0: values.copy(),
                          "site": "TB-0000-00"})

    filled = recover_gaps(clean, GapFillConfig()).set_index("DateTime")
    assert filled[0.4].iloc[:10].isna().all()
    assert filled[0.4].iloc[-10:].isna().all()
    assert filled[14.0].iloc[:10].isna().all()      # в т.ч. мост по границам
    assert filled[14.0].iloc[-10:].isna().all()


def test_drop_device_warmup_per_cluster():
    # переходный процесс отбрасывается в начале каждого кластера (периода
    # прибора), а не один раз на скважину
    idx = pd.date_range("2024-01-01", periods=40, freq="6h")
    df = pd.DataFrame({
        "site": "TB-0000-00",
        "cluster": [1] * 20 + [2] * 20,
        "DateTime": idx,
        "T1": np.arange(40, dtype=float),
    })
    warmed = drop_device_warmup(df, hours=12)
    # по 2 сообщения (12 ч) в начале каждого кластера отброшены
    assert list(warmed["T1"]) == list(np.arange(2, 20)) + list(np.arange(22, 40))


def test_regularize_grid_means_and_keeps_outliers():
    # осреднение внутри интервалов; одиночный выброс в интервале без
    # соседей сохраняется во весь рост
    idx = pd.to_datetime([
        "2024-01-01 00:00", "2024-01-01 01:00",
        "2024-01-01 07:00", "2024-01-01 13:00"])
    df = pd.DataFrame({"site": "TB-0000-00", "DateTime": idx,
                       "T1": [-1.0, -3.0, 2.0, 41.91],
                       "cluster": 1})
    regular = regularize_grid(df, "6h").set_index("DateTime")
    assert regular.loc["2024-01-01 00:00", "T1"] == -2.0     # среднее интервала
    assert regular.loc["2024-01-01 06:00", "T1"] == 2.0
    assert regular.loc["2024-01-01 12:00", "T1"] == 41.91    # выброс сохранён


def test_wide_temperature_table_maps_depths():
    df = pd.DataFrame({"site": ["TB-0000-00"], "DateTime": [pd.Timestamp("2024-01-01")],
                       "GNSSLatitude": [66.5], "GNSSLongitude": [66.6],
                       "T1": [-1.0], "T32": [-3.0], "cluster": [1]})
    wide = wide_temperature_table(df)
    assert list(wide.columns) == ["site", "DateTime", "cluster",
                                  "GNSSLatitude", "GNSSLongitude", 0.0, 25.0]
