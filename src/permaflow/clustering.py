"""Spatio-temporal clustering of telemetry and well construction.

Two-level model of the monitoring network:

* **cluster** — contiguous messages of ONE device (IMEI) at one location in
  one time period, found by ST-DBSCAN (space + time radii, Birant & Kut);
  devices with different IDs never share a cluster;
* **well (скважина)** — clusters of one base station that lie within each
  other's spatial eps (mutually, every cluster keeps its own accuracy-driven
  radius); a well is identified as ``TB-<код БПН>-<NN>``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .geo import haversine_matrix, to_radians

logger = logging.getLogger(__name__)


def st_dbscan(coords_rad: np.ndarray, times: np.ndarray, eps_rad: float,
              eps_t_days: float, min_samples: int) -> np.ndarray:
    """Spatio-temporal DBSCAN (Birant & Kut, 2007).

    Two points are neighbours when they are within ``eps_rad`` (great-circle
    distance, coordinates in radians) **and** within ``eps_t_days`` of each
    other in time.  Core points have at least ``min_samples`` neighbours
    (the point itself counts, as in canonical DBSCAN); clusters grow from
    cores through chains of neighbours.

    ``times`` — datetime64 array (NaT-free).
    """
    n = len(coords_rad)
    if n == 0:
        return np.array([], dtype=int)

    t = times.astype("datetime64[s]").astype(np.int64)
    eps_t_seconds = float(eps_t_days) * 86400.0

    nn = NearestNeighbors(radius=eps_rad, metric="haversine")
    nn.fit(coords_rad)
    # X=None: точка не считается соседом самой себя, поэтому в min_samples
    # она добавляется явно (+1) — окрестность точки включает её саму
    neighbours = nn.radius_neighbors(return_distance=False)
    neighbours = [np.asarray(idx, dtype=int) for idx in neighbours]

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        ni = neighbours[i][np.abs(t[neighbours[i]] - t[i]) <= eps_t_seconds]
        if len(ni) + 1 < min_samples:
            continue  # шум: непосещённая точка без плотного окружения
        cluster_id += 1
        labels[i] = cluster_id
        stack = list(ni)
        while stack:
            j = stack.pop()
            if labels[j] != -1:
                continue
            labels[j] = cluster_id
            nj = neighbours[j][np.abs(t[neighbours[j]] - t[j]) <= eps_t_seconds]
            if len(nj) + 1 < min_samples:
                continue  # граничная точка: в кластере, но не ядро
            stack.extend(int(k) for k in nj if labels[k] == -1)
    return labels


def cluster_by_device(raw: pd.DataFrame, devices: pd.DataFrame,
                      class_params: dict) -> np.ndarray:
    """Run ST-DBSCAN separately for every device (IMEI).

    Devices with different IDs never share a cluster: сообщения каждого
    устройства кластеризуются независимо, метки сдвигаются для глобальной
    уникальности.  Пространственный радиус берётся из класса точности
    координат устройства, временной (``eps_t_days``) общий для классов.
    MinPts — из реестра devices; пустое значение заменяется классовым,
    а порог не превышает числа сообщений устройства.
    """
    device_class = raw["IMEI"].map(
        devices.drop_duplicates("IMEI").set_index("IMEI")["coord_class"]).fillna("gnss")
    device_minpts = raw["IMEI"].map(
        devices.drop_duplicates("IMEI").set_index("IMEI")["MinPts"])

    labels = np.full(len(raw), -1)
    offset = 0
    for imei, part in raw.groupby("IMEI"):
        params = class_params[device_class.loc[part.index[0]]]
        registry_minpts = device_minpts.loc[part.index].iloc[0]
        if pd.isna(registry_minpts):  # пустой MinPts в реестре — из класса
            min_samples = params["min_samples"]
        else:
            min_samples = int(registry_minpts)
        min_samples = max(1, min(min_samples, len(part)))
        idx = part.index.to_numpy()
        coords = to_radians(part[["GNSSLatitude", "GNSSLongitude"]].to_numpy())
        times = part["DateTime"].to_numpy()
        lab = st_dbscan(coords, times, params["eps_rad"],
                        params["eps_t_days"], min_samples)
        if (lab >= 0).sum() == 0:
            logger.warning("Устройство %s: %d сообщений не образовали ни одного "
                           "плотного периода (MinPts=%d) — в продукт не попадут",
                           imei, len(part), min_samples)
        lab[lab >= 0] += offset
        if lab.max() >= 0:
            offset = int(lab.max()) + 1
        labels[raw.index.get_indexer(idx)] = lab
    return labels


def cluster_table(raw: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Aggregate labelled messages into a per-cluster table."""
    marked = raw.copy()
    marked["cluster"] = labels
    clusters = (marked[marked["cluster"] >= 0]
                .groupby("cluster")
                .agg(IMEI=("IMEI", "first"),
                     lat=("GNSSLatitude", "median"),
                     lon=("GNSSLongitude", "median"),
                     messages=("DateTime", "size"),
                     start=("DateTime", "min"),
                     end=("DateTime", "max"))
                .reset_index())
    return clusters


def assign_base_stations(clusters: pd.DataFrame, base_stations: pd.DataFrame,
                         max_attach_km: float = 50.0) -> pd.DataFrame:
    """Attach every cluster to the nearest base-station row.

    Кластеры дальше ``max_attach_km`` от ближайшего БПН (или когда у БПН нет
    координат) остаются без привязки (``БПН`` = NaN) — в продукт они не
    попадают, но остаются на карте как UNIDENTIFIED.  Ближайший БПН и
    расстояние до него сохраняются для всех кластеров, в том числе
    непривязанных: это hover карты и диагностика ``clusters.csv``.
    """
    clusters = clusters.copy()
    clusters["base station id"] = pd.Series(pd.NA, index=clusters.index, dtype="object")
    clusters["base station name"] = pd.Series(pd.NA, index=clusters.index, dtype="object")
    clusters["distance to the base station (km)"] = np.nan
    clusters["nearest base station"] = pd.Series(pd.NA, index=clusters.index, dtype="object")
    clusters["nearest distance (km)"] = np.nan
    bs = base_stations.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if bs.empty or clusters.empty:
        return clusters
    dist = haversine_matrix(to_radians(clusters[["lat", "lon"]]),
                            to_radians(bs[["lat", "lon"]]))
    nearest = dist.argmin(axis=1)
    nearest_km = dist[np.arange(len(dist)), nearest]
    clusters["nearest base station"] = bs.loc[nearest, "base station name"].to_numpy()
    clusters["nearest distance (km)"] = nearest_km.round(2)
    ok = nearest_km <= max_attach_km
    clusters.loc[ok, "base station id"] = bs.loc[nearest[ok], "base station id"].to_numpy()
    clusters.loc[ok, "base station name"] = bs.loc[nearest[ok], "base station name"].to_numpy()
    clusters.loc[ok, "distance to the base station (km)"] = nearest_km[ok].round(2)
    return clusters


def build_wells(clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge clusters of one base station into wells (скважины).

    Два кластера одного БПН образуют одну скважину, только если они лежат в
    окрестностях eps **друг друга** (каждый кластер хранит свой eps —
    радиус по классу точности его устройства): d(A, B) <= eps_A и
    d(A, B) <= eps_B.  Односторонняя близость — две разные скважины.
    Компоненты связности графа взаимной близости и есть скважины.

    Возвращает ``(clusters + well id, wells summary)``.
    """
    clusters = clusters.copy()
    clusters["well"] = pd.NA
    wells_rows = []
    for bp, group in clusters[clusters["base station id"].notna()].groupby("base station id"):
        idx = group.index.to_list()
        eps = group["eps_m"].to_numpy() / 1000.0  # метры -> километры матрицы
        dist = haversine_matrix(to_radians(group[["lat", "lon"]]),
                                to_radians(group[["lat", "lon"]]))
        adj = dist <= np.minimum(eps[:, None], eps[None, :])
        np.fill_diagonal(adj, True)

        components, seen = [], set()
        for i in range(len(idx)):
            if i in seen:
                continue
            stack, comp = [i], []
            while stack:
                j = stack.pop()
                if j in seen:
                    continue
                seen.add(j)
                comp.append(j)
                stack.extend(np.flatnonzero(adj[j]))
            components.append(sorted(comp))

        order = sorted(components, key=lambda c: (-group.iloc[c]["messages"].sum(), c))
        for number, comp in enumerate(order, start=1):
            well_id = f"TB-{bp}-{number:02d}"
            for i in comp:
                clusters.loc[idx[i], "well"] = well_id
            members = group.iloc[comp]
            wells_rows.append({
                "Base station name": group["base station name"].iloc[0],
                "base station id": bp,
                "TB id": well_id,
                "latitude": members["lat"].median(),
                "longitude": members["lon"].median(),
                "distance to the base station (km)": round(
                    float(members["distance to the base station (km)"].median()), 2),
                "clusters merged": len(comp),
                "messages": int(members["messages"].sum()),
            })
    wells = pd.DataFrame(wells_rows)
    if not wells.empty:
        wells = wells.sort_values(["base station id", "TB id"]).reset_index(drop=True)
    return clusters, wells


def unattached_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    """Clusters without a base station (for the map's red layer)."""
    cols = ["cluster", "lat", "lon", "messages"]
    cols += [c for c in ("nearest base station", "nearest distance (km)")
             if c in clusters.columns]
    return clusters[clusters["base station id"].isna()][cols].reset_index(drop=True)
