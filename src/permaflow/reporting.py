"""The interactive assignment map for the preprocessing report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _bp_palette(bp_names: list[str]) -> dict[str, str]:
    """Stable saturated colour per base station (golden-angle hues), hex."""
    import colorsys

    def to_hex(hue):
        r, g, b = colorsys.hls_to_rgb(hue / 360, 0.38, 0.85)
        return "#%02x%02x%02x" % tuple(int(c * 255) for c in (r, g, b))

    return {bp: to_hex((i * 137.508) % 360) for i, bp in enumerate(sorted(bp_names))}


def _lighten(hex_color: str, factor: float = 0.45) -> str:
    """Blend a colour with white (for lighter borehole markers)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    return "#%02x%02x%02x" % (r, g, b)


def plot_site_map_interactive(named: pd.DataFrame, bp_points: pd.DataFrame,
                              unidentified: pd.DataFrame,
                              noise: pd.DataFrame, path: Path) -> None:
    """Interactive HTML map of the well-to-base-station assignment.

    The base map is plotly's built-in physical geography (Natural Earth data
    bundled with plotly.js) — no tile servers, but plotly.js itself is loaded
    from CDN when the HTML is opened.  Wells are coloured by their base
    station; the base-station point is drawn larger, translucent, with a
    black outline.  Clusters without a well (red diamonds) hover with the
    nearest base station, the distance to it and the message count; DBSCAN
    noise is grey.
    """
    palette = _bp_palette(sorted(set(named["Base station name"]) |
                                 set(bp_points["Base station name"])))
    fig = go.Figure()

    hover_well = ("Well: %{text}<br>Base station: %{customdata[0]}<br>"
                  "measurements: %{customdata[1]}<br>"
                  "Distance to base station: %{customdata[2]} km<extra>well</extra>")
    fig.add_trace(go.Scattergeo(
        lat=named["lat"], lon=named["lon"], mode="markers",
        name="wells (TB)", text=named["TB id"],
        customdata=np.stack([named["Base station name"],
                             named["number of measurements"],
                             named["distance to the base station"]], axis=1),
        hovertemplate=hover_well,
        marker=dict(size=8,
                    color=[_lighten(palette[bp]) for bp in named["Base station name"]],
                    line=dict(color="#1a1a1a", width=1))))

    if not bp_points.empty:
        hover_bp = ("Base station: %{text}<br>ID: %{customdata[0]}<br>"
                    "wells: %{customdata[1]}<extra></extra>")
        fig.add_trace(go.Scattergeo(
            lat=bp_points["lat"], lon=bp_points["lon"], mode="markers",
            name="base stations", text=bp_points["Base station name"],
            customdata=np.stack([bp_points["base station id"],
                                 bp_points["n_boreholes"]], axis=1),
            hovertemplate=hover_bp,
            marker=dict(size=13, opacity=0.35,
                        color=[palette[bp] for bp in bp_points["Base station name"]],
                        line=dict(color="black", width=2))))

    if not unidentified.empty:
        if {"nearest base station", "nearest distance (km)"} <= set(unidentified.columns):
            hover_unid = ("Cluster without a base station<br>"
                          "nearest: %{customdata[0]} (%{customdata[1]:.0f} km)<br>"
                          "messages: %{customdata[2]}<extra></extra>")
            customdata = np.stack([unidentified["nearest base station"],
                                   unidentified["nearest distance (km)"],
                                   unidentified["messages"]], axis=1)
        else:
            hover_unid = ("Cluster without a base station<br>"
                          "messages: %{customdata}<extra></extra>")
            customdata = unidentified["messages"]
        fig.add_trace(go.Scattergeo(
            lat=unidentified["lat"], lon=unidentified["lon"], mode="markers",
            name="UNIDENTIFIED",
            customdata=customdata,
            hovertemplate=hover_unid,
            marker=dict(size=11, color="red", symbol="diamond")))

    if not noise.empty:
        fig.add_trace(go.Scattergeo(
            lat=noise["lat"], lon=noise["lon"], mode="markers",
            name="DBSCAN noise",
            hovertemplate="message outside clusters<extra>DBSCAN noise</extra>",
            marker=dict(size=4, color="grey", opacity=0.5)))

    fig.update_layout(
        geo=dict(projection=dict(type="mercator"),
                 resolution=110,
                 showland=True, landcolor="#f2eee3",
                 showocean=True, oceancolor="#dce9f4",
                 showcountries=True, countrycolor="#9aa5ad", countrywidth=1,
                 showcoastlines=True, coastlinecolor="#7b8894", coastlinewidth=1,
                 showlakes=True, lakecolor="#dce9f4",
                 lonaxis=dict(range=[18, 192]),
                 lataxis=dict(range=[40, 83]),
                 bgcolor="#dce9f4"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02),
        title="Привязка скважин к базовым пунктам наблюдений")
    fig.write_html(path, include_plotlyjs="cdn")
