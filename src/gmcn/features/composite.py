"""Turn per-scene statistics into regular, gap-aware time series.

Satellite looks arrive irregularly and clouds remove many of them. These helpers make that
missingness explicit instead of interpolating it away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_composite(scenes: pd.DataFrame, id_col: str, value_cols: list[str],
                     min_valid_frac: float, time_col: str = "datetime") -> pd.DataFrame:
    """Median of clear scenes per ISO week per AOI, reindexed so missing weeks show as NaN."""
    df = scenes.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, format="ISO8601").dt.tz_localize(None)
    df = df[df["valid_frac"] >= min_valid_frac]
    if df.empty:
        return pd.DataFrame(columns=[id_col, "week", *value_cols, "n_scenes", "observed"])
    df["week"] = df[time_col].dt.to_period("W-SUN").dt.start_time
    agg = df.groupby([id_col, "week"]).agg(
        **{c: (c, "median") for c in value_cols},
        n_scenes=("valid_frac", "size"),
        mean_valid_frac=("valid_frac", "mean"),
    ).reset_index()
    out = []
    for key, g in agg.groupby(id_col):
        full = pd.date_range(g.week.min(), g.week.max(), freq="7D")
        g = g.set_index("week").reindex(full)
        g.index.name = "week"
        g[id_col] = key
        g["n_scenes"] = g["n_scenes"].fillna(0).astype(int)
        g["observed"] = g["n_scenes"] > 0
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def climatology_anomaly(weekly: pd.DataFrame, id_col: str, value_col: str) -> pd.DataFrame:
    """Standardized anomaly vs the same week-of-year in OTHER years (leave-one-year-out).

    Excluding the current year keeps a drought year from shrinking its own anomaly.
    """
    df = weekly.copy()
    df["year"] = df["week"].dt.year
    df["woy"] = df["week"].dt.isocalendar().week.astype(int)
    z = np.full(len(df), np.nan)
    for (aoi, woy), g in df.groupby([id_col, "woy"]):
        for idx, row in g.iterrows():
            ref = g.loc[g.year != row.year, value_col].dropna()
            if len(ref) >= 2 and np.isfinite(row[value_col]):
                sd = ref.std(ddof=1)
                if sd > 0:
                    z[df.index.get_loc(idx)] = (row[value_col] - ref.mean()) / sd
    df[f"{value_col}_anom_z"] = z
    return df


def monthly_hotspot_activity(scenes: pd.DataFrame, min_valid_frac: float,
                             time_col: str = "datetime") -> pd.DataFrame:
    """Monthly facility activity from hot-pixel scenes.

    hot_pixels_per_look = mean hot-pixel count across clear looks (intensity)
    active_look_share   = share of clear looks with at least one hot pixel (operating status)
    """
    df = scenes.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, format="ISO8601").dt.tz_localize(None)
    df = df[df["valid_frac"] >= min_valid_frac]
    df["month"] = df[time_col].dt.to_period("M").dt.start_time
    g = df.groupby(["site_id", "month"])
    out = g.agg(
        clear_looks=("hot_pixels", "size"),
        hot_pixels_per_look=("hot_pixels", "mean"),
        active_look_share=("hot_pixels", lambda s: float((s > 0).mean())),
        b12_hot_sum_per_look=("b12_hot_sum", "mean"),
    ).reset_index()
    return out
