"""Google Earth Engine backend (optional): Sentinel-5P SO2/NO2 enhancement and AlphaEarth embeddings.

Requires `pip install earthengine-api`, `earthengine authenticate`, and a noncommercial Cloud
project id in EE_PROJECT. The arithmetic (enhancement, change score) lives in pure functions
that are unit-tested without Earth Engine.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from gmcn.features.indices import cosine_distance

log = logging.getLogger(__name__)


def enhancement(site_mean: float, ring_mean: float) -> float:
    """Column enhancement over a facility: site-buffer mean minus regional background ring mean."""
    if site_mean is None or ring_mean is None:
        return float("nan")
    return float(site_mean) - float(ring_mean)


def embedding_change(vec_prev: np.ndarray, vec_curr: np.ndarray) -> float:
    """Year-over-year land-change score from annual embedding vectors (cosine distance)."""
    return cosine_distance(vec_prev, vec_curr)


def _ee():
    import ee

    ee.Initialize(project=os.environ.get("EE_PROJECT"))
    return ee


def s5p_monthly_enhancement(site: dict, gas: str, start: str, end: str, cfg: dict) -> pd.DataFrame:
    """Monthly Sentinel-5P column enhancement (site buffer minus background ring) for one site."""
    ee = _ee()
    spec = cfg["collections"][gas]
    lon, lat = site["geometry"]["coordinates"]
    pt = ee.Geometry.Point([lon, lat])
    site_geom = pt.buffer(cfg["site_buffer_m"])
    ring = pt.buffer(cfg["ring_outer_m"]).difference(pt.buffer(cfg["ring_inner_m"]))
    coll = ee.ImageCollection(spec["id"]).select(spec["band"]).filterDate(start, end)

    months = pd.date_range(start, end, freq="MS")
    rows = []
    for m0, m1 in zip(months[:-1], months[1:]):
        img = coll.filterDate(m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d")).mean()
        red = ee.Reducer.mean()
        s = img.reduceRegion(red, site_geom, scale=1113).get(spec["band"]).getInfo()
        r = img.reduceRegion(red, ring, scale=1113).get(spec["band"]).getInfo()
        rows.append({"month": m0.strftime("%Y-%m"), "site_mean": s, "ring_mean": r,
                     "enhancement": enhancement(s, r)})
    out = pd.DataFrame(rows)
    out["site_id"] = site["properties"]["id"]
    out["gas"] = gas
    return out


def alphaearth_annual_vectors(feature: dict, years: list[int], cfg: dict, buffer_m: float = 0) -> pd.DataFrame:
    """Mean AlphaEarth embedding (64-dim, re-normalized) per year over an AOI, plus YoY change."""
    ee = _ee()
    geom = ee.Geometry(feature["geometry"])
    if buffer_m:
        geom = geom.buffer(buffer_m)
    coll = ee.ImageCollection(cfg["collection"])
    rows, prev = [], None
    for y in years:
        img = coll.filterDate(f"{y}-01-01", f"{y + 1}-01-01").filterBounds(geom).mosaic()
        d = img.reduceRegion(ee.Reducer.mean(), geom, scale=10, maxPixels=1e9).getInfo()
        vec = np.array([d[k] for k in sorted(d)], dtype=float)
        vec = vec / np.linalg.norm(vec)
        rows.append({"year": y, "change_vs_prev": embedding_change(prev, vec) if prev is not None else np.nan})
        prev = vec
    out = pd.DataFrame(rows)
    out["aoi_id"] = feature["properties"]["id"]
    return out
