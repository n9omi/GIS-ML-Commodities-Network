"""Synthetic Sentinel-2 scenes written as real GeoTIFFs, so the ingestion code is exercised
end to end (windowed reads, resampling, reprojection, masking) without network access."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

UTM = "EPSG:32615"          # UTM zone 15N (central Iowa)
X0, Y0 = 440000.0, 4655000.0  # upper-left corner of the synthetic tile
N10 = 200                   # 200 x 10 m = 2 km
N20 = 100                   # 100 x 20 m = 2 km

S2_CFG = {
    "reflectance_scale": 10000, "boa_offset_baseline": "04.00", "boa_offset": -1000,
    "scl_mask_vegetation": [0, 1, 3, 8, 9, 10, 11], "scl_mask_hotspot": [0, 3, 8, 9, 10, 11],
    "hotspot_nhi_threshold": 0.0, "hotspot_min_b12": 0.15, "catalog": "planetary-computer",
    "cdl_corn_class": 1,
}


def _write(path: Path, arr: np.ndarray, res: float, crs: str = UTM, x0: float = X0, y0: float = Y0):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=crs, transform=from_origin(x0, y0, res, res), nodata=0) as dst:
        dst.write(arr, 1)


def utm_to_wgs84(x: float, y: float) -> tuple[float, float]:
    tf = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    return tf.transform(x, y)


def aoi_polygon(inset_m: float = 100.0) -> dict:
    """Square AOI inside the tile, in WGS84."""
    corners = [(X0 + inset_m, Y0 - inset_m), (X0 + 2000 - inset_m, Y0 - inset_m),
               (X0 + 2000 - inset_m, Y0 - 2000 + inset_m), (X0 + inset_m, Y0 - 2000 + inset_m)]
    ring = [list(utm_to_wgs84(x, y)) for x, y in corners]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def make_scene(tmp: Path, name: str, baseline: str, *, cloud_quadrant: bool = True,
               right_half_ndvi_low: bool = False, hotspot: bool = False) -> pystac.Item:
    off = 1000 if float(baseline) >= 4.0 else 0
    d = tmp / name
    d.mkdir()
    red = np.full((N10, N10), 500 + off, dtype="uint16")
    nir = np.full((N10, N10), 4000 + off, dtype="uint16")
    blue = np.full((N10, N10), 300 + off, dtype="uint16")
    if right_half_ndvi_low:
        nir[:, N10 // 2:] = 750 + off          # NDVI = (0.075-0.05)/(0.125) = 0.2
    scl = np.full((N20, N20), 4, dtype="uint8")
    if cloud_quadrant:
        scl[:N20 // 2, :N20 // 2] = 9          # high-probability cloud, 25 % of the tile
    b8a = np.full((N20, N20), 3000 + off, dtype="uint16")
    b11 = np.full((N20, N20), 3000 + off, dtype="uint16")
    b12 = np.full((N20, N20), 2000 + off, dtype="uint16")
    if hotspot:
        r0, c0 = 60, 60
        b12[r0:r0 + 3, c0:c0 + 3] = 5000 + off  # 3 x 3 hot cluster: NHI_SWIR = 0.25
        b11[r0:r0 + 3, c0:c0 + 3] = 3000 + off
        b8a[r0:r0 + 3, c0:c0 + 3] = 2000 + off
        scl[r0:r0 + 3, c0:c0 + 3] = 1          # saturated / defective class, typical for hot pixels
    files = {"B02": (blue, 10), "B04": (red, 10), "B08": (nir, 10), "B8A": (b8a, 20),
             "B11": (b11, 20), "B12": (b12, 20), "SCL": (scl, 20)}
    item = pystac.Item(id=name, geometry=aoi_polygon(0), bbox=None,
                       datetime=__import__("datetime").datetime(2025, 7, 15),
                       properties={"s2:processing_baseline": baseline})
    for key, (arr, res) in files.items():
        p = d / f"{key}.tif"
        _write(p, arr, res)
        item.add_asset(key, pystac.Asset(href=str(p)))
    return item


@pytest.fixture
def s2_cfg():
    return dict(S2_CFG)


@pytest.fixture
def scene_factory(tmp_path):
    def _f(name, baseline="05.10", **kw):
        return make_scene(tmp_path, name, baseline, **kw)
    return _f


@pytest.fixture
def cdl_raster(tmp_path) -> Path:
    """30 m Cropland Data Layer stand-in in Albers (EPSG:5070): corn on the west half."""
    tf = Transformer.from_crs(UTM, "EPSG:5070", always_xy=True)
    xs, ys = [], []
    for x, y in [(X0, Y0), (X0 + 2000, Y0), (X0, Y0 - 2000), (X0 + 2000, Y0 - 2000)]:
        a, b = tf.transform(x, y)
        xs.append(a)
        ys.append(b)
    mid_utm_x = X0 + 1000
    left, top = min(xs) - 300, max(ys) + 300
    n = int((max(xs) - min(xs) + 600) / 30) + 1
    m = int((max(ys) - min(ys) + 600) / 30) + 1
    arr = np.full((m, n), 5, dtype="uint8")    # 5 = soybeans
    inv = Transformer.from_crs("EPSG:5070", UTM, always_xy=True)
    for i in range(m):
        for j in range(n):
            ax, ay = left + (j + 0.5) * 30, top - (i + 0.5) * 30
            ux, _ = inv.transform(ax, ay)
            if ux < mid_utm_x:
                arr[i, j] = 1                  # 1 = corn
    p = tmp_path / "cdl.tif"
    _write(p, arr, 30, crs="EPSG:5070", x0=left, y0=top)
    return p
