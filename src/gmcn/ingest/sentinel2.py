"""Sentinel-2 L2A ingestion: STAC search, windowed COG reads, cloud masking and zonal statistics.

Reads only the pixels inside each area of interest (AOI) straight from cloud-optimized
GeoTIFFs, so a laptop never downloads a full 110 km tile.

Two products:
  * vegetation: NDVI and EVI medians over cropland AOIs (corn module)
  * hotspots:   SWIR hot-pixel counts inside smelter and mine buffers (copper module)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import Window, from_bounds
import shapely
from shapely.geometry import mapping, shape
from pyproj import Transformer

from gmcn.features import indices as ix

log = logging.getLogger(__name__)

BAND_KEYS = {
    "planetary-computer": {"blue": "B02", "red": "B04", "nir": "B08", "b8a": "B8A",
                           "b11": "B11", "b12": "B12", "scl": "SCL"},
    "earth-search": {"blue": "blue", "red": "red", "nir": "nir", "b8a": "nir08",
                     "b11": "swir16", "b12": "swir22", "scl": "scl"},
}


# --------------------------------------------------------------------------- search
def search_items(geometry: dict, start: str, end: str, cfg: dict) -> list:
    """STAC search for Sentinel-2 L2A scenes intersecting `geometry` (GeoJSON, WGS84)."""
    import pystac_client

    catalog = cfg["catalog"]
    modifier = None
    if catalog == "planetary-computer":
        import planetary_computer

        modifier = planetary_computer.sign_inplace
    client = pystac_client.Client.open(cfg["stac_url"][catalog], modifier=modifier)
    search = client.search(
        collections=[cfg["collection"]],
        intersects=geometry,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cfg["max_scene_cloud_pct"]}},
    )
    items = list(search.items())
    log.info("STAC %s: %d scenes for %s..%s", catalog, len(items), start, end)
    return items


def boa_offset(item, cfg: dict) -> float:
    """Return the DN offset to apply for this scene (see indices.dn_to_reflectance)."""
    props = item.properties
    if props.get("earthsearch:boa_offset_applied") is True:
        return 0.0
    pb = props.get("s2:processing_baseline")
    try:
        if pb is not None and float(pb) >= float(cfg["boa_offset_baseline"]):
            return float(cfg["boa_offset"])
    except (TypeError, ValueError):
        pass
    return 0.0


# --------------------------------------------------------------------------- reading
@dataclass
class Grid:
    shape: tuple[int, int]
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    geom: dict  # AOI geometry in grid CRS


def _project_geometry(geom_wgs84: dict, dst_crs, buffer_m: float | None) -> dict:
    """Project a WGS84 GeoJSON geometry into `dst_crs`; buffer in metres if asked."""
    if buffer_m:
        tf = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        g = shapely.transform(shape(geom_wgs84),
                              lambda xy: np.column_stack(tf.transform(xy[:, 0], xy[:, 1])))
        return mapping(g.buffer(buffer_m))
    return transform_geom("EPSG:4326", dst_crs, geom_wgs84)


def reference_grid(href: str, geom_wgs84: dict, buffer_m: float | None = None) -> tuple[np.ndarray, Grid]:
    """Read the reference band window covering the AOI and return it with its grid."""
    with rasterio.open(href) as src:
        g = _project_geometry(geom_wgs84, src.crs, buffer_m)
        minx, miny, maxx, maxy = shape(g).bounds
        fw = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        c0, r0 = int(np.floor(fw.col_off)), int(np.floor(fw.row_off))
        c1, r1 = int(np.ceil(fw.col_off + fw.width)), int(np.ceil(fw.row_off + fw.height))
        win = Window(c0, r0, c1 - c0, r1 - r0).intersection(Window(0, 0, src.width, src.height))
        data = src.read(1, window=win)
        grid = Grid(shape=data.shape, transform=src.window_transform(win), crs=src.crs, geom=g)
    return data, grid


def read_on_grid(href: str, grid: Grid, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Read any band onto the reference grid (handles 10 m / 20 m / 60 m and CRS differences)."""
    from rasterio.vrt import WarpedVRT

    t = grid.transform
    left, top = t.c, t.f
    right, bottom = t.c + t.a * grid.shape[1], t.f + t.e * grid.shape[0]
    with rasterio.open(href) as src:
        if src.crs == grid.crs:
            win = from_bounds(left, bottom, right, top, transform=src.transform)
            return src.read(1, window=win, out_shape=grid.shape, resampling=resampling,
                            boundless=True, fill_value=0).astype("float64")
        # Different CRS (e.g. the 30 m Albers Cropland Data Layer): warp lazily onto the
        # reference grid itself, so only the AOI's source pixels are ever read.
        with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.shape[1],
                       height=grid.shape[0], resampling=resampling) as vrt:
            return vrt.read(1).astype("float64")


def aoi_mask(grid: Grid) -> np.ndarray:
    """True for pixels whose centre falls inside the AOI geometry."""
    return geometry_mask([grid.geom], out_shape=grid.shape, transform=grid.transform, invert=True)


def _href(item, key: str) -> str:
    return item.assets[key].href


# --------------------------------------------------------------------------- products
def vegetation_stats(item, geom_wgs84: dict, cfg: dict, catalog: str,
                     crop_mask_href: str | None = None, crop_class: int | None = None) -> dict:
    """NDVI / EVI statistics for one scene over one AOI, after SCL cloud masking."""
    keys = BAND_KEYS[catalog]
    red_dn, grid = reference_grid(_href(item, keys["red"]), geom_wgs84)
    nir_dn = read_on_grid(_href(item, keys["nir"]), grid)
    blue_dn = read_on_grid(_href(item, keys["blue"]), grid)
    scl = read_on_grid(_href(item, keys["scl"]), grid, Resampling.nearest)

    off = boa_offset(item, cfg)
    scale = cfg["reflectance_scale"]
    red = ix.dn_to_reflectance(red_dn, scale, off)
    nir = ix.dn_to_reflectance(nir_dn, scale, off)
    blue = ix.dn_to_reflectance(blue_dn, scale, off)

    inside = aoi_mask(grid)
    if crop_mask_href is not None and crop_class is not None:
        crop = read_on_grid(crop_mask_href, grid, Resampling.nearest)
        inside &= crop == crop_class
    clear = ix.scl_valid(scl, cfg["scl_mask_vegetation"]) & np.isfinite(red) & np.isfinite(nir)
    use = inside & clear

    n_in = int(inside.sum())
    valid_frac = float(use.sum() / n_in) if n_in else 0.0
    v_ndvi = ix.ndvi(red, nir)[use]
    v_evi = ix.evi(blue, red, nir)[use]
    return {
        "item_id": item.id,
        "datetime": item.properties.get("datetime"),
        "n_aoi_pixels": n_in,
        "valid_frac": valid_frac,
        "ndvi_median": float(np.nanmedian(v_ndvi)) if v_ndvi.size else np.nan,
        "ndvi_mean": float(np.nanmean(v_ndvi)) if v_ndvi.size else np.nan,
        "ndvi_p90": float(np.nanpercentile(v_ndvi, 90)) if v_ndvi.size else np.nan,
        "evi_median": float(np.nanmedian(v_evi)) if v_evi.size else np.nan,
        "boa_offset": off,
    }


def hotspot_stats(item, geom_wgs84: dict, buffer_m: float, cfg: dict, catalog: str) -> dict:
    """SWIR hot-pixel statistics inside a facility buffer for one scene."""
    keys = BAND_KEYS[catalog]
    b12_dn, grid = reference_grid(_href(item, keys["b12"]), geom_wgs84, buffer_m=buffer_m)
    b11_dn = read_on_grid(_href(item, keys["b11"]), grid)
    b8a_dn = read_on_grid(_href(item, keys["b8a"]), grid)
    scl = read_on_grid(_href(item, keys["scl"]), grid, Resampling.nearest)

    off = boa_offset(item, cfg)
    scale = cfg["reflectance_scale"]
    b12 = ix.dn_to_reflectance(b12_dn, scale, off)
    b11 = ix.dn_to_reflectance(b11_dn, scale, off)
    b8a = ix.dn_to_reflectance(b8a_dn, scale, off)

    inside = aoi_mask(grid)
    clear = ix.scl_valid(scl, cfg["scl_mask_hotspot"])
    use = inside & clear
    hot = ix.hot_pixel_mask(b8a, b11, b12, cfg["hotspot_nhi_threshold"], cfg["hotspot_min_b12"]) & use

    n_in = int(inside.sum())
    n_use = int(use.sum())
    nhi = ix.nhi_swir(b11, b12)[use]
    return {
        "item_id": item.id,
        "datetime": item.properties.get("datetime"),
        "n_aoi_pixels": n_in,
        "valid_frac": float(n_use / n_in) if n_in else 0.0,
        "hot_pixels": int(hot.sum()),
        "hot_frac": float(hot.sum() / n_use) if n_use else np.nan,
        "b12_hot_sum": float(np.nansum(b12[hot])),
        "nhi_swir_max": float(np.nanmax(nhi)) if nhi.size and np.isfinite(nhi).any() else np.nan,
        "boa_offset": off,
    }


# --------------------------------------------------------------------------- batch runners
def _done_ids(out_csv: Path, key_cols: list[str]) -> set:
    if not out_csv.exists():
        return set()
    prev = pd.read_csv(out_csv, usecols=key_cols)
    return set(map(tuple, prev[key_cols].astype(str).values))


def _append(out_csv: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


def run_vegetation(features: Iterable[dict], start: str, end: str, cfg: dict, out_csv: Path,
                   items_by_feature: dict | None = None, crop_mask_hrefs: dict | None = None) -> pd.DataFrame:
    """Vegetation statistics for every (AOI, scene) pair; resumable via the output CSV."""
    catalog = cfg["catalog"]
    done = _done_ids(out_csv, ["aoi_id", "item_id"])
    for feat in features:
        aoi_id = feat["properties"]["id"]
        items = (items_by_feature or {}).get(aoi_id)
        if items is None:
            try:
                items = search_items(feat["geometry"], start, end, cfg)
            except Exception as exc:
                log.error("STAC search failed for %s: %s", aoi_id, exc)
                continue
        for item in items:
            if (aoi_id, item.id) in done:
                continue
            crop_href = None
            if crop_mask_hrefs:
                year = str(item.properties.get("datetime", ""))[:4]
                crop_href = crop_mask_hrefs.get((aoi_id, year))
            try:
                row = vegetation_stats(item, feat["geometry"], cfg, catalog,
                                       crop_mask_href=crop_href, crop_class=cfg.get("cdl_corn_class"))
            except Exception as exc:  # network hiccups and corrupt tiles should not stop the run
                log.warning("skip %s %s: %s", aoi_id, item.id, exc)
                continue
            row["aoi_id"] = aoi_id
            _append(out_csv, row)
    return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()


def run_hotspots(features: Iterable[dict], start: str, end: str, cfg: dict, out_csv: Path,
                 items_by_feature: dict | None = None) -> pd.DataFrame:
    """Hot-pixel statistics for every (site, scene) pair; resumable via the output CSV."""
    catalog = cfg["catalog"]
    done = _done_ids(out_csv, ["site_id", "item_id"])
    for feat in features:
        props = feat["properties"]
        site_id = props["id"]
        items = (items_by_feature or {}).get(site_id)
        if items is None:
            try:
                items = search_items(feat["geometry"], start, end, cfg)
            except Exception as exc:
                log.error("STAC search failed for %s: %s", site_id, exc)
                continue
        for item in items:
            if (site_id, item.id) in done:
                continue
            try:
                row = hotspot_stats(item, feat["geometry"], float(props.get("buffer_m", 1500)), cfg, catalog)
            except Exception as exc:
                log.warning("skip %s %s: %s", site_id, item.id, exc)
                continue
            row.update({"site_id": site_id, "site_type": props.get("type"), "country": props.get("country")})
            _append(out_csv, row)
    return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()


def find_cdl_href(geometry: dict, year: int, cfg: dict) -> str | None:
    """Locate the USDA Cropland Data Layer COG for `year` on Planetary Computer, if available."""
    try:
        import planetary_computer
        import pystac_client

        client = pystac_client.Client.open(cfg["stac_url"]["planetary-computer"],
                                           modifier=planetary_computer.sign_inplace)
        items = list(client.search(collections=[cfg["cdl_collection"]], intersects=geometry,
                                   datetime=f"{year}-01-01/{year}-12-31").items())
        for it in items:
            for key, asset in it.assets.items():
                if "cropland" in key.lower() or key.lower() == "data":
                    return asset.href
    except Exception as exc:
        log.warning("CDL lookup failed for %s: %s", year, exc)
    return None
