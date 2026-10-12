"""Spectral indices on Sentinel-2 L2A surface reflectance.

All functions take reflectance arrays (0-1 scale, NaN where invalid) and return float arrays.
"""

from __future__ import annotations

import numpy as np


def dn_to_reflectance(dn: np.ndarray, scale: float = 10000.0, offset: float = 0.0,
                      nodata: float | None = 0) -> np.ndarray:
    """Convert Sentinel-2 L2A digital numbers to surface reflectance.

    From processing baseline 04.00 (January 2022) ESA adds +1000 to every DN, so
    reflectance = (DN + BOA_ADD_OFFSET) / 10000 with BOA_ADD_OFFSET = -1000. Skipping
    this harmonization biases NDVI downward in post-2022 scenes and breaks time series.
    """
    out = dn.astype("float64")
    if nodata is not None:
        out[dn == nodata] = np.nan
    return (out + offset) / scale


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized difference vegetation index, (NIR - Red) / (NIR + Red)."""
    return _safe_ratio(nir - red, nir + red)


def evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Enhanced vegetation index; less saturated than NDVI over dense mid-season corn."""
    return _safe_ratio(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0)


def nhi_swir(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """Normalized hotspot index on SWIR, (B12 - B11) / (B12 + B11).

    High-temperature surfaces (molten slag, converters, flares) emit far more in B12
    (2.19 um) than B11 (1.61 um), so the index turns positive over hot pixels.
    """
    return _safe_ratio(b12 - b11, b12 + b11)


def nhi_swnir(b8a: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Normalized hotspot index on SWIR/NIR, (B11 - B8A) / (B11 + B8A); catches the hottest sources."""
    return _safe_ratio(b11 - b8a, b11 + b8a)


def hot_pixel_mask(b8a: np.ndarray, b11: np.ndarray, b12: np.ndarray,
                   nhi_threshold: float = 0.0, min_b12: float = 0.15) -> np.ndarray:
    """Boolean hot-pixel flag: either hotspot index above threshold and B12 bright enough.

    The B12 floor screens dark pixels where tiny reflectances make the ratios unstable.
    """
    a = nhi_swir(b11, b12)
    b = nhi_swnir(b8a, b11)
    flag = ((a > nhi_threshold) | (b > nhi_threshold)) & (b12 >= min_b12)
    return np.where(np.isnan(a) & np.isnan(b), False, flag)


def scl_valid(scl: np.ndarray, masked_classes: list[int]) -> np.ndarray:
    """True where the Scene Classification Layer class is not in `masked_classes`."""
    return ~np.isin(scl, np.asarray(masked_classes))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity; the change score between two embedding vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / denom)
