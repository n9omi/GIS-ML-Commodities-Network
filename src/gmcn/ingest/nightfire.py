"""VIIRS Nightfire (VNF) ingestion: nightly infrared detections -> basin-level flaring activity.

VNF is produced by the Earth Observation Group (EOG), Payne Institute, Colorado School of
Mines. Each nightly CSV lists sub-pixel combustion sources with a Planck-fit temperature
and radiant heat. Gas flares burn far hotter than biomass fires, so a temperature floor
isolates them. Summing radiant heat over a basin gives a daily upstream activity index.

Access needs a free EOG account. Credentials go in .env (see .env.example).
"""

from __future__ import annotations

import gzip
import io
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

log = logging.getLogger(__name__)

INVALID_SENTINELS = (999999, -999.3, 999999.0)


def get_token(token_url: str) -> str:
    """OIDC password-grant token for EOG downloads, from credentials in the environment."""
    import requests

    params = {
        "client_id": os.environ["EOG_CLIENT_ID"],
        "client_secret": os.environ["EOG_CLIENT_SECRET"],
        "username": os.environ["EOG_USERNAME"],
        "password": os.environ["EOG_PASSWORD"],
        "grant_type": "password",
    }
    r = requests.post(token_url, data=params, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def download_day(day: date, url_template: str, token: str, cache_dir: Path) -> Path:
    """Download one nightly VNF CSV (gzipped) into the cache, skipping existing files."""
    import requests

    url = url_template.format(date=day.strftime("%Y%m%d"))
    dest = cache_dir / Path(url).name
    if dest.exists():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def read_vnf_csv(path: Path, columns: dict) -> pd.DataFrame:
    """Read a VNF CSV (plain or .gz) into lat, lon, temp_k, rh_mw, date columns."""
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    df = pd.read_csv(io.BytesIO(raw))
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[columns["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[columns["lon"]], errors="coerce"),
        "temp_k": pd.to_numeric(df[columns["temp"]], errors="coerce"),
        "rh_mw": pd.to_numeric(df[columns["radiant_heat"]], errors="coerce"),
    })
    if columns.get("date") in df.columns:
        out["timestamp"] = pd.to_datetime(df[columns["date"]], errors="coerce", utc=True)
    for c in ("temp_k", "rh_mw"):
        out.loc[out[c].isin(INVALID_SENTINELS), c] = np.nan
    return out.dropna(subset=["lat", "lon"])


def aggregate_in_polygon(det: pd.DataFrame, polygon: dict, min_flare_temp_k: float) -> dict:
    """Daily activity summary for detections inside `polygon` (GeoJSON, WGS84)."""
    geom = shape(polygon)
    minx, miny, maxx, maxy = geom.bounds
    box = det[(det.lon.between(minx, maxx)) & (det.lat.between(miny, maxy))]
    inside = box[shapely.contains_xy(geom, box.lon.values, box.lat.values)]
    flares = inside[inside.temp_k >= min_flare_temp_k]
    return {
        "n_detections": int(len(inside)),
        "n_flares": int(len(flares)),
        "flare_rh_sum_mw": float(flares.rh_mw.sum()) if len(flares) else 0.0,
        "flare_rh_median_mw": float(flares.rh_mw.median()) if len(flares) else np.nan,
        "flare_temp_median_k": float(flares.temp_k.median()) if len(flares) else np.nan,
    }


def run(start: date, end: date, polygon: dict, cfg: dict, out_csv: Path,
        local_dir: Path | None = None, cache_dir: Path | None = None) -> pd.DataFrame:
    """Build a daily basin flaring series from local CSVs or live EOG downloads."""
    cols = cfg["columns"]
    token = None
    rows = []
    day = start
    while day <= end:
        path = None
        if local_dir is not None:
            hits = sorted(local_dir.glob(f"*d{day:%Y%m%d}*.csv*"))
            path = hits[0] if hits else None
        elif cfg.get("url_template"):
            token = token or get_token(cfg["token_url"])
            try:
                path = download_day(day, cfg["url_template"], token, cache_dir or Path("data/cache/vnf"))
            except Exception as exc:
                log.warning("VNF %s unavailable: %s", day, exc)
        else:
            raise RuntimeError("Set satellite.nightfire.url_template in config/settings.yaml or pass --nightfire-dir")
        if path is not None:
            stats = aggregate_in_polygon(read_vnf_csv(path, cols), polygon, cfg["min_flare_temp_k"])
            stats["date"] = day.isoformat()
            rows.append(stats)
        day += timedelta(days=1)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out_csv, index=False)
    return df
