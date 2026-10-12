"""02 - Live satellite ingestion (needs internet; runs on a laptop or in GitHub Actions).

    python pipelines/02_ingest_satellite.py --live --module corn            # Sentinel-2 NDVI/EVI
    python pipelines/02_ingest_satellite.py --live --module copper          # Sentinel-2 SWIR hotspots
    python pipelines/02_ingest_satellite.py --live --module permian --nightfire-dir ~/vnf
    python pipelines/02_ingest_satellite.py --live --module s5p             # Earth Engine SO2/NO2
    python pipelines/02_ingest_satellite.py --live --thumbnails             # true-colour AOI checks

Without --live the script only reports what it would do, so `bash run.sh` works offline.
Outputs append to outputs/satellite/*.csv and resume where a previous run stopped.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from gmcn.config import OUTPUTS, ensure_dirs, load_dotenv, load_geojson, settings
from gmcn.ingest import sentinel2 as s2

log = logging.getLogger("ingest")
SAT = OUTPUTS / "satellite"


def corn(cfg: dict, start: str | None, end: str | None) -> None:
    w = settings()["windows"]["corn"]
    start, end = start or w["start"], end or w["end"]
    fc = load_geojson("corn_belt.geojson")
    crop = {}
    if cfg["catalog"] == "planetary-computer":
        for f in fc["features"]:
            for y in range(int(start[:4]), int(end[:4]) + 1):
                href = s2.find_cdl_href(f["geometry"], y, cfg)
                if href:
                    crop[(f["properties"]["id"], str(y))] = href
        log.info("CDL corn masks found for %d AOI-years", len(crop))
    df = s2.run_vegetation(fc["features"], start, end, cfg, SAT / "corn_vegetation_scenes.csv",
                           crop_mask_hrefs=crop or None)
    log.info("corn: %d scene rows", len(df))


def copper(cfg: dict, start: str | None, end: str | None) -> None:
    w = settings()["windows"]["copper"]
    fc = load_geojson("copper_sites.geojson")
    df = s2.run_hotspots(fc["features"], start or w["start"], end or w["end"], cfg,
                         SAT / "copper_hotspot_scenes.csv")
    log.info("copper: %d scene rows", len(df))


def permian(start: str | None, end: str | None, local_dir: str | None) -> None:
    from gmcn.ingest import nightfire

    w = settings()["windows"]["permian"]
    poly = load_geojson("permian.geojson")["features"][0]["geometry"]
    df = nightfire.run(date.fromisoformat(start or w["start"]), date.fromisoformat(end or w["end"]), poly,
                       settings()["satellite"]["nightfire"], SAT / "permian_flares_daily.csv",
                       local_dir=Path(local_dir).expanduser() if local_dir else None)
    log.info("permian: %d days", len(df))


def s5p(start: str | None, end: str | None) -> None:
    from gmcn.ingest import earthengine as eeb

    w = settings()["windows"]["copper"]
    cfg = settings()["satellite"]["sentinel5p"]
    frames = []
    for f in load_geojson("copper_sites.geojson")["features"]:
        if f["properties"]["type"] != "smelter":
            continue
        for gas in ("SO2", "NO2"):
            frames.append(eeb.s5p_monthly_enhancement(f, gas, start or w["start"], end or w["end"], cfg))
    pd.concat(frames).to_csv(SAT / "smelter_s5p_monthly.csv", index=False)


def thumbnails(cfg: dict) -> None:
    """True-colour chips of every AOI from the clearest recent scene, to verify coordinates by eye."""
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path("figures/aoi_checks")
    out.mkdir(parents=True, exist_ok=True)
    keys = s2.BAND_KEYS[cfg["catalog"]]
    feats = load_geojson("corn_belt.geojson")["features"] + load_geojson("copper_sites.geojson")["features"]
    for f in feats:
        items = s2.search_items(f["geometry"], "2025-06-01", "2025-09-30", dict(cfg, max_scene_cloud_pct=10))
        if not items:
            continue
        it = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
        buf = float(f["properties"].get("buffer_m", 0)) * 3 or None
        red, grid = s2.reference_grid(it.assets[keys["red"]].href, f["geometry"], buffer_m=buf)
        green = s2.read_on_grid(it.assets["B03" if cfg["catalog"] == "planetary-computer" else "green"].href, grid)
        blue = s2.read_on_grid(it.assets[keys["blue"]].href, grid)
        rgb = np.dstack([red, green, blue]).astype(float)
        rgb = np.clip(rgb / np.nanpercentile(rgb, 98), 0, 1)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgb)
        ax.set_title(f["properties"]["name"], fontsize=9)
        ax.axis("off")
        fig.savefig(out / f"{f['properties']['id']}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually call satellite APIs")
    ap.add_argument("--module", default="corn", choices=["corn", "copper", "permian", "s5p", "all"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--nightfire-dir")
    ap.add_argument("--thumbnails", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ensure_dirs()
    load_dotenv()
    cfg = settings()["satellite"]["sentinel2"]
    if not a.live:
        print("Satellite ingestion skipped (offline mode). Re-run with --live on a machine with internet:\n"
              "  Sentinel-2 via Planetary Computer needs no key; Nightfire needs EOG credentials;\n"
              "  Sentinel-5P needs EE_PROJECT. Existing outputs/satellite/*.csv files are kept.")
        raise SystemExit(0)
    if a.thumbnails:
        thumbnails(cfg)
    mods = ["corn", "copper", "permian", "s5p"] if a.module == "all" else [a.module]
    for m in mods:
        if m == "corn":
            corn(cfg, a.start, a.end)
        elif m == "copper":
            copper(cfg, a.start, a.end)
        elif m == "permian" and (a.nightfire_dir or settings()["satellite"]["nightfire"]["url_template"]):
            permian(a.start, a.end, a.nightfire_dir)
        elif m == "s5p" and os.environ.get("EE_PROJECT"):
            s5p(a.start, a.end)
        else:
            log.info("skip %s (credentials or source not configured)", m)
