import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from conftest import UTM, X0, Y0, aoi_polygon
from gmcn.ingest import sentinel2 as s2

TRUE_NDVI = (0.40 - 0.05) / (0.40 + 0.05)


def test_ndvi_recovered_and_clouds_masked(scene_factory, s2_cfg):
    item = scene_factory("post2022", baseline="05.10")
    row = s2.vegetation_stats(item, aoi_polygon(), s2_cfg, "planetary-computer")
    assert row["ndvi_median"] == pytest.approx(TRUE_NDVI, abs=1e-6)
    # 25 % of the tile is cloud; the AOI is inset 100 m so the masked share is a bit lower
    assert 0.70 < row["valid_frac"] < 0.78
    assert row["boa_offset"] == -1000


def test_baseline_harmonization_makes_scenes_comparable(scene_factory, s2_cfg):
    old = s2.vegetation_stats(scene_factory("pre2022", baseline="03.01"), aoi_polygon(), s2_cfg, "planetary-computer")
    new = s2.vegetation_stats(scene_factory("post2022", baseline="05.10"), aoi_polygon(), s2_cfg, "planetary-computer")
    assert old["boa_offset"] == 0.0
    assert old["ndvi_median"] == pytest.approx(new["ndvi_median"], abs=1e-9)


def test_without_harmonization_ndvi_would_be_biased(scene_factory, s2_cfg):
    cfg = dict(s2_cfg, boa_offset=0)  # simulate forgetting the offset
    row = s2.vegetation_stats(scene_factory("post2022", baseline="05.10"), aoi_polygon(), cfg, "planetary-computer")
    assert row["ndvi_median"] < TRUE_NDVI - 0.1


def test_crop_mask_reprojected_from_albers(scene_factory, s2_cfg, cdl_raster):
    item = scene_factory("mixed", right_half_ndvi_low=True, cloud_quadrant=False)
    unmasked = s2.vegetation_stats(item, aoi_polygon(), s2_cfg, "planetary-computer")
    masked = s2.vegetation_stats(item, aoi_polygon(), s2_cfg, "planetary-computer",
                                 crop_mask_href=str(cdl_raster), crop_class=1)
    assert masked["ndvi_median"] == pytest.approx(TRUE_NDVI, abs=1e-6)
    assert masked["n_aoi_pixels"] < 0.6 * unmasked["n_aoi_pixels"]


def test_hotspot_counts_saturated_hot_pixels(scene_factory, s2_cfg):
    item = scene_factory("smelter", hotspot=True, cloud_quadrant=False)
    # site at the centre of the 3 x 3 hot cluster (20 m pixels starting at row/col 60)
    cx, cy = X0 + (60 + 1.5) * 20, Y0 - (60 + 1.5) * 20
    lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(cx, cy)
    site = {"type": "Point", "coordinates": [lon, lat]}
    row = s2.hotspot_stats(item, site, 300, s2_cfg, "planetary-computer")
    assert row["hot_pixels"] == 9
    assert row["nhi_swir_max"] == pytest.approx(0.25, abs=1e-6)
    # vegetation masking would have discarded those saturated pixels; hotspot masking keeps them
    assert 1 not in s2_cfg["scl_mask_hotspot"]


def test_no_hotspots_in_plain_scene(scene_factory, s2_cfg):
    item = scene_factory("plain", cloud_quadrant=False)
    row = s2.hotspot_stats(item, aoi_polygon(), 0, s2_cfg, "planetary-computer")
    assert row["hot_pixels"] == 0


def test_batch_runner_is_resumable(scene_factory, s2_cfg, tmp_path):
    items = [scene_factory("a"), scene_factory("b", baseline="03.01")]
    feats = [{"type": "Feature", "properties": {"id": "aoi1"}, "geometry": aoi_polygon()}]
    out = tmp_path / "veg.csv"
    df1 = s2.run_vegetation(feats, "2025-01-01", "2025-12-31", s2_cfg, out, items_by_feature={"aoi1": items})
    df2 = s2.run_vegetation(feats, "2025-01-01", "2025-12-31", s2_cfg, out, items_by_feature={"aoi1": items})
    assert len(df1) == 2 and len(df2) == 2
    assert set(df2.item_id) == {"a", "b"}
