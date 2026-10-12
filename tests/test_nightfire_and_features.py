import gzip
from datetime import date

import numpy as np
import pandas as pd
import pytest

from gmcn.features.changepoint import pelt_breaks
from gmcn.features.composite import climatology_anomaly, monthly_hotspot_activity, weekly_composite
from gmcn.features.indices import cosine_distance, evi, hot_pixel_mask, ndvi
from gmcn.ingest import nightfire
from gmcn.ingest.earthengine import embedding_change, enhancement

COLS = {"lat": "Lat_GMTCO", "lon": "Lon_GMTCO", "temp": "Temp_BB", "radiant_heat": "RH", "date": "Date_Mscan"}
BASIN = {"type": "Polygon", "coordinates": [[[-104, 31], [-101, 31], [-101, 33], [-104, 33], [-104, 31]]]}
CFG = {"columns": COLS, "min_flare_temp_k": 1400, "url_template": None, "token_url": ""}


def _vnf_csv(path, rows):
    df = pd.DataFrame(rows, columns=["Lat_GMTCO", "Lon_GMTCO", "Temp_BB", "RH", "Date_Mscan"])
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)


def test_nightfire_counts_only_hot_detections_inside_basin(tmp_path):
    _vnf_csv(tmp_path / "VNF_npp_d20250102_noaa_v30-ez.csv.gz", [
        [32.0, -102.5, 1800, 2.0, "2025/01/02 08:00:00"],    # flare inside
        [32.1, -102.4, 1900, 3.0, "2025/01/02 08:00:00"],    # flare inside
        [32.2, -102.3, 1100, 5.0, "2025/01/02 08:00:00"],    # biomass fire inside (too cool)
        [29.0, -98.0, 1850, 4.0, "2025/01/02 08:00:00"],     # flare outside basin (Eagle Ford)
        [32.3, -102.2, 999999, 999999, "2025/01/02 08:00:00"],  # invalid fit
    ])
    out = nightfire.run(date(2025, 1, 2), date(2025, 1, 3), BASIN, CFG, tmp_path / "flares.csv",
                        local_dir=tmp_path)
    assert len(out) == 1  # only one day had a file
    r = out.iloc[0]
    assert r.n_detections == 4
    assert r.n_flares == 2
    assert r.flare_rh_sum_mw == pytest.approx(5.0)


def test_nightfire_requires_a_source():
    with pytest.raises(RuntimeError):
        nightfire.run(date(2025, 1, 2), date(2025, 1, 2), BASIN, CFG, None)


def test_indices_basic_values():
    red, nir, blue = np.array([0.05]), np.array([0.40]), np.array([0.03])
    assert ndvi(red, nir)[0] == pytest.approx(0.35 / 0.45)
    assert evi(blue, red, nir)[0] == pytest.approx(2.5 * 0.35 / (0.40 + 0.30 - 0.225 + 1))
    hot = hot_pixel_mask(np.array([0.2, 0.3]), np.array([0.3, 0.3]), np.array([0.5, 0.2]))
    assert hot.tolist() == [True, False]


def test_weekly_composite_marks_missing_weeks():
    scenes = pd.DataFrame({
        "aoi_id": ["a"] * 4,
        "datetime": ["2025-06-02T17:00:00Z", "2025-06-04T17:00:00Z", "2025-06-23T17:00:00Z", "2025-06-25T17:00:00Z"],
        "valid_frac": [0.9, 0.1, 0.8, 0.8],
        "ndvi_median": [0.5, 0.0, 0.7, 0.9],
    })
    wk = weekly_composite(scenes, "aoi_id", ["ndvi_median"], min_valid_frac=0.2)
    assert len(wk) == 4                       # weeks of Jun 2, 9, 16, 23
    assert wk.observed.tolist() == [True, False, False, True]
    assert wk.ndvi_median.iloc[0] == pytest.approx(0.5)   # cloudy scene excluded
    assert wk.ndvi_median.iloc[-1] == pytest.approx(0.8)  # median of 0.7 and 0.9


def test_climatology_anomaly_is_leave_one_year_out():
    weeks = pd.to_datetime(["2022-07-04", "2023-07-03", "2024-07-01", "2025-06-30"])
    df = pd.DataFrame({"aoi_id": "a", "week": weeks, "ndvi_median": [0.8, 0.82, 0.78, 0.5]})
    z = climatology_anomaly(df, "aoi_id", "ndvi_median")
    assert z["ndvi_median_anom_z"].iloc[-1] < -5     # the drought year stands out


def test_monthly_hotspot_activity():
    scenes = pd.DataFrame({
        "site_id": ["s"] * 4,
        "datetime": ["2025-09-01T03:00:00Z", "2025-09-11T03:00:00Z", "2025-10-01T03:00:00Z", "2025-10-21T03:00:00Z"],
        "valid_frac": [1.0, 1.0, 1.0, 1.0],
        "hot_pixels": [10, 8, 0, 0],
        "b12_hot_sum": [5.0, 4.0, 0.0, 0.0],
    })
    m = monthly_hotspot_activity(scenes, 0.2)
    assert m.active_look_share.tolist() == [1.0, 0.0]
    assert m.hot_pixels_per_look.tolist() == [9.0, 0.0]


def test_pelt_finds_a_level_shift():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(10, 0.5, 40), rng.normal(4, 0.5, 30)]
    s = pd.Series(x, index=pd.date_range("2023-01-01", periods=70, freq="W"))
    br = pelt_breaks(s)
    assert len(br) == 1
    assert abs(s.index.get_loc(br[0]) - 40) <= 1


def test_earth_engine_pure_functions():
    assert enhancement(3.0, 1.0) == 2.0
    assert np.isnan(enhancement(None, 1.0))
    v = np.ones(64) / 8
    assert embedding_change(v, v) == pytest.approx(0.0, abs=1e-12)
    assert cosine_distance(np.array([1, 0]), np.array([0, 1])) == pytest.approx(1.0)
