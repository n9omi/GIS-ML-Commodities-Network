"""03 - Satellite scenes -> regular activity indices, anomalies and dated structural breaks.

Runs only on whatever outputs/satellite/*.csv exist; reports and exits cleanly otherwise.
"""

from __future__ import annotations

import pandas as pd

from gmcn.config import OUTPUTS, events, settings
from gmcn.features.changepoint import pelt_breaks
from gmcn.features.composite import climatology_anomaly, monthly_hotspot_activity, weekly_composite

SAT = OUTPUTS / "satellite"


def corn() -> None:
    p = SAT / "corn_vegetation_scenes.csv"
    if not p.exists():
        print("corn: no scenes yet (run 02 with --live --module corn)")
        return
    cfg = settings()["satellite"]["sentinel2"]
    scenes = pd.read_csv(p)
    wk = weekly_composite(scenes, "aoi_id", ["ndvi_median", "evi_median"], cfg["min_valid_fraction"])
    wk = climatology_anomaly(wk, "aoi_id", "ndvi_median")
    wk.to_csv(SAT / "corn_ndvi_weekly.csv", index=False)
    # season summary: peak greenness and its anomaly per AOI-year (a classic yield proxy)
    wk["year"] = wk["week"].dt.year
    season = (wk[wk["week"].dt.month.between(6, 9)].groupby(["aoi_id", "year"])
              .agg(peak_ndvi=("ndvi_median", "max"), mean_jja_anom=("ndvi_median_anom_z", "mean"),
                   observed_weeks=("observed", "sum")).reset_index())
    season.to_csv(SAT / "corn_season_summary.csv", index=False)
    print(f"corn: {len(wk)} AOI-weeks, {season.shape[0]} AOI-seasons")


def copper() -> None:
    p = SAT / "copper_hotspot_scenes.csv"
    if not p.exists():
        print("copper: no scenes yet (run 02 with --live --module copper)")
        return
    cfg = settings()["satellite"]["sentinel2"]
    m = monthly_hotspot_activity(pd.read_csv(p), cfg["min_valid_fraction"])
    m.to_csv(SAT / "copper_activity_monthly.csv", index=False)
    rows = []
    ev = {e["id"]: e for e in events()}
    for site, g in m.groupby("site_id"):
        s = g.set_index("month")["hot_pixels_per_look"].sort_index()
        for b in pelt_breaks(s, min_size=3):
            nearest = min(ev.values(), key=lambda e: abs(pd.Timestamp(e["date"]) - b))
            rows.append({"site_id": site, "break_month": b.date().isoformat(),
                         "nearest_event": nearest["id"],
                         "days_from_event": int((b - pd.Timestamp(nearest["date"])).days)})
    pd.DataFrame(rows).to_csv(SAT / "copper_breaks.csv", index=False)
    print(f"copper: {len(m)} site-months, {len(rows)} dated breaks")


def permian() -> None:
    p = SAT / "permian_flares_daily.csv"
    if not p.exists():
        print("permian: no flare data yet (run 02 with --live --module permian)")
        return
    d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    mth = d.resample("MS").agg({"n_flares": "mean", "flare_rh_sum_mw": "mean"})
    mth.columns = ["flares_per_night", "flare_rh_mw_per_night"]
    mth.index.name = "month"
    mth.to_csv(SAT / "permian_flares_monthly.csv")
    print(f"permian: {len(mth)} months of flaring activity")


if __name__ == "__main__":
    corn()
    copper()
    permian()
