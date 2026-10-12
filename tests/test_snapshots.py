"""Data-integrity checks on the committed raw snapshots."""

import json

import pytest

from gmcn.config import DATA_RAW
from gmcn.ingest import snapshots as sn


@pytest.mark.parametrize("fname,flow", [("genimp_2023_2025_hts6.json", "import"), ("totexp_2023_2025_hts6.json", "export")])
def test_usitc_rows_sum_to_dataweb_totals(fname, flow):
    path = DATA_RAW / "usitc" / fname
    tidy = sn.usitc_table(path, flow)
    total = json.loads(path.read_text())["dto"]["tables"][0]["total"]["values"]
    years = sorted(tidy.year.unique())
    reported = [float(v.replace(",", "")) for v in total[-len(years):]]
    for y, rep in zip(years, reported):
        assert tidy.loc[tidy.year == y, "value_usd"].sum() == pytest.approx(rep, rel=1e-9)


@pytest.mark.parametrize("sid", ["GS10", "DFII10", "FEDFUNDS", "UNRATE", "DTWEXBGS"])
def test_fred_parser_keeps_every_published_value(sid):
    path = DATA_RAW / "fred" / f"{sid}.json"
    raw = json.loads(path.read_text())["observations"]
    s = sn.fred_series(path)
    assert len(s) == sum(o["value"] != "." for o in raw)
    assert s.index.is_monotonic_increasing


def test_eia_and_weekly_parsers():
    us = sn.eia_series(DATA_RAW / "eia" / "MCRFPUS2_monthly.json")
    tx = sn.eia_series(DATA_RAW / "eia" / "MCRFPTX2_monthly.json")
    assert (tx.reindex(us.index).dropna() < us.reindex(tx.index).dropna()).all()   # a state never exceeds the US
    fcx = sn.av_weekly_close(DATA_RAW / "alphavantage" / "FCX_weekly.json")
    assert fcx.name == "FCX" and fcx.index.is_monotonic_increasing
