"""01 - Parse committed raw API responses into tidy CSVs (and optionally refresh them live).

    python pipelines/01_build_snapshots.py            # parse committed snapshots
    python pipelines/01_build_snapshots.py --refresh  # re-pull from APIs (needs keys in .env)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gmcn.config import DATA_PROC, DATA_RAW, commodities, ensure_dirs, load_dotenv, settings
from gmcn.ingest import snapshots as sn

RAW = {
    "fred": ["GS10", "DFII10", "FEDFUNDS", "UNRATE", "DTWEXBGS"],
    "eia": {"us": "MCRFPUS2_monthly", "tx": "MCRFPTX2_monthly", "nm": "MCRFPNM2_monthly"},
    "av_weekly": ["FCX", "SPY", "CPER", "CORN"],
}


def refresh() -> None:
    from gmcn.ingest import apis

    load_dotenv()
    for sid, start in [("GS10", "1962-01-01"), ("DFII10", "2003-01-01"), ("FEDFUNDS", "1954-07-01"),
                       ("UNRATE", "1948-01-01"), ("DTWEXBGS", "2006-01-01")]:
        apis.fred(sid, start)
    for code, name in [("PET.MCRFPUS2.M", "MCRFPUS2_monthly"), ("PET.MCRFPTX2.M", "MCRFPTX2_monthly"),
                       ("PET.MCRFPNM2.M", "MCRFPNM2_monthly")]:
        apis.eia(code, name)
    apis.alphavantage({"function": "WTI", "interval": "daily"}, "wti_daily")
    apis.alphavantage({"function": "CPI", "interval": "monthly"}, "cpi_monthly")
    for sym in RAW["av_weekly"]:
        apis.alphavantage({"function": "TIME_SERIES_WEEKLY", "symbol": sym}, f"{sym}_weekly")
    cfg = settings()["network"]
    com = commodities()
    apis.usitc(apis.usitc_body("GenImp", "GEN_CUSTOMS_VALUE", list(com["imports"]), cfg["years"]),
               "genimp_2023_2025_hts6")
    apis.usitc(apis.usitc_body("TotExp", "FAS_VALUE", list(com["exports"]), cfg["years"]),
               "totexp_2023_2025_hts6")


def build() -> None:
    ensure_dirs()
    # ---- macro, monthly
    gs10 = sn.fred_series(DATA_RAW / "fred/GS10.json")
    ff = sn.fred_series(DATA_RAW / "fred/FEDFUNDS.json")
    un = sn.fred_series(DATA_RAW / "fred/UNRATE.json")
    real10 = sn.fred_series(DATA_RAW / "fred/DFII10.json").resample("MS").mean()
    usd = sn.fred_series(DATA_RAW / "fred/DTWEXBGS.json").resample("MS").mean()
    cpi = sn.av_series(DATA_RAW / "alphavantage/cpi_monthly.json")
    macro = pd.concat({"gs10": gs10, "real10": real10, "fedfunds": ff, "unrate": un, "usd_broad": usd,
                       "cpi": cpi}, axis=1, sort=True)
    macro["cpi_yoy"] = 100 * (np.log(macro["cpi"]) - np.log(macro["cpi"].shift(12)))
    macro["usd_yoy"] = 100 * (np.log(macro["usd_broad"]) - np.log(macro["usd_broad"].shift(12)))
    macro.index.name = "month"
    macro.to_csv(DATA_PROC / "macro_monthly.csv")

    # ---- prices
    wti = sn.av_series(DATA_RAW / "alphavantage/wti_daily.json")
    wti_m = wti.resample("MS").mean().rename("wti")
    wti_m.index.name = "month"
    wti_m.to_frame().to_csv(DATA_PROC / "wti_monthly.csv")
    weekly = pd.concat({s: sn.av_weekly_close(DATA_RAW / f"alphavantage/{s}_weekly.json")
                        for s in RAW["av_weekly"]}, axis=1, sort=True)
    weekly.index.name = "week_ending"
    weekly.to_csv(DATA_PROC / "prices_weekly.csv")

    # ---- production
    prod = pd.concat({k: sn.eia_series(DATA_RAW / f"eia/{v}.json") for k, v in RAW["eia"].items()}, axis=1, sort=True)
    prod["permian_proxy_tx_nm"] = prod["tx"] + prod["nm"]
    prod.index.name = "month"
    prod.to_csv(DATA_PROC / "crude_production_monthly.csv")

    # ---- trade
    imp = sn.usitc_table(DATA_RAW / "usitc/genimp_2023_2025_hts6.json", "import")
    exp = sn.usitc_table(DATA_RAW / "usitc/totexp_2023_2025_hts6.json", "export")
    imp.to_csv(DATA_PROC / "trade_imports_hts6.csv", index=False)
    exp.to_csv(DATA_PROC / "trade_exports_hts6.csv", index=False)

    write_sources()
    print(f"macro {macro.index.min():%Y-%m}..{macro.index.max():%Y-%m}; prices weekly {len(weekly)} rows; "
          f"production {prod.index.min():%Y-%m}..{prod.index.max():%Y-%m}; trade {len(imp)}+{len(exp)} rows")


def write_sources() -> None:
    """Provenance table for every raw file: publisher, public URL, access date."""
    lines = ["# Raw data sources", "",
             "Every file in this folder is the provider's unmodified API response. Snapshots were pulled",
             "on 2026-10-11/12 through Claude connectors (USAFacts gateway for FRED, EIA and USITC;",
             "Alpha Vantage MCP) and can be re-pulled with `python pipelines/01_build_snapshots.py --refresh`.", "",
             "| File | Publisher | Series / query | Public page |", "| --- | --- | --- | --- |"]
    for p in sorted(DATA_RAW.rglob("*.json")):
        c = sn.citation(p)
        rel = p.relative_to(DATA_RAW)
        if c and "fred" in str(rel):
            lines.append(f"| `{rel}` | {c.get('publisher', '')} | {p.stem} | {c.get('public_url', '')} |")
        elif c:
            lines.append(f"| `{rel}` | {c.get('publisher', '')} | {c.get('resource_name') or c.get('title', '')} "
                         f"| {c.get('public_url', '')} |")
        elif "alphavantage" in str(rel):
            lines.append(f"| `{rel}` | Alpha Vantage | {p.stem} | https://www.alphavantage.co/documentation/ |")
        elif "fred" in str(rel):
            lines.append(f"| `{rel}` | Federal Reserve Bank of St. Louis (FRED) | {p.stem} "
                         f"| https://fred.stlouisfed.org/series/{p.stem} |")
        else:
            lines.append(f"| `{rel}` |  | {p.stem} |  |")
    lines += ["", "FRED: This product uses the FRED API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.",
              "USITC: Trade figures are official U.S. merchandise trade statistics published by the U.S. Census Bureau, "
              "compiled via the U.S. International Trade Commission's DataWeb.",
              "EIA: U.S. Energy Information Administration, Petroleum Supply Monthly (field production of crude oil).",
              "Alpha Vantage: weekly closes are not dividend-adjusted."]
    (DATA_RAW / "SOURCES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        refresh()
    build()
