"""Parsers for the raw API responses committed under data/raw/.

Each raw file is the provider's own JSON response, byte for byte, so the parsers here work
identically on committed snapshots and on fresh pulls from `gmcn.ingest.apis`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _num(x) -> float:
    if x is None:
        return np.nan
    s = str(x).replace(",", "").strip()
    if s in {"", ".", "--", "None", "null"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def fred_series(path: Path) -> pd.Series:
    """FRED `series/observations` JSON -> float Series indexed by date ('.' = missing)."""
    obs = json.loads(Path(path).read_text())["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): _num(o["value"]) for o in obs}).sort_index()
    return s.dropna()


def av_series(path: Path) -> pd.Series:
    """Alpha Vantage economic/commodity JSON ({data: [{date, value}]}) -> float Series."""
    data = json.loads(Path(path).read_text())["data"]
    s = pd.Series({pd.Timestamp(d["date"]): _num(d["value"]) for d in data}).sort_index()
    return s.dropna()


def av_weekly_close(path: Path) -> pd.Series:
    """Alpha Vantage TIME_SERIES_WEEKLY JSON -> weekly close (not dividend-adjusted)."""
    d = json.loads(Path(path).read_text())
    ts = d["Weekly Time Series"]
    s = pd.Series({pd.Timestamp(k): _num(v["4. close"]) for k, v in ts.items()}).sort_index()
    s.name = d["Meta Data"]["2. Symbol"]
    return s.dropna()


def eia_series(path: Path) -> pd.Series:
    """EIA API v2 JSON (seriesid route) -> monthly Series indexed by month start."""
    rows = json.loads(Path(path).read_text())["response"]["data"]
    s = pd.Series({pd.Timestamp(r["period"] + "-01"): _num(r["value"]) for r in rows}).sort_index()
    return s.dropna()


def usitc_table(path: Path, flow: str) -> pd.DataFrame:
    """USITC DataWeb runReport JSON (commodities x countries x years) -> tidy long table."""
    d = json.loads(Path(path).read_text())
    tb = d["dto"]["tables"][0]
    cols = [c["label"] for g in tb["column_groups"] for c in g["columns"]]
    years = [c for c in cols if c.isdigit()]
    recs = []
    for r in tb["row_groups"][0]["rowsNew"]:
        vals = [e.get("value") for e in r["rowEntries"]]
        row = dict(zip(cols, vals))
        for y in years:
            recs.append({
                "hts6": str(row["HTS Number"]),
                "description": row["Description"],
                "country": row["Country"],
                "year": y,
                "value_usd": _num(row[y]),
                "flow": flow,
            })
    out = pd.DataFrame(recs)
    out["value_usd"] = out["value_usd"].fillna(0.0)
    return out


def citation(path: Path) -> dict:
    """Return the `_citation` block attached by the data gateway, if present."""
    try:
        return json.loads(Path(path).read_text()).get("_citation", {})
    except Exception:
        return {}
