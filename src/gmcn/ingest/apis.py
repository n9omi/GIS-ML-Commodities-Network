"""Live refresh of the raw market, macro, production and trade snapshots.

Each function calls the provider's public API with a key from .env and writes the raw JSON to
data/raw/, in the same format the committed snapshots use. Run via
`python pipelines/01_build_snapshots.py --refresh`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from gmcn.config import DATA_RAW


def _save(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def fred(series_id: str, start: str) -> Path:
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "observation_start": start,
                "api_key": os.environ["FRED_API_KEY"], "file_type": "json"},
        timeout=60,
    )
    r.raise_for_status()
    return _save(r.json(), DATA_RAW / "fred" / f"{series_id}.json")


def eia(series_id_v1: str, name: str) -> Path:
    r = requests.get(f"https://api.eia.gov/v2/seriesid/{series_id_v1}",
                     params={"api_key": os.environ["EIA_API_KEY"]}, timeout=60)
    r.raise_for_status()
    return _save(r.json(), DATA_RAW / "eia" / f"{name}.json")


def alphavantage(params: dict, name: str) -> Path:
    q = dict(params, apikey=os.environ["ALPHAVANTAGE_API_KEY"], datatype="json")
    r = requests.get("https://www.alphavantage.co/query", params=q, timeout=60)
    r.raise_for_status()
    time.sleep(1.2)  # free tier: about one request per second
    return _save(r.json(), DATA_RAW / "alphavantage" / f"{name}.json")


def usitc(body: dict, name: str) -> Path:
    r = requests.post(
        "https://datawebws.usitc.gov/dataweb/api/v2/report2/runReport",
        headers={"Authorization": f"Bearer {os.environ['USITC_DATAWEB_TOKEN']}",
                 "Content-Type": "application/json"},
        data=json.dumps(body), timeout=180,
    )
    r.raise_for_status()
    return _save(r.json(), DATA_RAW / "usitc" / f"{name}.json")


def usitc_body(trade_type: str, measure: str, hts_codes: list[str], years: list[str]) -> dict:
    """DataWeb request body: HTS6 lines broken out by partner country, annual totals."""
    return {
        "savedQueryName": "",
        "reportOptions": {"tradeType": trade_type, "classificationSystem": "HTS"},
        "searchOptions": {
            "componentSettings": {"dataToReport": [measure], "scale": "1",
                                  "timeframeSelectType": "fullYears", "years": years,
                                  "startDate": None, "endDate": None, "startMonth": None,
                                  "endMonth": None, "yearsTimeline": "Annual"},
            "countries": {"aggregation": "Break Out Countries", "countries": [],
                          "countriesExpanded": [{"name": "All Countries", "value": "all"}],
                          "countriesSelectType": "all",
                          "countryGroups": {"systemGroups": [], "userGroups": []}},
            "commodities": {"aggregation": "Break Out Commodities", "codeDisplayFormat": "YES",
                            "commodities": hts_codes, "commoditiesExpanded": [],
                            "commoditiesManual": "",
                            "commodityGroups": {"systemGroups": [], "userGroups": []},
                            "commoditySelectType": "list", "granularity": "6",
                            "groupGranularity": None, "searchGranularity": None},
            "MiscGroup": {
                "districts": {"aggregation": "Aggregate District", "districtGroups": {"userGroups": []},
                              "districts": [], "districtsExpanded": [{"name": "All Districts", "value": "all"}],
                              "districtsSelectType": "all"},
                "importPrograms": {"aggregation": None, "importPrograms": [], "programsSelectType": "all"},
                "extImportPrograms": {"aggregation": "Aggregate CSC", "extImportPrograms": [],
                                      "extImportProgramsExpanded": [], "programsSelectType": "all"},
                "provisionCodes": {"aggregation": "Aggregate RPCODE", "provisionCodesSelectType": "all",
                                   "rateProvisionCodes": [], "rateProvisionCodesExpanded": []},
            },
        },
        "sortingAndDataFormat": {
            "DataSort": {"columnOrder": [], "fullColumnOrder": [], "sortOrder": []},
            "reportCustomizations": {"exportCombineTables": False, "showAllSubtotal": True,
                                     "subtotalRecords": "", "totalRecords": "20000", "exportRawData": False},
        },
    }
