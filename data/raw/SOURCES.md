# Raw data sources

Every file in this folder is the provider's unmodified API response. Snapshots were pulled
on 2026-10-11/12 through Claude connectors (USAFacts gateway for FRED, EIA and USITC;
Alpha Vantage MCP) and can be re-pulled with `python pipelines/01_build_snapshots.py --refresh`.

| File | Publisher | Series / query | Public page |
| --- | --- | --- | --- |
| `alphavantage/CORN_weekly.json` | Alpha Vantage | CORN_weekly | https://www.alphavantage.co/documentation/ |
| `alphavantage/CPER_weekly.json` | Alpha Vantage | CPER_weekly | https://www.alphavantage.co/documentation/ |
| `alphavantage/FCX_weekly.json` | Alpha Vantage | FCX_weekly | https://www.alphavantage.co/documentation/ |
| `alphavantage/SPY_weekly.json` | Alpha Vantage | SPY_weekly | https://www.alphavantage.co/documentation/ |
| `alphavantage/cpi_monthly.json` | Alpha Vantage | cpi_monthly | https://www.alphavantage.co/documentation/ |
| `alphavantage/wti_daily.json` | Alpha Vantage | wti_daily | https://www.alphavantage.co/documentation/ |
| `eia/MCRFPNM2_monthly.json` | U.S. Energy Information Administration | PET.MCRFPNM2.M | https://www.eia.gov/opendata/browser/petroleum/crd/crpdn?frequency=monthly&data=value;&facets=series;&series=MCRFPNM2;&sortColumn=period;&sortDirection=desc; |
| `eia/MCRFPTX2_monthly.json` | U.S. Energy Information Administration | PET.MCRFPTX2.M | https://www.eia.gov/opendata/browser/petroleum/crd/crpdn?frequency=monthly&data=value;&facets=series;&series=MCRFPTX2;&sortColumn=period;&sortDirection=desc; |
| `eia/MCRFPUS2_monthly.json` | U.S. Energy Information Administration | PET.MCRFPUS2.M | https://www.eia.gov/opendata/browser/petroleum/sum/snd?frequency=monthly&data=value;&facets=series;&series=MCRFPUS2;&sortColumn=period;&sortDirection=desc; |
| `fred/DFII10.json` | Federal Reserve Bank of St. Louis | DFII10 | https://fred.stlouisfed.org/series/DFII10 |
| `fred/DTWEXBGS.json` | Federal Reserve Bank of St. Louis | DTWEXBGS | https://fred.stlouisfed.org/series/DTWEXBGS |
| `fred/FEDFUNDS.json` | Federal Reserve Bank of St. Louis | FEDFUNDS | https://fred.stlouisfed.org/series/FEDFUNDS |
| `fred/GS10.json` | Federal Reserve Bank of St. Louis | GS10 | https://fred.stlouisfed.org/series/GS10 |
| `fred/UNRATE.json` | Federal Reserve Bank of St. Louis | UNRATE | https://fred.stlouisfed.org/series/UNRATE |
| `usitc/genimp_2023_2025_hts6.json` | U.S. International Trade Commission | General Imports: General Customs Value (In Actual Dollars) | https://dataweb.usitc.gov/trade/search/GenImp/HTS |
| `usitc/totexp_2023_2025_hts6.json` | U.S. International Trade Commission | Total Exports: FAS Value (In Actual Dollars) | https://dataweb.usitc.gov/trade/search/TotExp/HTS |

FRED: This product uses the FRED API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
USITC: Trade figures are official U.S. merchandise trade statistics published by the U.S. Census Bureau, compiled via the U.S. International Trade Commission's DataWeb.
EIA: U.S. Energy Information Administration, Petroleum Supply Monthly (field production of crude oil).
Alpha Vantage: weekly closes are not dividend-adjusted.
