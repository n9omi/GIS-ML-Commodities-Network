#!/usr/bin/env bash
# One command, whole pipeline.
#   bash run.sh            # offline: committed snapshots -> every table, figure and reports/RESULTS.md
#   bash run.sh --live     # also pulls Sentinel-2 (and Nightfire / Sentinel-5P if configured) first
#   bash run.sh --refresh  # also re-pulls market, macro, production and trade data (API keys in .env)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

LIVE=""; REFRESH=""
for a in "$@"; do
  case "$a" in
    --live) LIVE="--live" ;;
    --refresh) REFRESH="--refresh" ;;
  esac
done

python pipelines/01_build_snapshots.py $REFRESH
python pipelines/02_ingest_satellite.py $LIVE --module all
python pipelines/03_satellite_features.py
python pipelines/04_network.py
python pipelines/05_nowcast.py
python pipelines/06_regimes.py
python pipelines/07_local_projections.py
python pipelines/08_event_study.py
python pipelines/09_report.py
echo "Done. See reports/RESULTS.md and figures/."
