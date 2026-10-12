"""Paths and settings shared by every pipeline step."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"


def ensure_dirs() -> None:
    for p in (DATA_PROC, OUTPUTS, OUTPUTS / "satellite", FIGURES, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def settings() -> dict:
    with open(CONFIG / "settings.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def commodities() -> dict:
    with open(CONFIG / "commodities.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def events() -> list[dict]:
    with open(CONFIG / "events.yaml") as f:
        return yaml.safe_load(f)["events"]


def load_geojson(name: str) -> dict:
    with open(CONFIG / "aois" / name) as f:
        return json.load(f)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency). Existing environment variables win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
