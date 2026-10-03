"""Load settings.yaml plus tuned weights into one dict-like config."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"
TUNED_WEIGHTS = ROOT / "config" / "weights.json"
STYLE_GUIDE = ROOT / "prompts" / "style_guide.md"
ARCHIVE = ROOT / "data" / "archive" / "episodes_2020.json"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Read settings.yaml, apply config/weights.json if present, then explicit overrides."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS
    with open(settings_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    tuned_path = settings_path.parent / "weights.json"
    if tuned_path.exists():
        tuned = json.loads(tuned_path.read_text())
        if tuned.get("weights"):
            cfg["weights"] = {k: float(v) for k, v in tuned["weights"].items()}
            cfg["weights_source"] = str(tuned_path)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg.setdefault("weights_source", "settings.yaml")
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
