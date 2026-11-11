"""Shared helpers: config loading, path resolution, logging."""
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_dir(cfg: dict, key: str) -> Path:
    """Resolve (and create) one of the directories listed under `paths` in the config."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def write_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
