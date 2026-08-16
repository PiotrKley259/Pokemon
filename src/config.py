"""Central config loader. Every module reads config.yaml through here."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the project root."""
    return PROJECT_ROOT / cfg["paths"][key]


def set_seeds(seed: int) -> None:
    """Seed python and numpy RNGs and log the seed used."""
    random.seed(seed)
    np.random.seed(seed)
    logging.getLogger("config").info("Random seed set to %d", seed)
