"""Chargement de la configuration du projet."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def load_config(path: Path = ROOT / "config.yaml") -> dict[str, Any]:
    """Charge config.yaml et renvoie un dictionnaire."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
