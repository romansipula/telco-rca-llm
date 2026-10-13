"""YAML configuration with optional ``base:`` inheritance and ``--set key.sub=value`` overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(base, cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
