from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """Dict with attribute access and dotted-path resolution relative to REPO_ROOT."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return _wrap(value)

    def resolve_path(self, dotted_key: str) -> Path:
        value = self
        for part in dotted_key.split("."):
            value = value[part]
        return resolve_path(value)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config(value)
    return value


def resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
