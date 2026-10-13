"""Configuration loading and path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class Config:
    """Resolved configuration. `raw` keeps the merged YAML for anything not modelled here."""

    raw: dict[str, Any]
    repo_root: Path
    crops: dict[str, Any] = field(default_factory=dict)
    factors: dict[str, Any] = field(default_factory=dict)
    changes: dict[str, Any] = field(default_factory=dict)

    # ---- paths -------------------------------------------------------------
    @property
    def raw_root(self) -> Path:
        # BBP_RAW_ROOT (e.g. set by Docker) wins over the YAML files
        p = Path(os.environ.get("BBP_RAW_ROOT") or self.raw["raw_root"])
        return p if p.is_absolute() else (self.repo_root / p).resolve()

    @property
    def data_dir(self) -> Path:
        p = Path(os.environ.get("BBP_DATA_DIR") or self.raw.get("data_dir", "data"))
        return p if p.is_absolute() else self.repo_root / p

    def input_path(self, key: str) -> Path | list[Path]:
        value = self.raw["inputs"][key]
        if isinstance(value, list):
            return [self.raw_root / v for v in value]
        return self.raw_root / value

    def sidra_tables(self) -> dict[int, dict[str, Any]]:
        """The `sidra` section with each table's files resolved against raw_root."""
        out = {}
        for table, spec in (self.raw.get("sidra") or {}).items():
            files = spec["files"] if isinstance(spec["files"], list) else [spec["files"]]
            out[int(table)] = {**spec, "files": [self.raw_root / f for f in files]}
        return out

    def input_files(self) -> list[tuple[str, Path]]:
        """Every configured input file as (label, path): `inputs` entries, then each SIDRA export."""
        files: list[tuple[str, Path]] = []
        for key in self.raw["inputs"]:
            paths = self.input_path(key)
            files.extend((key, p) for p in (paths if isinstance(paths, list) else [paths]))
        for table, spec in self.sidra_tables().items():
            files.extend((f"sidra_{table}", p) for p in spec["files"])
        return files

    def out(self, *parts: str) -> Path:
        path = self.data_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ---- convenience -------------------------------------------------------
    @property
    def years(self) -> list[int]:
        y = self.raw["years"]
        return list(range(int(y["start"]), int(y["end"]) + 1))

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})


def load_config(config_dir: Path | None = None) -> Config:
    config_dir = Path(config_dir) if config_dir else REPO_ROOT / "config"
    repo_root = config_dir.parent
    raw = _read_yaml(config_dir / "pipeline.yaml")
    local = config_dir / "paths.local.yaml"
    if local.exists():
        raw = _deep_update(raw, _read_yaml(local))
    return Config(
        raw=raw,
        repo_root=repo_root,
        crops=_read_yaml(config_dir / "crops.yaml")["crops"],
        factors=_read_yaml(config_dir / "conversion_factors.yaml"),
        changes=_read_yaml(config_dir / "municipality_changes.yaml"),
    )
