"""Configuration loading.

``config.yaml`` is the single source of truth for every tunable number quoted
in the report. :class:`Config` wraps it in a small object that supports
dotted lookups (``cfg["reminders.max_breaks_per_day"]``) and resolves the
project's output paths relative to the project root.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Repository root - ``src/focustrack/config.py`` is three levels down.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

_MISSING = object()


@dataclass
class Config:
    """A dotted-path view over the parsed ``config.yaml``."""

    data: dict[str, Any] = field(default_factory=dict)
    source: Path = DEFAULT_CONFIG_PATH
    root: Path = PROJECT_ROOT

    # -- lookup --------------------------------------------------------------
    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Return the value at a dotted ``path``, e.g. ``"models.focus_drop"``."""
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"no such config key: {path!r}")
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def __contains__(self, path: str) -> bool:
        return self.get(path, None) is not None

    def section(self, name: str) -> dict[str, Any]:
        """Return a whole section as a plain dict."""
        value = self.get(name)
        if not isinstance(value, dict):
            raise TypeError(f"config key {name!r} is not a section")
        return value

    # -- derived values ------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 26138))

    def path(self, name: str) -> Path:
        """Resolve one of the ``paths:`` entries against the project root."""
        raw = self.get(f"paths.{name}")
        p = Path(raw)
        return p if p.is_absolute() else self.root / p

    def ensure_dirs(self) -> None:
        """Create every directory named under ``paths:``."""
        for name in self.section("paths"):
            self.path(name).mkdir(parents=True, exist_ok=True)

    def store_path(self, key: str) -> Path:
        """Resolve a ``storage:`` path (the live agent's database and key)."""
        p = Path(self.get(f"storage.{key}"))
        return p if p.is_absolute() else self.root / p

    # -- overriding ----------------------------------------------------------
    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with dotted-path ``overrides`` applied.

        ``cfg.with_overrides(**{"dataset.n_users": 4})`` is handy in tests and
        for the ``--quick`` smoke run.
        """
        data = copy.deepcopy(self.data)
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data=data, source=self.source, root=self.root)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load ``config.yaml`` (or an explicit ``path``)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    root = cfg_path.resolve().parent
    return Config(data=data, source=cfg_path.resolve(), root=root)
