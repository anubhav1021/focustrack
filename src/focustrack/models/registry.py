"""Saving and loading fitted models.

Models are stored with the feature list and the code version they were fitted
under. Serving a model against a different feature order silently produces
nonsense, so :func:`load_bundle` checks rather than trusts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from focustrack import __version__
from focustrack.config import Config
from focustrack.preprocessing.features import FEATURE_COLUMNS

FOCUS_STATE_FILE = "focus_state.joblib"
FOCUS_DROP_FILE = "focus_drop.joblib"
METADATA_FILE = "model_metadata.json"


@dataclass
class ModelBundle:
    """The two fitted models plus the metadata needed to serve them safely."""

    focus_state: Any
    focus_drop: Any
    metadata: dict[str, Any]

    @property
    def features(self) -> list[str]:
        return list(self.metadata.get("features", FEATURE_COLUMNS))


def save_bundle(
    focus_state_model: Any,
    focus_drop_model: Any,
    cfg: Config,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist both models and their metadata under ``models/``."""
    out = cfg.path("models")
    out.mkdir(parents=True, exist_ok=True)

    state_path = out / FOCUS_STATE_FILE
    drop_path = out / FOCUS_DROP_FILE
    meta_path = out / METADATA_FILE

    joblib.dump(focus_state_model, state_path)
    joblib.dump(focus_drop_model, drop_path)

    payload = {
        "focustrack_version": __version__,
        "features": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        **(metadata or {}),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    return {"focus_state": state_path, "focus_drop": drop_path, "metadata": meta_path}


def load_bundle(cfg: Config) -> ModelBundle:
    """Load both models, refusing a bundle whose feature set no longer matches."""
    out = cfg.path("models")
    state_path = out / FOCUS_STATE_FILE
    drop_path = out / FOCUS_DROP_FILE
    meta_path = out / METADATA_FILE

    if not state_path.exists() or not drop_path.exists():
        raise FileNotFoundError(
            f"no fitted models in {out} - run `focustrack train` first."
        )

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    stored = list(metadata.get("features", FEATURE_COLUMNS))
    if stored != list(FEATURE_COLUMNS):
        raise ValueError(
            "the saved models were fitted on a different feature set.\n"
            f"  saved:   {len(stored)} features\n"
            f"  current: {len(FEATURE_COLUMNS)} features\n"
            "Re-run `focustrack train` to refit against the current pipeline."
        )

    return ModelBundle(
        focus_state=joblib.load(state_path),
        focus_drop=joblib.load(drop_path),
        metadata=metadata,
    )


def models_available(cfg: Config) -> bool:
    out = cfg.path("models")
    return (out / FOCUS_STATE_FILE).exists() and (out / FOCUS_DROP_FILE).exists()
