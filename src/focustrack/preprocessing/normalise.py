"""Step 9 - normalise activity counts within each participant.

A writer types about five times faster than a designer, so a raw keystroke
count says as much about who is at the keyboard as about what they are doing.
Every count is therefore rescaled against that participant's own distribution.

The statistics are computed from a participant's own history and nothing else,
so an unseen user can be normalised from their first days without the model
having met them - which is what makes the by-user train/test split honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import NORMALISED_COLUMNS
from focustrack.preprocessing.steps import Audit

EPS = 1e-6


@dataclass
class UserScaler:
    """Per-user location and scale for each normalised activity count."""

    method: str = "robust"
    stats: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: tuple[str, ...]) -> "UserScaler":
        for user, group in df.groupby("user_id", sort=False, observed=True):
            per_user: dict[str, tuple[float, float]] = {}
            for col in columns:
                values = group[col].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                per_user[col] = self._location_scale(values)
            self.stats[str(user)] = per_user
        return self

    def _location_scale(self, values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 1.0
        if self.method == "p90":
            scale = float(np.percentile(values, 90))
            return 0.0, max(scale, EPS)
        # robust: median and the 10-90 spread, which survives the long right
        # tail of burst typing better than a standard deviation would.
        location = float(np.median(values))
        spread = float(np.percentile(values, 90) - np.percentile(values, 10))
        return location, max(spread, EPS)

    def transform(self, df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            loc = df["user_id"].map(
                {u: s[col][0] for u, s in self.stats.items()}
            ).astype(float)
            scale = df["user_id"].map(
                {u: s[col][1] for u, s in self.stats.items()}
            ).astype(float)
            out[f"{col}_n"] = (df[col] - loc) / scale
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "stats": {
                u: {c: [float(a), float(b)] for c, (a, b) in cols.items()}
                for u, cols in self.stats.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserScaler":
        scaler = cls(method=payload.get("method", "robust"))
        scaler.stats = {
            u: {c: (float(v[0]), float(v[1])) for c, v in cols.items()}
            for u, cols in payload.get("stats", {}).items()
        }
        return scaler


def normalise_per_user(
    df: pd.DataFrame,
    cfg: Config,
    audit: Audit | None = None,
    scaler: UserScaler | None = None,
) -> tuple[pd.DataFrame, UserScaler]:
    """Add ``<column>_n`` normalised counterparts for the activity counts."""
    rows_in = len(df)
    method = str(cfg.get("preprocessing.normalisation", "robust"))
    scaler = scaler or UserScaler(method=method).fit(df, NORMALISED_COLUMNS)
    out = scaler.transform(df, NORMALISED_COLUMNS)

    if audit is not None:
        spreads = {
            col: round(
                float(np.median([s[col][1] for s in scaler.stats.values()])), 2
            )
            for col in NORMALISED_COLUMNS
        }
        audit.add(
            9, "normalise counts within each user", rows_in, len(out),
            len(scaler.stats),
            method=method,
            users=len(scaler.stats),
            median_scale=spreads,
        )
    return out, scaler
