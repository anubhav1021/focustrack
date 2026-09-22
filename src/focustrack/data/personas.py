"""Participant personas.

Forty participants across five roles. Each carries the traits the behaviour
model needs: how fast they type, how much they use the mouse, how easily they
are distracted, and when their day starts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import CHRONOTYPES, ROLES

#: Keystrokes per minute while actively typing. A writer types about five
#: times faster than a designer, which is why activity counts have to be
#: normalised per user before any model sees them.
ROLE_TYPING_RATE: dict[str, float] = {
    "writer": 190.0,
    "support": 120.0,
    "developer": 95.0,
    "analyst": 70.0,
    "designer": 38.0,
}

#: Relative mouse usage - designers live on the pointer, writers barely touch it.
ROLE_MOUSE_FACTOR: dict[str, float] = {
    "designer": 2.4,
    "analyst": 1.3,
    "support": 1.1,
    "developer": 0.9,
    "writer": 0.55,
}

#: Per-role mix over the six application categories while focused.
ROLE_APP_MIX: dict[str, dict[str, float]] = {
    "developer": {
        "development": 0.62, "research": 0.20, "docs_analysis": 0.08,
        "communication": 0.09, "creative": 0.01, "personal": 0.00,
    },
    "designer": {
        "creative": 0.66, "research": 0.14, "docs_analysis": 0.08,
        "communication": 0.10, "development": 0.02, "personal": 0.00,
    },
    "writer": {
        "docs_analysis": 0.60, "research": 0.23, "communication": 0.12,
        "creative": 0.04, "development": 0.01, "personal": 0.00,
    },
    "analyst": {
        "docs_analysis": 0.58, "research": 0.19, "development": 0.11,
        "communication": 0.11, "creative": 0.01, "personal": 0.00,
    },
    "support": {
        "communication": 0.55, "docs_analysis": 0.18, "research": 0.17,
        "development": 0.06, "creative": 0.04, "personal": 0.00,
    },
}

CHRONOTYPE_PROBS: dict[str, float] = {"morning": 0.35, "intermediate": 0.45, "evening": 0.20}


@dataclass(frozen=True)
class UserPersona:
    """One participant and the traits that drive their simulated behaviour."""

    user_id: str
    role: str
    years_experience: float
    chronotype: str
    #: Personal distractibility in [0, 1]; scales the focus-to-distraction hazard.
    distractibility: float
    #: Multiplier on the role's typing rate.
    typing_factor: float
    #: Multiplier on the role's mouse usage.
    mouse_factor: float
    #: Multiplier on the role's meeting rate.
    meeting_factor: float
    #: Mean tracked minutes on a workday.
    day_length_mean: float
    #: Hour of the local clock the workday tends to start.
    start_hour: float

    @property
    def typing_rate(self) -> float:
        """Keystrokes per minute of active typing for this participant."""
        return ROLE_TYPING_RATE[self.role] * self.typing_factor

    @property
    def mouse_scale(self) -> float:
        return ROLE_MOUSE_FACTOR[self.role] * self.mouse_factor

    @property
    def app_mix(self) -> dict[str, float]:
        return ROLE_APP_MIX[self.role]

    def to_row(self) -> dict[str, Any]:
        """The public ``users.csv`` row - traits stay internal to the simulator."""
        return {
            "user_id": self.user_id,
            "role": self.role,
            "years_experience": round(self.years_experience, 1),
            "chronotype": self.chronotype,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_users(cfg: Config, rng: np.random.Generator | None = None) -> list[UserPersona]:
    """Build the 40-participant cohort described in the config."""
    rng = rng or np.random.default_rng(cfg.seed)
    role_counts: dict[str, int] = cfg["dataset.role_counts"]
    unknown = set(role_counts) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown roles in config: {sorted(unknown)}")

    beta_a, beta_b = cfg["behaviour.distractibility_beta"]
    chrono_start: dict[str, float] = cfg["dataset.chronotype_start_hour"]
    start_sd = float(cfg["dataset.start_hour_sd"])
    day_mean = float(cfg["dataset.workday.mean_minutes"])

    chrono_names = list(CHRONOTYPES)
    chrono_p = np.array([CHRONOTYPE_PROBS[c] for c in chrono_names], dtype=float)
    chrono_p /= chrono_p.sum()

    personas: list[UserPersona] = []
    index = 1
    for role in ROLES:
        for _ in range(int(role_counts.get(role, 0))):
            chronotype = str(rng.choice(chrono_names, p=chrono_p))
            years = float(np.clip(rng.lognormal(mean=1.15, sigma=0.62), 0.5, 22.0))
            # More experienced participants are a little harder to distract.
            trait = float(rng.beta(beta_a, beta_b))
            trait = float(np.clip(trait * (1.0 - 0.012 * min(years, 15.0)), 0.02, 0.98))
            personas.append(
                UserPersona(
                    user_id=f"U{index:03d}",
                    role=role,
                    years_experience=years,
                    chronotype=chronotype,
                    distractibility=trait,
                    typing_factor=float(np.clip(rng.normal(1.0, 0.17), 0.55, 1.6)),
                    mouse_factor=float(np.clip(rng.normal(1.0, 0.20), 0.5, 1.8)),
                    meeting_factor=float(np.clip(rng.normal(1.0, 0.30), 0.3, 2.2)),
                    day_length_mean=float(np.clip(rng.normal(day_mean, 28.0), 380.0, 610.0)),
                    start_hour=float(rng.normal(chrono_start[chronotype], start_sd)),
                )
            )
            index += 1
    return personas


def users_frame(personas: list[UserPersona]) -> pd.DataFrame:
    """The published ``users.csv``: role, experience and chronotype only."""
    return pd.DataFrame([p.to_row() for p in personas])
