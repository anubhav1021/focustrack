"""Split by participant, not by window.

Windows from the same person are highly correlated, so a random split would
let the model recognise the person rather than the behaviour and report a
score it could never reproduce on a new user. Ten users - two from each role -
are held out entirely, and the model never sees a single window of theirs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import ROLES

SPLIT_FILE = "user_split.json"


@dataclass
class UserSplit:
    """Which participants are held out, and why."""

    train_users: list[str]
    test_users: list[str]
    by_role: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserSplit":
        return cls(
            train_users=list(payload["train_users"]),
            test_users=list(payload["test_users"]),
            by_role={k: list(v) for k, v in payload.get("by_role", {}).items()},
        )

    def mask(self, frame: pd.DataFrame, which: str) -> pd.Series:
        users = self.test_users if which == "test" else self.train_users
        return frame["user_id"].isin(users)

    def describe(self) -> str:
        return (
            f"  train  {len(self.train_users)} users\n"
            f"  test   {len(self.test_users)} users (unseen): "
            + ", ".join(self.test_users)
        )


def make_split(users: pd.DataFrame, cfg: Config, seed: int | None = None) -> UserSplit:
    """Hold out ``n_test_users_per_role`` participants from every role."""
    per_role = int(cfg["split.n_test_users_per_role"])
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    test: list[str] = []
    by_role: dict[str, list[str]] = {}
    for role in ROLES:
        members = sorted(users.loc[users["role"] == role, "user_id"].astype(str))
        if not members:
            continue
        take = min(per_role, len(members))
        chosen = sorted(rng.choice(members, size=take, replace=False).tolist())
        by_role[role] = chosen
        test.extend(chosen)

    test_set = set(test)
    train = sorted(u for u in users["user_id"].astype(str) if u not in test_set)
    if not train or not test:
        raise ValueError(
            f"the split leaves {len(train)} training and {len(test)} test users.\n"
            f"  {len(users)} participants, holding out "
            f"{per_role} per role from {len(by_role)} roles.\n"
            "  Reduce split.n_test_users_per_role, or add participants."
        )
    return UserSplit(train_users=train, test_users=sorted(test), by_role=by_role)


def save_split(split: UserSplit, cfg: Config) -> Path:
    path = cfg.path("processed") / SPLIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(split.to_dict(), fh, indent=2)
    return path


def load_split(cfg: Config) -> UserSplit:
    path = cfg.path("processed") / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `focustrack preprocess` first.")
    with path.open("r", encoding="utf-8") as fh:
        return UserSplit.from_dict(json.load(fh))
