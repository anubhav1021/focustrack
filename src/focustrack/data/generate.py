"""Generate and write the three dataset files described in Table 1.

``raw_activity_logs.csv``  minute-level counts plus the hidden state label
``daily_survey.csv``       end-of-day self-rated productivity and energy
``users.csv``              role, years of experience, chronotype

All three are written to ``data/raw/``. A clean, uncorrupted copy of the log
is kept alongside them so the preprocessing audit has ground truth to compare
its repairs against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import APP_CATALOGUE, STATES
from focustrack.data.corrupt import corrupt_logs
from focustrack.data.personas import build_users, users_frame
from focustrack.io_utils import write_csv, write_json

RAW_LOG_FILE = "raw_activity_logs.csv"
CLEAN_LOG_FILE = "clean_activity_logs.csv"
SURVEY_FILE = "daily_survey.csv"
USERS_FILE = "users.csv"
MANIFEST_FILE = "dataset_manifest.json"


@dataclass
class DatasetSummary:
    """The headline numbers quoted in section 2 of the report."""

    n_users: int
    n_workdays: int
    n_user_days: int
    n_minute_records: int
    n_applications: int
    state_distribution: dict[str, float]
    survey_response_rate: float
    corruption: dict[str, Any]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_users": self.n_users,
            "n_workdays": self.n_workdays,
            "n_user_days": self.n_user_days,
            "n_minute_records": self.n_minute_records,
            "n_applications": self.n_applications,
            "state_distribution": self.state_distribution,
            "survey_response_rate": self.survey_response_rate,
            "corruption": self.corruption,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def describe(self) -> str:
        lines = [
            f"  participants      {self.n_users}",
            f"  workdays          {self.n_workdays}",
            f"  user-days         {self.n_user_days}",
            f"  minute records    {self.n_minute_records:,}",
            f"  applications      {self.n_applications}",
            f"  survey response   {self.survey_response_rate:.1%}",
            "  state mix         "
            + ", ".join(f"{s} {self.state_distribution.get(s, 0.0):.1%}" for s in STATES),
        ]
        return "\n".join(lines)


def generate_dataset(
    cfg: Config,
    out_dir: Path | None = None,
    progress: bool = False,
) -> DatasetSummary:
    """Simulate the cohort, corrupt the log and write the dataset files."""
    # Imported here so ``focustrack.data`` can expose both without a cycle.
    from focustrack.data.simulate import simulate_cohort

    started = time.perf_counter()
    out = Path(out_dir) if out_dir is not None else cfg.path("raw")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    personas = build_users(cfg, rng)

    clean_logs, survey = simulate_cohort(personas, cfg, rng, progress=progress)
    raw_logs, corruption = corrupt_logs(clean_logs, cfg, rng)

    users = users_frame(personas)
    write_csv(users, out / USERS_FILE)
    write_csv(survey, out / SURVEY_FILE)
    write_csv(raw_logs, out / RAW_LOG_FILE)
    write_csv(clean_logs, out / CLEAN_LOG_FILE)

    n_user_days = int(
        clean_logs.assign(date=clean_logs["timestamp"].dt.normalize())
        .groupby(["user_id", "date"], observed=True)
        .ngroups
    )
    counts = clean_logs["state_label"].value_counts(normalize=True)
    responded = survey["productivity_rating"].notna().mean() if len(survey) else 0.0

    summary = DatasetSummary(
        n_users=len(personas),
        n_workdays=int(cfg["dataset.n_workdays"]),
        n_user_days=n_user_days,
        n_minute_records=len(clean_logs),
        n_applications=len(APP_CATALOGUE),
        state_distribution={s: float(counts.get(s, 0.0)) for s in STATES},
        survey_response_rate=float(responded),
        corruption=corruption.to_dict(),
        elapsed_seconds=time.perf_counter() - started,
    )

    write_json(summary.to_dict(), out / MANIFEST_FILE)
    return summary


def load_raw(cfg: Config, corrupted: bool = True) -> pd.DataFrame:
    """Load the generated activity log, parsing timestamps."""
    name = RAW_LOG_FILE if corrupted else CLEAN_LOG_FILE
    path = cfg.path("raw") / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `focustrack generate` first."
        )
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_users(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.path("raw") / USERS_FILE)


def load_survey(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.path("raw") / SURVEY_FILE, parse_dates=["date"])
