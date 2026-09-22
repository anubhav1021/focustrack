"""Shared fixtures.

The fixtures build a small but *complete* cohort - every role, several days,
all four states - rather than a handful of hand-written rows. Tests of a
pipeline are only worth much if the data going in has the same shape as the
real thing, including the defects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from focustrack.config import Config, load_config
from focustrack.data.corrupt import corrupt_logs
from focustrack.data.personas import build_users
from focustrack.data.simulate import simulate_cohort
from focustrack.preprocessing.pipeline import run_pipeline


@pytest.fixture(scope="session")
def base_config() -> Config:
    """The project configuration, unmodified."""
    return load_config()


@pytest.fixture(scope="session")
def small_config(base_config: Config) -> Config:
    """A fifteen-participant, four-day cohort - fast, but complete.

    Three per role is the smallest cohort that still leaves two participants
    per role for training once one per role is held out.
    """
    return base_config.with_overrides(
        **{
            "dataset.role_counts": {
                "developer": 3, "designer": 3, "writer": 3, "analyst": 3, "support": 3
            },
            "dataset.n_workdays": 4,
            "split.n_test_users_per_role": 1,
            "split.cv_folds": 2,
        }
    )


@pytest.fixture(scope="session")
def personas(small_config: Config):
    return build_users(small_config, np.random.default_rng(small_config.seed))


@pytest.fixture(scope="session")
def clean_logs(small_config: Config, personas) -> pd.DataFrame:
    logs, _survey = simulate_cohort(
        personas, small_config, np.random.default_rng(small_config.seed)
    )
    return logs


@pytest.fixture(scope="session")
def survey(small_config: Config, personas) -> pd.DataFrame:
    _logs, survey = simulate_cohort(
        personas, small_config, np.random.default_rng(small_config.seed)
    )
    return survey


@pytest.fixture(scope="session")
def raw_logs(small_config: Config, clean_logs: pd.DataFrame) -> pd.DataFrame:
    """The clean log with realistic defects injected."""
    corrupted, _report = corrupt_logs(
        clean_logs, small_config, np.random.default_rng(small_config.seed + 7)
    )
    return corrupted


@pytest.fixture(scope="session")
def pipeline_result(small_config: Config, raw_logs: pd.DataFrame):
    return run_pipeline(raw_logs, small_config)


@pytest.fixture(scope="session")
def windows(pipeline_result) -> pd.DataFrame:
    return pipeline_result.windows


@pytest.fixture(scope="session")
def minutes(pipeline_result) -> pd.DataFrame:
    return pipeline_result.minutes
