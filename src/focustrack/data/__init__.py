"""Dataset layer: personas, the behaviour model and the raw-log generator.

Public datasets of remote work are one-row-per-employee surveys with neither
minute-level activity nor focus labels, so the collection schema is defined
here and a pilot dataset is generated from a documented behaviour model. Real
volunteer data is collected against the same schema (see
``focustrack.agent``), which is why every downstream stage reads the schema
in :mod:`focustrack.constants` rather than anything simulation-specific.
"""

from focustrack.data.personas import build_users, UserPersona
from focustrack.data.simulate import simulate_cohort, simulate_user_day
from focustrack.data.corrupt import corrupt_logs, CorruptionReport
from focustrack.data.generate import generate_dataset

__all__ = [
    "UserPersona",
    "build_users",
    "simulate_user_day",
    "simulate_cohort",
    "corrupt_logs",
    "CorruptionReport",
    "generate_dataset",
]
