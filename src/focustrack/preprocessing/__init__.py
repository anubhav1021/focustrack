"""Layer 3 - the processing engine.

Eleven steps turn a raw agent log into the five-minute window table the models
are trained and served on. See :mod:`focustrack.preprocessing.pipeline` for
the running order.
"""

from focustrack.preprocessing.breaks import (
    break_episodes,
    detect_breaks,
    longest_stretch_without_break,
)
from focustrack.preprocessing.features import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    N_FEATURES,
    describe_features,
    feature_matrix,
)
from focustrack.preprocessing.normalise import UserScaler, normalise_per_user
from focustrack.preprocessing.pipeline import (
    PipelineResult,
    load_minutes,
    load_scaler,
    load_windows,
    run_pipeline,
    save_result,
)
from focustrack.preprocessing.split import UserSplit, load_split, make_split, save_split
from focustrack.preprocessing.steps import Audit, AuditEntry

__all__ = [
    "Audit",
    "AuditEntry",
    "FEATURE_COLUMNS",
    "FEATURE_GROUPS",
    "N_FEATURES",
    "PipelineResult",
    "UserScaler",
    "UserSplit",
    "break_episodes",
    "describe_features",
    "detect_breaks",
    "feature_matrix",
    "load_minutes",
    "load_scaler",
    "load_split",
    "load_windows",
    "longest_stretch_without_break",
    "make_split",
    "normalise_per_user",
    "run_pipeline",
    "save_result",
    "save_split",
]
