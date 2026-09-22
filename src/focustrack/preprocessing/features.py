"""Step 11 - the 31-feature description of a five-minute window.

The feature set is deliberately small and readable, because the model's job is
to be trusted by the person it is watching. The 31 features fall into five
groups:

==========================  ==  =========================================
group                        n  what it captures
==========================  ==  =========================================
activity rates               7  how hard the keyboard and mouse are working
application-usage mix        8  which of the six surfaces the time went to
idle and break signals       5  how quiet the window was, and time at desk
time of day                  5  the hour, day progress, afternoon slump
previous-15-minute history   6  where the user was coming from
==========================  ==  =========================================

Every activity rate is the **per-user normalised** count, not the raw one.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from focustrack.constants import CATEGORIES

# --- the feature groups -----------------------------------------------------
ACTIVITY_FEATURES: Final[tuple[str, ...]] = (
    "keystrokes_mean",
    "keystrokes_max",
    "mouse_clicks_mean",
    "mouse_distance_mean",
    "scroll_events_mean",
    "window_switches_sum",
    "activity_index_std",
)

APP_MIX_FEATURES: Final[tuple[str, ...]] = (
    *(f"app_frac_{c}" for c in CATEGORIES),
    "n_unique_apps",
    "app_switch_rate",
)

IDLE_BREAK_FEATURES: Final[tuple[str, ...]] = (
    "idle_mean",
    "idle_max",
    "idle_frac",
    "minutes_since_break",
    "break_minutes",
)

TIME_FEATURES: Final[tuple[str, ...]] = (
    "hour_sin",
    "hour_cos",
    "minutes_since_day_start",
    "day_progress",
    "is_afternoon_slump",
)

HISTORY_FEATURES: Final[tuple[str, ...]] = (
    "prev_keystrokes_mean",
    "prev_idle_mean",
    "prev_switch_sum",
    "hist15_keystrokes_mean",
    "hist15_idle_frac",
    "hist15_personal_frac",
)

#: The 31 model inputs, in a stable order.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *ACTIVITY_FEATURES,
    *APP_MIX_FEATURES,
    *IDLE_BREAK_FEATURES,
    *TIME_FEATURES,
    *HISTORY_FEATURES,
)

FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "activity rates": ACTIVITY_FEATURES,
    "application-usage mix": APP_MIX_FEATURES,
    "idle and break signals": IDLE_BREAK_FEATURES,
    "time of day": TIME_FEATURES,
    "previous 15 minutes": HISTORY_FEATURES,
}

N_FEATURES: Final[int] = len(FEATURE_COLUMNS)

#: Columns kept alongside the features for analytics, evaluation and the
#: dashboard. They are never shown to a model.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "date",
    "window_start",
    "n_minutes",
    "n_labelled",
    "n_distinct_states",
    "label_purity",
    "raw_keystrokes",
    "raw_clicks",
    "state_label",
)

#: Human-readable names for figures and the dashboard.
FEATURE_LABELS: Final[dict[str, str]] = {
    "keystrokes_mean": "typing rate (user-normalised)",
    "keystrokes_max": "peak typing rate",
    "mouse_clicks_mean": "click rate",
    "mouse_distance_mean": "pointer distance",
    "scroll_events_mean": "scroll rate",
    "window_switches_sum": "window switches",
    "activity_index_std": "activity variability",
    "app_frac_development": "time in development apps",
    "app_frac_creative": "time in creative apps",
    "app_frac_docs_analysis": "time in docs / analysis",
    "app_frac_research": "time in research / browsing",
    "app_frac_communication": "time in communication apps",
    "app_frac_personal": "time in personal apps",
    "n_unique_apps": "distinct apps used",
    "app_switch_rate": "app switch rate",
    "idle_mean": "mean idle seconds",
    "idle_max": "peak idle seconds",
    "idle_frac": "share of idle minutes",
    "minutes_since_break": "minutes since last break",
    "break_minutes": "break minutes in window",
    "hour_sin": "hour of day (sin)",
    "hour_cos": "hour of day (cos)",
    "minutes_since_day_start": "minutes into the workday",
    "day_progress": "progress through the day",
    "is_afternoon_slump": "in the 14:00-16:00 slump",
    "prev_keystrokes_mean": "typing rate, 5 min ago",
    "prev_idle_mean": "idle seconds, 5 min ago",
    "prev_switch_sum": "window switches, 5 min ago",
    "hist15_keystrokes_mean": "typing rate, previous 15 min",
    "hist15_idle_frac": "idle share, previous 15 min",
    "hist15_personal_frac": "personal-app share, previous 15 min",
}


def label(feature: str) -> str:
    """Human-readable name for a feature."""
    return FEATURE_LABELS.get(feature, feature)


def feature_matrix(windows: pd.DataFrame) -> pd.DataFrame:
    """Return the 31 feature columns, with any residual gaps filled.

    Missing values only survive to here when a whole window had no usable
    reading for one signal; a zero is the honest neutral value for a rate.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in windows.columns]
    if missing:
        raise KeyError(f"windows frame is missing features: {missing}")
    matrix = windows.loc[:, list(FEATURE_COLUMNS)].astype(float)
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def describe_features() -> pd.DataFrame:
    """A tidy table of the feature set, for the report and the dashboard."""
    rows = [
        {"group": group, "feature": f, "description": label(f)}
        for group, features in FEATURE_GROUPS.items()
        for f in features
    ]
    return pd.DataFrame(rows)
