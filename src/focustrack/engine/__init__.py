"""Layer 4/5 - reminders, the Focus Score and daily analytics.

This is where the models stop being predictions and start being decisions:
what to say to the user, when to stay quiet, and what the day is worth.
"""

from focustrack.engine.analytics import (
    Correlation,
    cohort_averages,
    daily_summary,
    hourly_profile,
    per_user_summary,
    score_agreement,
    survey_correlations,
)
from focustrack.engine.focus_score import (
    FocusScore,
    compute_focus_score,
    count_break_episodes,
    longest_run_without_break,
    score_all_days,
    score_day,
)
from focustrack.engine.reminders import (
    EngineSettings,
    Notification,
    ReminderEngine,
    notifications_per_day,
    run_engine,
    run_engine_on_day,
    timer_baseline,
)

__all__ = [
    "Correlation",
    "EngineSettings",
    "FocusScore",
    "Notification",
    "ReminderEngine",
    "cohort_averages",
    "compute_focus_score",
    "count_break_episodes",
    "daily_summary",
    "hourly_profile",
    "longest_run_without_break",
    "notifications_per_day",
    "per_user_summary",
    "run_engine",
    "run_engine_on_day",
    "score_agreement",
    "score_all_days",
    "score_day",
    "survey_correlations",
    "timer_baseline",
]
