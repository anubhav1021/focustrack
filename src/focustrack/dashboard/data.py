"""Loading and scoring data for the dashboard.

Kept separate from the page itself so the loading logic is testable without
Streamlit, and so the caching decorators sit in one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from focustrack.config import Config, load_config
from focustrack.constants import CATEGORIES


@dataclass
class DashboardData:
    """Everything the dashboard needs, already scored."""

    windows: pd.DataFrame
    daily: pd.DataFrame
    users: pd.DataFrame
    notifications: pd.DataFrame
    survey: pd.DataFrame
    results: dict[str, Any]
    state_column: str

    @property
    def user_ids(self) -> list[str]:
        return sorted(self.windows["user_id"].unique().tolist())

    def for_user(self, user_id: str) -> "DashboardData":
        return DashboardData(
            windows=self.windows.loc[self.windows["user_id"] == user_id],
            daily=self.daily.loc[self.daily["user_id"] == user_id],
            users=self.users.loc[self.users.get("user_id", pd.Series(dtype=str)) == user_id],
            notifications=(
                self.notifications.loc[self.notifications["user_id"] == user_id]
                if len(self.notifications) else self.notifications
            ),
            survey=(
                self.survey.loc[self.survey["user_id"] == user_id]
                if len(self.survey) else self.survey
            ),
            results=self.results,
            state_column=self.state_column,
        )


def load_dashboard_data(cfg: Config | None = None) -> DashboardData:
    """Load the processed windows, score them and build the daily summary."""
    from focustrack.engine.analytics import daily_summary
    from focustrack.models.registry import load_bundle, models_available
    from focustrack.preprocessing.features import feature_matrix
    from focustrack.preprocessing.pipeline import load_windows

    cfg = cfg or load_config()
    windows = load_windows(cfg)
    windows["date"] = pd.to_datetime(windows["date"])
    windows["window_start"] = pd.to_datetime(windows["window_start"])

    state_column = "state_label"
    if models_available(cfg):
        try:
            bundle = load_bundle(cfg)
            features = feature_matrix(windows)
            windows["predicted_state"] = bundle.focus_state.predict(features)
            windows["drop_probability"] = bundle.focus_drop.predict_proba(features)[:, 1]
            state_column = "predicted_state"
        except Exception:
            # A stale or mismatched bundle should not stop the dashboard from
            # showing the recorded data.
            pass

    daily = daily_summary(windows, cfg, state_column=state_column)
    if state_column != "state_label":
        truth = daily_summary(windows, cfg, state_column="state_label")
        daily = daily.merge(
            truth[["user_id", "date", "focus_score"]],
            on=["user_id", "date"], how="left", suffixes=("", "_actual"),
        )

    return DashboardData(
        windows=windows,
        daily=daily,
        users=_load_optional_csv(cfg.path("raw") / "users.csv"),
        notifications=_load_notifications(cfg),
        survey=_load_survey(cfg),
        results=_load_results(cfg),
        state_column=state_column,
    )


def _load_optional_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _load_notifications(cfg: Config) -> pd.DataFrame:
    frame = _load_optional_csv(cfg.path("reports") / "notifications.csv")
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _load_survey(cfg: Config) -> pd.DataFrame:
    frame = _load_optional_csv(cfg.path("raw") / "daily_survey.csv")
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _load_results(cfg: Config) -> dict[str, Any]:
    path = cfg.path("reports") / "results.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
def state_minutes(day: pd.DataFrame, state_column: str, window_minutes: int = 5) -> pd.DataFrame:
    """Minutes per state for one day."""
    counts = day[state_column].value_counts()
    return pd.DataFrame(
        {"state": counts.index, "minutes": counts.to_numpy() * window_minutes}
    )


def category_minutes(day: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """Minutes per application category for one day."""
    rows = []
    for category in CATEGORIES:
        column = f"app_frac_{category}"
        if column in day.columns:
            rows.append(
                {
                    "category": category.replace("_", " / "),
                    "minutes": int(round(day[column].sum() * window_minutes)),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.loc[frame["minutes"] > 0] if not frame.empty else frame


def live_store_data(cfg: Config) -> pd.DataFrame:
    """Activity recorded by the live agent, if the store has any."""
    from focustrack.storage.db import FocusStore

    try:
        store = FocusStore.from_config(cfg)
        return store.load_activity()
    except Exception:
        return pd.DataFrame()
