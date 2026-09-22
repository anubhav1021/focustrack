"""Step 10 - group minutes into five-minute windows.

Five minutes is the cadence the processing engine runs at, so it is also the
unit the classifier predicts on. A window takes the majority state of its
minutes as its label. Windows with too few observed minutes are dropped rather
than padded.

``n_distinct_states`` is carried through because it separates the windows the
model finds easy (one state throughout) from the transition windows where
almost all of its errors live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import BREAK, CATEGORIES, DISTRACTED, FOCUSED, MEETING
from focustrack.preprocessing.steps import Audit

#: Columns describing a window, before feature engineering.
WINDOW_KEYS = ("user_id", "date", "window_start")


def build_windows(df: pd.DataFrame, cfg: Config, audit: Audit | None = None) -> pd.DataFrame:
    """Aggregate the minute-level frame into five-minute windows."""
    rows_in = len(df)
    minutes = int(cfg["preprocessing.window_minutes"])
    min_minutes = int(cfg["preprocessing.min_minutes_per_window"])

    work = df.loc[~df.get("is_gap", pd.Series(False, index=df.index))].copy()
    work["window_start"] = work["timestamp"].dt.floor(f"{minutes}min")

    # One-hot the six categories so a groupby mean gives the usage mix.
    for category in CATEGORIES:
        work[f"cat_{category}"] = (work["app_category"] == category).astype(float)

    # An app switch is a change of application between consecutive minutes.
    work["app_changed"] = (
        work.groupby(["user_id", "date"], observed=True)["active_app"].shift() != work["active_app"]
    ).astype(float)

    idle_threshold = float(cfg["preprocessing.idle_minute_threshold_seconds"])
    work["idle_flag"] = (work["idle_seconds"] >= idle_threshold).astype(float)
    # A single composite activity index, used for within-window variability.
    work["activity_index"] = (
        work["keystrokes_n"].fillna(0.0)
        + work["mouse_clicks_n"].fillna(0.0)
        + work["scroll_events_n"].fillna(0.0)
    )

    grouped = work.groupby(list(WINDOW_KEYS), sort=True, observed=True)

    agg = grouped.agg(
        n_minutes=("timestamp", "size"),
        n_labelled=("has_label", "sum"),
        keystrokes_mean=("keystrokes_n", "mean"),
        keystrokes_max=("keystrokes_n", "max"),
        mouse_clicks_mean=("mouse_clicks_n", "mean"),
        mouse_distance_mean=("mouse_distance_px_n", "mean"),
        scroll_events_mean=("scroll_events_n", "mean"),
        window_switches_sum=("window_switches", "sum"),
        activity_index_std=("activity_index", "std"),
        idle_mean=("idle_seconds", "mean"),
        idle_max=("idle_seconds", "max"),
        idle_frac=("idle_flag", "mean"),
        minutes_since_break=("minutes_since_break", "last"),
        break_minutes=("is_break", "sum"),
        n_unique_apps=("active_app", "nunique"),
        app_switch_rate=("app_changed", "mean"),
        raw_keystrokes=("keystrokes", "sum"),
        raw_clicks=("mouse_clicks", "sum"),
        **{f"app_frac_{c}": (f"cat_{c}", "mean") for c in CATEGORIES},
    )

    # --- label: the majority state of the window --------------------------
    labels = _majority_label(work)
    agg = agg.join(labels)
    agg = agg.reset_index()

    before_drop = len(agg)
    agg = agg.loc[agg["n_minutes"] >= min_minutes].reset_index(drop=True)
    dropped = before_drop - len(agg)

    agg["activity_index_std"] = agg["activity_index_std"].fillna(0.0)
    agg["break_minutes"] = agg["break_minutes"].astype(float)
    agg["window_switches_sum"] = agg["window_switches_sum"].astype(float)

    if audit is not None:
        audit.add(
            10, f"group into {minutes}-minute windows", rows_in, len(agg), dropped,
            windows_before_drop=before_drop,
            sparse_windows_dropped=dropped,
            labelled_windows=int(agg["state_label"].notna().sum()),
        )
    return agg


def _majority_label(work: pd.DataFrame) -> pd.DataFrame:
    """Majority state per window, plus how many distinct states it contained."""
    labelled = work.loc[work["has_label"]]
    if labelled.empty:
        idx = pd.MultiIndex.from_arrays(
            [[], [], []], names=list(WINDOW_KEYS)
        )
        return pd.DataFrame(
            {"state_label": [], "n_distinct_states": [], "label_purity": []}, index=idx
        )

    counts = (
        labelled.groupby([*WINDOW_KEYS, "state_label"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    counts = counts.sort_values(
        [*WINDOW_KEYS, "n", "state_label"], ascending=[True, True, True, False, True]
    )
    top = counts.groupby(list(WINDOW_KEYS), observed=True).tail(1).set_index(list(WINDOW_KEYS))
    totals = counts.groupby(list(WINDOW_KEYS), observed=True)["n"].sum()
    distinct = counts.groupby(list(WINDOW_KEYS), observed=True)["state_label"].nunique()

    return pd.DataFrame(
        {
            "state_label": top["state_label"],
            "n_distinct_states": distinct,
            "label_purity": top["n"] / totals,
        }
    )


def add_history(windows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the previous-15-minute history features (three windows back)."""
    minutes = int(cfg["preprocessing.window_minutes"])
    lookback = max(1, 15 // minutes)          # 3 windows = 15 minutes

    out = windows.sort_values(["user_id", "window_start"], kind="stable").reset_index(drop=True)
    by_day = out.groupby(["user_id", "date"], observed=True)

    # Lag 1 - what the user was doing five minutes ago.
    out["prev_keystrokes_mean"] = by_day["keystrokes_mean"].shift(1)
    out["prev_idle_mean"] = by_day["idle_mean"].shift(1)
    out["prev_switch_sum"] = by_day["window_switches_sum"].shift(1)

    # Rolling mean over the previous three windows, excluding the current one.
    def _hist(col: str) -> pd.Series:
        shifted = by_day[col].shift(1)
        return (
            shifted.groupby([out["user_id"], out["date"]], observed=True)
            .rolling(lookback, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    out["hist15_keystrokes_mean"] = _hist("keystrokes_mean")
    out["hist15_idle_frac"] = _hist("idle_frac")
    out["hist15_personal_frac"] = _hist("app_frac_personal")

    # At the start of a day there is no history; fall back to the current
    # window rather than to a constant that would look like a real reading.
    out["prev_keystrokes_mean"] = out["prev_keystrokes_mean"].fillna(out["keystrokes_mean"])
    out["prev_idle_mean"] = out["prev_idle_mean"].fillna(out["idle_mean"])
    out["prev_switch_sum"] = out["prev_switch_sum"].fillna(out["window_switches_sum"])
    out["hist15_keystrokes_mean"] = out["hist15_keystrokes_mean"].fillna(out["keystrokes_mean"])
    out["hist15_idle_frac"] = out["hist15_idle_frac"].fillna(out["idle_frac"])
    out["hist15_personal_frac"] = out["hist15_personal_frac"].fillna(out["app_frac_personal"])
    return out


def add_time_of_day(windows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the five time-of-day features, including the afternoon slump flag."""
    out = windows.copy()
    ts = out["window_start"]
    hour = ts.dt.hour + ts.dt.minute / 60.0

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    day_start = out.groupby(["user_id", "date"], observed=True)["window_start"].transform("min")
    day_end = out.groupby(["user_id", "date"], observed=True)["window_start"].transform("max")
    out["minutes_since_day_start"] = (ts - day_start).dt.total_seconds() / 60.0
    span = (day_end - day_start).dt.total_seconds() / 60.0
    out["day_progress"] = out["minutes_since_day_start"] / span.where(span > 0, 1.0)

    slump_lo, slump_hi = cfg["behaviour.slump_window"]
    out["is_afternoon_slump"] = ((hour >= slump_lo) & (hour < slump_hi)).astype(float)
    return out
