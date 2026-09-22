"""The adaptive reminder engine.

A fixed timer interrupts you every 25 or 90 minutes whether or not you are
mid-thought. This engine only interrupts when the evidence says the
interruption will help, and it is built to be *quiet*: the constraints below
exist to protect the user from the system.

**When it suggests a break**

* the user is focused now, the predicted probability of losing focus in the
  next 15 minutes is at least 0.45, and at least 20 minutes have passed since
  the last break - an interruption worth its cost; or
* 100 minutes have passed since the last break, regardless of the model. No
  prediction is confident enough to justify letting someone work indefinitely.

**When it nudges**

* after 10 continuous minutes of distraction.

**When it stays silent**

* during meetings and breaks - never interrupt those;
* within 15 minutes of the previous notification;
* after 6 break reminders or 4 nudges in a day.

The silence rules are checked last and override everything, so a firing
condition is a *candidate*, never a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import (
    BREAK_REMINDER,
    DISTRACTED,
    FOCUS_NUDGE,
    FOCUSED,
)


@dataclass
class Notification:
    """One reminder the engine decided to send."""

    user_id: str
    timestamp: pd.Timestamp
    kind: str                      # break_reminder | focus_nudge
    reason: str
    message: str
    minutes_since_break: float
    drop_probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["timestamp"] = pd.Timestamp(self.timestamp).isoformat()
        return out


@dataclass
class EngineSettings:
    """The engine's thresholds, read from ``config.yaml``."""

    drop_threshold: float = 0.45
    min_minutes_since_break: int = 20
    hard_break_minutes: int = 100
    nudge_after_distracted_minutes: int = 10
    min_gap_minutes: int = 15
    max_breaks_per_day: int = 6
    max_nudges_per_day: int = 4
    protected_states: tuple[str, ...] = ("Meeting", "Break")

    @classmethod
    def from_config(
        cls, cfg: Config, drop_threshold: float | None = None
    ) -> "EngineSettings":
        """Build the settings, optionally overriding the break threshold.

        ``drop_threshold`` is passed when the threshold has been calibrated
        against the model's own predicted distribution rather than taken as an
        absolute probability.
        """
        r = cfg.section("reminders")
        return cls(
            drop_threshold=float(
                r["drop_probability_threshold"] if drop_threshold is None
                else drop_threshold
            ),
            min_minutes_since_break=int(r["min_minutes_since_break"]),
            hard_break_minutes=int(r["hard_break_minutes"]),
            nudge_after_distracted_minutes=int(r["distraction_nudge_minutes"]),
            min_gap_minutes=int(r["min_gap_between_notifications"]),
            max_breaks_per_day=int(r["max_breaks_per_day"]),
            max_nudges_per_day=int(r["max_nudges_per_day"]),
            protected_states=tuple(r["protected_states"]),
        )


@dataclass
class DayState:
    """The engine's memory within one workday."""

    breaks_sent: int = 0
    nudges_sent: int = 0
    last_notification: pd.Timestamp | None = None
    distracted_run_minutes: int = 0
    last_nudge_run: int = 0
    #: ``minutes_since_break`` when the 100-minute rule last fired, so an
    #: ignored reminder is not simply repeated 15 minutes later.
    last_hard_reminder_at: float | None = None
    previous_minutes_since_break: float = 0.0
    suppressed: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.breaks_sent = 0
        self.nudges_sent = 0
        self.last_notification = None
        self.distracted_run_minutes = 0
        self.last_nudge_run = 0
        self.last_hard_reminder_at = None
        self.previous_minutes_since_break = 0.0
        self.suppressed.clear()


class ReminderEngine:
    """Decides, window by window, whether to say anything at all."""

    def __init__(self, settings: EngineSettings, window_minutes: int = 5) -> None:
        self.settings = settings
        self.window_minutes = window_minutes
        self.state = DayState()

    # -- the decision ------------------------------------------------------
    def step(
        self,
        user_id: str,
        timestamp: pd.Timestamp,
        state: str,
        minutes_since_break: float,
        drop_probability: float | None,
    ) -> Notification | None:
        """Advance one window and return a notification if one is due."""
        s = self.settings
        self._track_distraction(state)
        self._track_break_clock(minutes_since_break)

        # --- silence rule 1: never interrupt a meeting or a break ---------
        if state in s.protected_states:
            return None

        candidate = self._candidate(state, minutes_since_break, drop_probability)
        if candidate is None:
            return None
        kind, reason = candidate

        # --- silence rule 2: daily caps -----------------------------------
        if kind == BREAK_REMINDER and self.state.breaks_sent >= s.max_breaks_per_day:
            self.state.suppressed.append("break cap reached")
            return None
        if kind == FOCUS_NUDGE and self.state.nudges_sent >= s.max_nudges_per_day:
            self.state.suppressed.append("nudge cap reached")
            return None

        # --- silence rule 3: spacing --------------------------------------
        last = self.state.last_notification
        if last is not None:
            gap = (timestamp - last).total_seconds() / 60.0
            if gap < s.min_gap_minutes:
                self.state.suppressed.append(f"only {gap:.0f} min since last notification")
                return None

        self.state.last_notification = timestamp
        if kind == BREAK_REMINDER:
            self.state.breaks_sent += 1
            if reason.endswith("without a break"):
                self.state.last_hard_reminder_at = minutes_since_break
        else:
            self.state.nudges_sent += 1
            self.state.last_nudge_run = self.state.distracted_run_minutes

        return Notification(
            user_id=user_id,
            timestamp=timestamp,
            kind=kind,
            reason=reason,
            message=self._message(kind, reason, minutes_since_break),
            minutes_since_break=float(minutes_since_break),
            drop_probability=(
                None if drop_probability is None else float(drop_probability)
            ),
        )

    # -- helpers -----------------------------------------------------------
    def _track_distraction(self, state: str) -> None:
        if state == DISTRACTED:
            self.state.distracted_run_minutes += self.window_minutes
        else:
            self.state.distracted_run_minutes = 0
            self.state.last_nudge_run = 0

    def _track_break_clock(self, minutes_since_break: float) -> None:
        """Clear the hard-reminder memory once the user actually takes a break."""
        if minutes_since_break < self.state.previous_minutes_since_break:
            self.state.last_hard_reminder_at = None
        self.state.previous_minutes_since_break = minutes_since_break

    def _candidate(
        self,
        state: str,
        minutes_since_break: float,
        drop_probability: float | None,
    ) -> tuple[str, str] | None:
        """Which reminder, if any, the evidence currently supports."""
        s = self.settings

        # A long stretch at the desk outranks everything the model might say.
        # It fires once on crossing the limit, then backs off for another full
        # interval: a reminder repeated every 15 minutes until you comply is
        # the nagging this engine exists to avoid, and it trains people to
        # dismiss the notification rather than take the break.
        if minutes_since_break >= s.hard_break_minutes:
            last = self.state.last_hard_reminder_at
            if last is None or minutes_since_break - last >= s.hard_break_minutes:
                return BREAK_REMINDER, f"{minutes_since_break:.0f} min without a break"

        if (
            state == FOCUSED
            and drop_probability is not None
            and drop_probability >= s.drop_threshold
            and minutes_since_break >= s.min_minutes_since_break
        ):
            return (
                BREAK_REMINDER,
                f"focus likely to drop (p={drop_probability:.2f}) after "
                f"{minutes_since_break:.0f} min",
            )

        run = self.state.distracted_run_minutes
        if run >= s.nudge_after_distracted_minutes and run > self.state.last_nudge_run:
            return FOCUS_NUDGE, f"{run} min of distraction"

        return None

    @staticmethod
    def _message(kind: str, reason: str, minutes_since_break: float) -> str:
        if kind == BREAK_REMINDER:
            return (
                f"Time for a break - you have been at it for "
                f"{minutes_since_break:.0f} minutes. Five minutes away from the "
                f"screen now beats twenty minutes of drift later."
            )
        return "Still on track? You have drifted for a few minutes - worth a reset."


# ---------------------------------------------------------------------------
# running the engine over recorded days
# ---------------------------------------------------------------------------
def run_engine_on_day(
    day: pd.DataFrame,
    cfg: Config,
    state_column: str = "state_label",
    probability_column: str = "drop_probability",
    drop_threshold: float | None = None,
) -> list[Notification]:
    """Replay one participant-day through the engine."""
    settings = EngineSettings.from_config(cfg, drop_threshold=drop_threshold)
    engine = ReminderEngine(settings, int(cfg["preprocessing.window_minutes"]))

    notifications: list[Notification] = []
    ordered = day.sort_values("window_start")
    user_id = str(ordered["user_id"].iloc[0])

    for row in ordered.itertuples(index=False):
        probability = getattr(row, probability_column, None)
        if probability is not None and (
            isinstance(probability, float) and np.isnan(probability)
        ):
            probability = None
        note = engine.step(
            user_id=user_id,
            timestamp=pd.Timestamp(row.window_start),
            state=str(getattr(row, state_column)),
            minutes_since_break=float(row.minutes_since_break),
            drop_probability=probability,
        )
        if note is not None:
            notifications.append(note)
    return notifications


def run_engine(
    windows: pd.DataFrame,
    cfg: Config,
    state_column: str = "state_label",
    probability_column: str = "drop_probability",
    drop_threshold: float | None = None,
) -> pd.DataFrame:
    """Replay every participant-day and return the notifications fired."""
    rows: list[dict[str, Any]] = []
    for (_user, date), day in windows.groupby(["user_id", "date"], sort=True, observed=True):
        for note in run_engine_on_day(
            day, cfg, state_column, probability_column, drop_threshold
        ):
            payload = note.to_dict()
            payload["date"] = date
            rows.append(payload)
    if not rows:
        return pd.DataFrame(
            columns=["user_id", "date", "timestamp", "kind", "reason", "message",
                     "minutes_since_break", "drop_probability"]
        )
    return pd.DataFrame(rows)


def resolve_drop_threshold(cfg: Config, calibrated: float | None) -> float:
    """The break threshold the engine should use, per ``drop_threshold_mode``.

    ``absolute`` uses the published 0.45. ``percentile`` uses the value
    calibrated against the model's own training distribution, which is what
    keeps the adaptive path alive when the model is well calibrated on a rare
    event and simply never predicts 0.45.
    """
    mode = str(cfg.get("reminders.drop_threshold_mode", "absolute")).lower()
    absolute = float(cfg["reminders.drop_probability_threshold"])
    if mode == "percentile" and calibrated:
        return float(calibrated)
    return absolute


def notifications_per_day(notifications: pd.DataFrame, n_user_days: int) -> dict[str, float]:
    """Average break reminders and nudges per participant-day.

    The breakdown by trigger matters as much as the totals: a break reminder
    fired by the 100-minute rule is a fallback, not the adaptive behaviour the
    model is supposed to provide. If almost none come from the model, the
    probability threshold is mis-set for how this model is calibrated - and
    that is invisible in the headline count.
    """
    if n_user_days <= 0 or not len(notifications):
        return {
            "break_reminders_per_day": 0.0,
            "focus_nudges_per_day": 0.0,
            "breaks_from_model": 0,
            "breaks_from_time_limit": 0,
        }

    counts = notifications["kind"].value_counts()
    breaks = notifications.loc[notifications["kind"] == BREAK_REMINDER]
    from_limit = int(breaks["reason"].str.endswith("without a break").sum())

    return {
        "break_reminders_per_day": float(counts.get(BREAK_REMINDER, 0)) / n_user_days,
        "focus_nudges_per_day": float(counts.get(FOCUS_NUDGE, 0)) / n_user_days,
        "breaks_from_model": int(len(breaks) - from_limit),
        "breaks_from_time_limit": from_limit,
    }


# ---------------------------------------------------------------------------
# the control condition
# ---------------------------------------------------------------------------
def timer_baseline(
    windows: pd.DataFrame,
    cfg: Config,
    interval_minutes: int | None = None,
) -> pd.DataFrame:
    """A fixed timer: remind whenever ``interval_minutes`` have passed at the desk.

    This is what FocusTrack has to beat. It fires on the clock alone, with no
    idea whether the user is focused, in a meeting, or already drifting.
    """
    interval = int(
        interval_minutes
        if interval_minutes is not None
        else cfg["reminders.timer_baseline_minutes"]
    )
    rows: list[dict[str, Any]] = []
    for (user, date), day in windows.groupby(["user_id", "date"], sort=True, observed=True):
        ordered = day.sort_values("window_start")
        fired_at = -np.inf
        for row in ordered.itertuples(index=False):
            since = float(row.minutes_since_break)
            if since >= interval and since - fired_at >= interval:
                rows.append(
                    {
                        "user_id": user,
                        "date": date,
                        "timestamp": pd.Timestamp(row.window_start),
                        "kind": BREAK_REMINDER,
                        "reason": f"fixed {interval}-minute timer",
                        "minutes_since_break": since,
                    }
                )
                fired_at = since
    return pd.DataFrame(rows)
