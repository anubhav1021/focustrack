"""The Focus Score.

A single 0-100 number for a day, combining three things a productive day needs
and that a raw hours-logged figure cannot distinguish between:

.. math::

    FS = 100 \\times (0.55 R + 0.25 (1 - S) + 0.20 B)

``R`` - **focus ratio**. Focused minutes as a share of minutes at the desk.
Breaks are excluded from the denominator: resting is not a failure to focus,
so a well-rested day is not penalised for it.

``S`` - **context-switch load**, in [0, 1]. Window switches per hour divided by
the cap at which switching is considered saturated. It enters as ``1 - S``, so
fragmented attention costs points even when the fragments are all work.

``B`` - **break balance**, in [0, 1]. Full marks when the longest stretch
without a break is 90 minutes or less, tapering to zero at 180. This is the
term that separates FocusTrack from a stopwatch: a day of unbroken focus is
not a perfect day.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import BREAK, FOCUSED


@dataclass
class FocusScore:
    """A Focus Score and the three components it is built from."""

    score: float
    focus_ratio: float          # R
    switch_load: float          # S
    break_balance: float        # B
    focused_minutes: int
    active_minutes: int
    total_minutes: int
    switches_per_hour: float
    longest_stretch_minutes: int
    n_breaks: int

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("score", "focus_ratio", "switch_load", "break_balance",
                    "switches_per_hour"):
            out[key] = round(float(out[key]), 4)
        return out

    def explain(self) -> str:
        """A plain-language breakdown, shown on the dashboard."""
        return (
            f"Focus Score {self.score:.0f}/100\n"
            f"  focus ratio      R = {self.focus_ratio:.2f}  "
            f"({self.focused_minutes} of {self.active_minutes} minutes at the desk)\n"
            f"  switch load      S = {self.switch_load:.2f}  "
            f"({self.switches_per_hour:.0f} window switches per hour)\n"
            f"  break balance    B = {self.break_balance:.2f}  "
            f"(longest stretch without a break: {self.longest_stretch_minutes} min)"
        )


def compute_focus_score(
    states: Sequence[str],
    switches: float,
    minutes_per_state: float,
    is_break: Sequence[bool] | None,
    cfg: Config,
) -> FocusScore:
    """Compute the Focus Score from a day's sequence of states.

    ``states`` is one entry per unit of time, ``minutes_per_state`` says how
    long each unit is (1 for minutes, 5 for windows), and ``is_break`` marks
    detected breaks. When ``is_break`` is omitted the ``Break`` state is used.
    """
    fs_cfg = cfg.section("focus_score")
    states = np.asarray(states, dtype=object)
    n = len(states)
    if n == 0:
        return FocusScore(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0, 0)

    breaks = (
        np.asarray(is_break, dtype=bool)
        if is_break is not None
        else (states == BREAK)
    )

    total_minutes = int(round(n * minutes_per_state))
    focused_units = int(np.sum(states == FOCUSED))
    active_units = int(np.sum(~breaks))
    focused_minutes = int(round(focused_units * minutes_per_state))
    active_minutes = int(round(active_units * minutes_per_state))

    # --- R: focus ratio ---------------------------------------------------
    focus_ratio = focused_units / active_units if active_units else 0.0

    # --- S: context-switch load -------------------------------------------
    active_hours = active_minutes / 60.0
    switches_per_hour = float(switches) / active_hours if active_hours > 0 else 0.0
    cap = float(fs_cfg["switch_cap_per_hour"])
    switch_load = float(np.clip(switches_per_hour / cap, 0.0, 1.0))

    # --- B: break balance --------------------------------------------------
    longest = longest_run_without_break(breaks, minutes_per_state)
    healthy = float(fs_cfg["healthy_stretch_minutes"])
    zero_at = float(fs_cfg["stretch_zero_minutes"])
    if longest <= healthy:
        break_balance = 1.0
    elif longest >= zero_at:
        break_balance = 0.0
    else:
        break_balance = float(1.0 - (longest - healthy) / (zero_at - healthy))

    score = 100.0 * (
        float(fs_cfg["w_focus_ratio"]) * focus_ratio
        + float(fs_cfg["w_switch_load"]) * (1.0 - switch_load)
        + float(fs_cfg["w_break_balance"]) * break_balance
    )

    return FocusScore(
        score=float(np.clip(score, 0.0, 100.0)),
        focus_ratio=focus_ratio,
        switch_load=switch_load,
        break_balance=break_balance,
        focused_minutes=focused_minutes,
        active_minutes=active_minutes,
        total_minutes=total_minutes,
        switches_per_hour=switches_per_hour,
        longest_stretch_minutes=longest,
        n_breaks=count_break_episodes(breaks),
    )


def longest_run_without_break(
    is_break: Sequence[bool], minutes_per_unit: float = 1.0
) -> int:
    """Longest unbroken stretch at the desk, in minutes."""
    longest = run = 0
    for flag in np.asarray(is_break, dtype=bool):
        run = 0 if flag else run + 1
        if run > longest:
            longest = run
    return int(round(longest * minutes_per_unit))


def count_break_episodes(is_break: Sequence[bool]) -> int:
    """Number of distinct break episodes, not break minutes."""
    flags = np.asarray(is_break, dtype=bool)
    if flags.size == 0:
        return 0
    starts = flags & ~np.concatenate([[False], flags[:-1]])
    return int(starts.sum())


def score_day(
    day_windows: pd.DataFrame,
    cfg: Config,
    state_column: str = "state_label",
) -> FocusScore:
    """Focus Score for a single participant-day of five-minute windows."""
    window_minutes = float(cfg["preprocessing.window_minutes"])
    states = day_windows[state_column].to_numpy()
    switches = float(day_windows["window_switches_sum"].sum())
    breaks = (
        day_windows["break_minutes"].to_numpy() > 0
        if "break_minutes" in day_windows.columns
        else None
    )
    return compute_focus_score(states, switches, window_minutes, breaks, cfg)


def score_all_days(
    windows: pd.DataFrame,
    cfg: Config,
    state_column: str = "state_label",
) -> pd.DataFrame:
    """Focus Score for every participant-day in a window frame."""
    rows: list[dict[str, Any]] = []
    for (user, date), day in windows.groupby(["user_id", "date"], sort=True, observed=True):
        day = day.sort_values("window_start")
        score = score_day(day, cfg, state_column=state_column)
        rows.append({"user_id": user, "date": date, **score.to_dict()})
    return pd.DataFrame(rows)
