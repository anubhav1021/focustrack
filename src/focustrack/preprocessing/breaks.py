"""Step 8 - break detection, and the minutes-since-break clock.

A minute counts as idle when its ``idle_seconds`` reading crosses the
threshold. A **break** is three or more consecutive idle minutes; a single
quiet minute while reading is not a break, which is exactly the distinction
that keeps reading spells from being mistaken for time away from the desk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.preprocessing.steps import Audit

#: A data gap at least this long is treated as time away from the desk, so the
#: minutes-since-break clock restarts. It is not counted as a detected break -
#: the agent was offline, and absence of evidence is not evidence of a break.
DEFAULT_GAP_RESET_MINUTES = 10


def detect_breaks(df: pd.DataFrame, cfg: Config, audit: Audit | None = None) -> pd.DataFrame:
    """Add ``is_idle``, ``is_break``, ``break_id`` and ``minutes_since_break``."""
    rows_in = len(df)
    threshold = float(cfg["preprocessing.idle_minute_threshold_seconds"])
    min_break = int(cfg["preprocessing.min_break_minutes"])
    gap_reset = int(cfg.get("preprocessing.gap_reset_break_minutes", DEFAULT_GAP_RESET_MINUTES))

    out = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    if "date" not in out.columns:
        out["date"] = out["timestamp"].dt.normalize()
    if "is_gap" not in out.columns:
        out["is_gap"] = False

    # A minute with no reading at all cannot be called idle or active.
    idle = out["idle_seconds"] >= threshold
    out["is_idle"] = idle.fillna(False) & ~out["is_gap"]

    is_break = np.zeros(len(out), dtype=bool)
    break_id = np.full(len(out), -1, dtype=np.int64)
    since_break = np.zeros(len(out), dtype=np.int32)

    next_break_id = 0
    n_breaks = 0
    for _, index in out.groupby(["user_id", "date"], sort=False, observed=True).indices.items():
        idx = np.sort(index)
        day_idle = out["is_idle"].to_numpy()[idx]
        day_gap = out["is_gap"].to_numpy()[idx]

        # --- find runs of consecutive idle minutes ------------------------
        run_start = 0
        while run_start < len(idx):
            if not day_idle[run_start]:
                run_start += 1
                continue
            run_end = run_start
            while run_end + 1 < len(idx) and day_idle[run_end + 1]:
                run_end += 1
            if run_end - run_start + 1 >= min_break:
                is_break[idx[run_start:run_end + 1]] = True
                break_id[idx[run_start:run_end + 1]] = next_break_id
                next_break_id += 1
                n_breaks += 1
            run_start = run_end + 1

        # --- the minutes-since-break clock --------------------------------
        counter = 0
        gap_run = 0
        day_break = is_break[idx]
        for j in range(len(idx)):
            if day_gap[j]:
                gap_run += 1
                since_break[idx[j]] = counter
                continue
            if gap_run:
                # The agent was offline; a long absence resets the clock.
                counter = 0 if gap_run >= gap_reset else counter + gap_run
                gap_run = 0
            if day_break[j]:
                counter = 0
            else:
                counter += 1
            since_break[idx[j]] = counter

    out["is_break"] = is_break
    out["break_id"] = break_id
    out["minutes_since_break"] = since_break

    if audit is not None:
        audit.add(
            8, "detect breaks (3+ idle minutes)", rows_in, len(out), int(is_break.sum()),
            idle_minutes=int(out["is_idle"].sum()),
            break_minutes=int(is_break.sum()),
            break_episodes=n_breaks,
            mean_break_minutes=round(float(is_break.sum() / max(n_breaks, 1)), 2),
        )
    return out


def break_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each detected break as one row."""
    taken = df.loc[df["break_id"] >= 0]
    if taken.empty:
        return pd.DataFrame(
            columns=["user_id", "date", "break_id", "start", "end", "minutes"]
        )
    return (
        taken.groupby("break_id", observed=True)
        .agg(
            user_id=("user_id", "first"),
            date=("date", "first"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            minutes=("timestamp", "size"),
        )
        .reset_index()
        .loc[:, ["user_id", "date", "break_id", "start", "end", "minutes"]]
    )


def longest_stretch_without_break(is_break: np.ndarray) -> int:
    """Longest run of consecutive non-break minutes, in minutes."""
    longest = run = 0
    for b in np.asarray(is_break, dtype=bool):
        run = 0 if b else run + 1
        if run > longest:
            longest = run
    return int(longest)
