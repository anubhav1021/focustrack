"""Inject the problems a real collection agent produces.

The clean simulated log is too clean to be useful: a real agent misreports its
timezone, re-uploads the same batch twice, emits stuck-key counts, and goes
offline for minutes at a time. Every defect injected here has a matching
repair step in :mod:`focustrack.preprocessing.pipeline`, and the counts
reported by the two sides are what the preprocessing audit compares.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config

#: The offset a misconfigured agent logging in UTC introduces for an IST cohort.
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


@dataclass
class CorruptionReport:
    """What was injected, so the pipeline audit has something to be checked against."""

    tz_broken_users: list[str] = field(default_factory=list)
    tz_shifted_rows: int = 0
    duplicate_rows: int = 0
    impossible_values: int = 0
    dropout_gaps: int = 0
    dropout_rows: int = 0
    hiccup_gaps: int = 0
    hiccup_rows: int = 0
    rows_before: int = 0
    rows_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def corrupt_logs(
    logs: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, CorruptionReport]:
    """Return a realistically broken copy of ``logs`` plus a report of the damage."""
    rng = rng or np.random.default_rng(cfg.seed + 7)
    pp = cfg.section("preprocessing")
    out = logs.copy()
    report = CorruptionReport(rows_before=len(out))

    out = _inject_dropouts(out, pp, rng, report)
    out = _inject_hiccups(out, pp, rng, report)
    out = _inject_timezone_fault(out, pp, rng, report)
    out = _inject_impossible_values(out, pp, rng, report)
    out = _inject_duplicates(out, pp, rng, report)

    out = out.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    report.rows_after = len(out)
    return out, report


# ---------------------------------------------------------------------------
def _cut_runs(
    logs: pd.DataFrame,
    fraction: float,
    length_range: tuple[int, int] | list[int],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int, int]:
    """Delete runs of consecutive minutes, never straddling two participants."""
    target = int(len(logs) * float(fraction))
    if target <= 0:
        return logs, 0, 0

    lo, hi = int(length_range[0]), int(length_range[1])
    users = logs["user_id"].to_numpy()
    drop_mask = np.zeros(len(logs), dtype=bool)
    n_gaps = removed = guard = 0
    while removed < target and guard < 200_000:
        guard += 1
        start = int(rng.integers(0, len(logs)))
        end = min(start + int(rng.integers(lo, hi + 1)), len(logs))
        if users[start] != users[end - 1]:
            continue
        if drop_mask[start:end].any():
            continue
        drop_mask[start:end] = True
        removed += end - start
        n_gaps += 1

    return logs.loc[~drop_mask].reset_index(drop=True), n_gaps, int(drop_mask.sum())


def _inject_dropouts(
    logs: pd.DataFrame,
    pp: dict,
    rng: np.random.Generator,
    report: CorruptionReport,
) -> pd.DataFrame:
    """Long outages - the agent was offline, asleep or killed.

    These cannot be interpolated: the pipeline must leave them empty rather
    than invent activity for minutes nobody observed.
    """
    out, n_gaps, n_rows = _cut_runs(
        logs, pp["dropout_fraction"], pp["dropout_length"], rng
    )
    report.dropout_gaps = n_gaps
    report.dropout_rows = n_rows
    return out


def _inject_hiccups(
    logs: pd.DataFrame,
    pp: dict,
    rng: np.random.Generator,
    report: CorruptionReport,
) -> pd.DataFrame:
    """Brief 1-2 minute stalls - the agent hiccuped but the user was there.

    Short enough that interpolating across them is safe, which is the case
    step 5 of the pipeline handles.
    """
    out, n_gaps, n_rows = _cut_runs(
        logs,
        pp.get("hiccup_fraction", 0.0),
        pp.get("hiccup_length", [1, 2]),
        rng,
    )
    report.hiccup_gaps = n_gaps
    report.hiccup_rows = n_rows
    return out


def _inject_timezone_fault(
    logs: pd.DataFrame,
    pp: dict,
    rng: np.random.Generator,
    report: CorruptionReport,
) -> pd.DataFrame:
    """Make a few agents log in UTC instead of the participant's local time."""
    users = np.sort(logs["user_id"].unique())
    n_broken = max(1, int(round(len(users) * float(pp["tz_broken_agent_fraction"]))))
    broken = list(rng.choice(users, size=min(n_broken, len(users)), replace=False))

    mask = logs["user_id"].isin(broken)
    out = logs.copy()
    out.loc[mask, "timestamp"] = out.loc[mask, "timestamp"] - IST_OFFSET

    report.tz_broken_users = sorted(str(u) for u in broken)
    report.tz_shifted_rows = int(mask.sum())
    return out


def _inject_impossible_values(
    logs: pd.DataFrame,
    pp: dict,
    rng: np.random.Generator,
    report: CorruptionReport,
) -> pd.DataFrame:
    """Stuck keys, negative counts and out-of-range idle readings."""
    out = logs.copy()
    n = len(out)
    target = int(n * float(pp["impossible_injection_fraction"]))
    if target <= 0:
        return out

    # The faults are deliberately outside the normal range, so widen the count
    # columns first rather than letting the assignment be rejected as lossy.
    for col in ("keystrokes", "idle_seconds", "mouse_clicks"):
        out[col] = out[col].astype("int64")

    idx = rng.choice(n, size=target, replace=False)
    # Split the damage across three plausible sensor faults.
    third = max(1, target // 3)
    stuck, bad_idle, negatives = idx[:third], idx[third:2 * third], idx[2 * third:]

    max_keys = int(pp["max_keystrokes_per_min"])
    out.loc[out.index[stuck], "keystrokes"] = rng.integers(
        max_keys + 40, max_keys * 6, size=len(stuck)
    )
    out.loc[out.index[bad_idle], "idle_seconds"] = rng.choice(
        [-5, -1, 75, 120, 3600], size=len(bad_idle)
    )
    out.loc[out.index[negatives], "mouse_clicks"] = -rng.integers(
        1, 20, size=len(negatives)
    )

    report.impossible_values = target
    return out


def _inject_duplicates(
    logs: pd.DataFrame,
    pp: dict,
    rng: np.random.Generator,
    report: CorruptionReport,
) -> pd.DataFrame:
    """Re-upload some batches - the agent retried and the server kept both copies."""
    n = len(logs)
    target = int(n * float(pp["duplicate_upload_fraction"]))
    if target <= 0:
        return logs

    # Duplicates arrive as contiguous batches, not scattered single rows.
    batches: list[np.ndarray] = []
    taken = 0
    while taken < target:
        size = int(rng.integers(20, 120))
        start = int(rng.integers(0, max(1, n - size)))
        batches.append(np.arange(start, min(start + size, n)))
        taken += size

    dup = logs.iloc[np.concatenate(batches)].copy()
    report.duplicate_rows = len(dup)
    return pd.concat([logs, dup], ignore_index=True)
