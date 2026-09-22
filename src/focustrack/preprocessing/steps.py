"""The individual repair steps of the processing engine.

Each function takes the minute-level frame, fixes exactly one class of defect
and appends an :class:`AuditEntry` describing what it changed. The audit is
what Figure 1 of the report plots.

The guiding rule throughout: repair what can be repaired from evidence, and
leave the rest empty. Activity is never invented for a long gap, and a label
is never imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import ACTIVITY_COLUMNS, RAW_COLUMNS
from focustrack.preprocessing.apps import categorise_series

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


@dataclass
class AuditEntry:
    """One row of the preprocessing audit."""

    step: int
    name: str
    rows_in: int
    rows_out: int
    affected: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Audit(list):
    """An ordered list of :class:`AuditEntry` with a readable summary."""

    def add(
        self,
        step: int,
        name: str,
        rows_in: int,
        rows_out: int,
        affected: int,
        **detail: Any,
    ) -> AuditEntry:
        entry = AuditEntry(step, name, rows_in, rows_out, affected, detail)
        self.append(entry)
        return entry

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self])

    def describe(self) -> str:
        lines = [f"{'#':>2}  {'step':34} {'rows in':>10} {'rows out':>10} {'affected':>10}"]
        lines.append("-" * 70)
        for e in self:
            lines.append(
                f"{e.step:>2}  {e.name:34} {e.rows_in:>10,} {e.rows_out:>10,} {e.affected:>10,}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# step 1 - schema
# ---------------------------------------------------------------------------
def step_validate_schema(df: pd.DataFrame, cfg: Config, audit: Audit) -> pd.DataFrame:
    """Check the columns exist and coerce them to their declared types."""
    rows_in = len(df)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"raw log is missing required columns: {missing}")

    out = df.loc[:, list(RAW_COLUMNS)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for col in ACTIVITY_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["user_id"] = out["user_id"].astype(str)
    out["active_app"] = out["active_app"].astype(str)

    bad_ts = int(out["timestamp"].isna().sum())
    out = out.loc[out["timestamp"].notna()].reset_index(drop=True)
    audit.add(
        1, "validate schema and types", rows_in, len(out), bad_ts,
        unparseable_timestamps=bad_ts,
    )
    return out


# ---------------------------------------------------------------------------
# step 2 - timezone
# ---------------------------------------------------------------------------
def step_fix_timezones(df: pd.DataFrame, cfg: Config, audit: Audit) -> pd.DataFrame:
    """Detect and shift agents that logged in UTC instead of local time.

    Nobody in the cohort starts work before 06:00 local. An agent whose day
    consistently appears to start before that is logging UTC, so its rows are
    shifted forward by the IST offset.
    """
    rows_in = len(df)
    earliest = float(cfg["preprocessing.earliest_plausible_local_hour"])
    out = df.copy()

    day = out["timestamp"].dt.normalize()
    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    # The first timestamp of each user-day.
    start_hours = (
        pd.DataFrame({"user_id": out["user_id"], "day": day, "hour": hour})
        .groupby(["user_id", "day"], observed=True)["hour"]
        .min()
        .reset_index()
    )
    # An agent is broken if the *median* of its day-start hours is implausible;
    # a single early morning is not enough evidence.
    median_start = start_hours.groupby("user_id")["hour"].median()
    broken = sorted(median_start.index[median_start < earliest].astype(str))

    mask = out["user_id"].isin(broken)
    shifted = int(mask.sum())
    if shifted:
        out.loc[mask, "timestamp"] = out.loc[mask, "timestamp"] + IST_OFFSET

    audit.add(
        2, "repair UTC timestamps", rows_in, len(out), shifted,
        broken_agents=broken,
        median_start_hour={u: round(float(median_start[u]), 2) for u in broken},
    )
    return out


# ---------------------------------------------------------------------------
# step 3 - duplicates
# ---------------------------------------------------------------------------
def step_drop_duplicates(df: pd.DataFrame, cfg: Config, audit: Audit) -> pd.DataFrame:
    """Remove re-uploaded rows - one record per participant per minute."""
    rows_in = len(df)
    out = (
        df.sort_values(["user_id", "timestamp"], kind="stable")
        .drop_duplicates(subset=["user_id", "timestamp"], keep="first")
        .reset_index(drop=True)
    )
    removed = rows_in - len(out)
    audit.add(3, "drop duplicate uploads", rows_in, len(out), removed, removed=removed)
    return out


# ---------------------------------------------------------------------------
# step 4 - impossible values
# ---------------------------------------------------------------------------
def step_flag_impossible(df: pd.DataFrame, cfg: Config, audit: Audit) -> pd.DataFrame:
    """Set physically impossible readings to missing rather than clipping them.

    A stuck key reporting 3,000 keystrokes in a minute is not a very fast
    typist, and clipping it to the maximum would assert something the sensor
    never observed.
    """
    rows_in = len(df)
    out = df.copy()
    max_keys = float(cfg["preprocessing.max_keystrokes_per_min"])
    idle_lo, idle_hi = cfg["preprocessing.idle_seconds_range"]

    faults: dict[str, int] = {}

    stuck = out["keystrokes"] > max_keys
    faults["stuck_keys"] = int(stuck.sum())
    out.loc[stuck, "keystrokes"] = np.nan

    bad_idle = (out["idle_seconds"] < idle_lo) | (out["idle_seconds"] > idle_hi)
    faults["idle_out_of_range"] = int(bad_idle.sum())
    out.loc[bad_idle, "idle_seconds"] = np.nan

    negative = 0
    for col in ACTIVITY_COLUMNS:
        neg = out[col] < 0
        negative += int(neg.sum())
        out.loc[neg, col] = np.nan
    faults["negative_counts"] = negative

    total = sum(faults.values())
    audit.add(
        4, "impossible values to missing", rows_in, len(out), total, **faults
    )
    return out


# ---------------------------------------------------------------------------
# step 5 - gaps
# ---------------------------------------------------------------------------
def step_fill_gaps(df: pd.DataFrame, cfg: Config, audit: Audit) -> pd.DataFrame:
    """Reindex each user-day to a continuous minute grid and fill short gaps.

    Gaps of one or two minutes are interpolated - the agent hiccuped but the
    user was there. Anything longer means the agent was offline, and those
    minutes stay empty instead of being invented.
    """
    rows_in = len(df)
    max_fill = int(cfg["preprocessing.max_gap_fill_minutes"])

    out = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    out["date"] = out["timestamp"].dt.normalize()

    frames: list[pd.DataFrame] = []
    short_gaps = long_gaps = filled_rows = long_gap_rows = 0

    for (user, _date), group in out.groupby(["user_id", "date"], sort=False, observed=True):
        grid = pd.date_range(
            group["timestamp"].iloc[0], group["timestamp"].iloc[-1], freq="1min"
        )
        g = group.set_index("timestamp").reindex(grid)
        g.index.name = "timestamp"
        g["user_id"] = user

        inserted = g["state_label"].isna()
        if inserted.any():
            # Measure the length of each inserted run.
            run_id = (inserted != inserted.shift()).cumsum()
            for _, run in g.loc[inserted].groupby(run_id[inserted], observed=True):
                if len(run) <= max_fill:
                    short_gaps += 1
                    filled_rows += len(run)
                else:
                    long_gaps += 1
                    long_gap_rows += len(run)

            lengths = inserted.groupby(run_id).transform("sum")
            fillable = inserted & (lengths <= max_fill)
            g["is_imputed"] = fillable
            if fillable.any():
                num = g.loc[:, list(ACTIVITY_COLUMNS)].interpolate(
                    method="time", limit=max_fill, limit_area="inside"
                )
                g.loc[fillable, list(ACTIVITY_COLUMNS)] = num.loc[fillable]
                g["active_app"] = g["active_app"].ffill(limit=max_fill)
            # Rows belonging to long gaps carry no evidence at all.
            g.loc[inserted & ~fillable, "is_imputed"] = False
        else:
            g["is_imputed"] = False

        g["is_gap"] = inserted & ~g["is_imputed"]
        frames.append(g.reset_index())

    out = pd.concat(frames, ignore_index=True)
    audit.add(
        5, "fill short gaps, keep long ones", rows_in, len(out), filled_rows,
        short_gaps=short_gaps, filled_rows=filled_rows,
        long_gaps=long_gaps, long_gap_rows=long_gap_rows,
    )
    return out


# ---------------------------------------------------------------------------
# step 6 - label integrity
# ---------------------------------------------------------------------------
def step_label_integrity(df: pd.DataFrame, cfg: Config, audit: Audit) -> pd.DataFrame:
    """Guarantee labels are observed, never invented.

    Imputing a label would teach the model the interpolation rule rather than
    the behaviour, so minutes without an observed label stay unlabelled and are
    excluded from supervised training later.
    """
    rows_in = len(df)
    out = df.copy()
    out["state_label"] = out["state_label"].where(out["state_label"].notna(), None)
    unlabelled = int(out["state_label"].isna().sum())
    # A label carried into an imputed minute would be invented, so drop it.
    out.loc[out.get("is_imputed", False) == True, "state_label"] = None  # noqa: E712
    out["has_label"] = out["state_label"].notna()
    audit.add(
        6, "protect label integrity", rows_in, len(out), unlabelled,
        unlabelled_rows=int((~out["has_label"]).sum()),
    )
    return out


# ---------------------------------------------------------------------------
# step 7 - app categories
# ---------------------------------------------------------------------------
def step_categorise_apps(df: pd.DataFrame, cfg: Config, audit: Audit) -> pd.DataFrame:
    """Group ``active_app`` into the six categories."""
    rows_in = len(df)
    out = df.copy()
    out["app_category"] = categorise_series(out["active_app"])
    counts = out["app_category"].value_counts(dropna=False)
    audit.add(
        7, "group apps into 6 categories", rows_in, len(out), int(out["active_app"].nunique()),
        distinct_apps=int(out["active_app"].nunique()),
        category_counts={str(k): int(v) for k, v in counts.items()},
    )
    return out
