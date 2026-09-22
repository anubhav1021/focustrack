"""The processing engine: eleven steps from a raw agent log to model input.

    1  validate the schema and coerce types
    2  repair agents that logged in UTC
    3  drop duplicate uploads
    4  set impossible readings to missing
    5  fill 1-2 minute gaps, leave longer ones empty
    6  protect label integrity - never impute a label
    7  group applications into six categories
    8  detect breaks (three or more consecutive idle minutes)
    9  normalise activity counts within each user
   10  group minutes into five-minute windows
   11  build the 31 features

Steps 1-9 produce a repaired minute-level frame, which the live agent also
uses for its own five-minute cycle. Steps 10-11 produce the window-level
frame the models are trained and served on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from focustrack.config import Config
from focustrack.io_utils import write_json, write_parquet
from focustrack.preprocessing.breaks import detect_breaks
from focustrack.preprocessing.features import FEATURE_COLUMNS, METADATA_COLUMNS
from focustrack.preprocessing.normalise import UserScaler, normalise_per_user
from focustrack.preprocessing.steps import (
    Audit,
    step_categorise_apps,
    step_drop_duplicates,
    step_fill_gaps,
    step_flag_impossible,
    step_fix_timezones,
    step_label_integrity,
    step_validate_schema,
)
from focustrack.preprocessing.windows import add_history, add_time_of_day, build_windows

MINUTES_FILE = "minutes.parquet"
WINDOWS_FILE = "windows.parquet"
AUDIT_FILE = "preprocessing_audit.json"
SCALER_FILE = "user_scaler.json"


@dataclass
class PipelineResult:
    """Everything the processing engine produces in one pass."""

    minutes: pd.DataFrame
    windows: pd.DataFrame
    audit: Audit
    scaler: UserScaler

    def summary(self) -> dict[str, Any]:
        labelled = self.windows["state_label"].notna()
        return {
            "minute_rows": int(len(self.minutes)),
            "windows": int(len(self.windows)),
            "labelled_windows": int(labelled.sum()),
            "n_features": len(FEATURE_COLUMNS),
            "users": int(self.windows["user_id"].nunique()),
            "user_days": int(
                self.windows.groupby(["user_id", "date"], observed=True).ngroups
            ),
            "state_distribution": {
                str(k): float(v)
                for k, v in self.windows.loc[labelled, "state_label"]
                .value_counts(normalize=True)
                .items()
            },
            "steps": [e.to_dict() for e in self.audit],
        }


def run_pipeline(
    raw: pd.DataFrame,
    cfg: Config,
    scaler: UserScaler | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """Run all eleven steps over a raw activity log."""
    audit = Audit()

    def announce(msg: str) -> None:
        if verbose:
            print(f"    {msg}", flush=True)

    announce("1/11 schema")
    df = step_validate_schema(raw, cfg, audit)
    announce("2/11 timezones")
    df = step_fix_timezones(df, cfg, audit)
    announce("3/11 duplicates")
    df = step_drop_duplicates(df, cfg, audit)
    announce("4/11 impossible values")
    df = step_flag_impossible(df, cfg, audit)
    announce("5/11 gaps")
    df = step_fill_gaps(df, cfg, audit)
    announce("6/11 label integrity")
    df = step_label_integrity(df, cfg, audit)
    announce("7/11 app categories")
    df = step_categorise_apps(df, cfg, audit)
    announce("8/11 break detection")
    df = detect_breaks(df, cfg, audit)
    announce("9/11 per-user normalisation")
    df, scaler = normalise_per_user(df, cfg, audit, scaler=scaler)

    announce("10/11 windowing")
    windows = build_windows(df, cfg, audit)
    announce("11/11 features")
    windows = add_time_of_day(windows, cfg)
    windows = add_history(windows, cfg)

    keep = [c for c in (*METADATA_COLUMNS, *FEATURE_COLUMNS) if c in windows.columns]
    windows = windows.loc[:, keep]
    audit.add(
        11, "build the 31-feature window table", len(df), len(windows),
        len(FEATURE_COLUMNS), n_features=len(FEATURE_COLUMNS),
    )
    return PipelineResult(minutes=df, windows=windows, audit=audit, scaler=scaler)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_result(result: PipelineResult, cfg: Config) -> dict[str, Path]:
    """Write the minute frame, the window frame, the audit and the scaler."""
    interim = cfg.path("interim")
    processed = cfg.path("processed")
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    minutes_path = interim / MINUTES_FILE
    windows_path = processed / WINDOWS_FILE
    audit_path = cfg.path("reports") / AUDIT_FILE
    scaler_path = processed / SCALER_FILE

    minutes = result.minutes.copy()
    # Parquet wants a concrete dtype for the label column.
    minutes["state_label"] = minutes["state_label"].astype("string")
    minutes["app_category"] = minutes["app_category"].astype("string")
    write_parquet(minutes, minutes_path)

    windows = result.windows.copy()
    windows["state_label"] = windows["state_label"].astype("string")
    write_parquet(windows, windows_path)

    write_json(result.summary(), audit_path)
    write_json(result.scaler.to_dict(), scaler_path)

    return {
        "minutes": minutes_path,
        "windows": windows_path,
        "audit": audit_path,
        "scaler": scaler_path,
    }


def load_windows(cfg: Config) -> pd.DataFrame:
    """Load the processed window table."""
    path = cfg.path("processed") / WINDOWS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `focustrack preprocess` first.")
    df = pd.read_parquet(path)
    df["state_label"] = df["state_label"].astype(object).where(df["state_label"].notna(), None)
    return df


def load_minutes(cfg: Config) -> pd.DataFrame:
    """Load the repaired minute-level table."""
    path = cfg.path("interim") / MINUTES_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `focustrack preprocess` first.")
    return pd.read_parquet(path)


def load_scaler(cfg: Config) -> UserScaler:
    path = cfg.path("processed") / SCALER_FILE
    with path.open("r", encoding="utf-8") as fh:
        return UserScaler.from_dict(json.load(fh))
