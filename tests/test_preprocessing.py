"""The 11-step pipeline.

These tests check the pipeline's *promises*, not just that it runs: that it
repairs what it claims to repair, and - just as importantly - that it refuses
to invent anything it cannot observe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from focustrack.constants import CATEGORIES, RAW_COLUMNS, STATES
from focustrack.data.corrupt import IST_OFFSET, corrupt_logs
from focustrack.preprocessing.apps import categorise_app, categorise_series
from focustrack.preprocessing.breaks import detect_breaks, longest_stretch_without_break
from focustrack.preprocessing.features import FEATURE_COLUMNS, N_FEATURES, feature_matrix
from focustrack.preprocessing.pipeline import run_pipeline
from focustrack.preprocessing.split import make_split
from focustrack.preprocessing.steps import Audit, step_flag_impossible, step_fix_timezones


# --- structure --------------------------------------------------------------
def test_pipeline_runs_all_eleven_steps(pipeline_result):
    assert [entry.step for entry in pipeline_result.audit] == list(range(1, 12))


def test_feature_set_is_exactly_31(windows):
    assert N_FEATURES == 31
    assert len(FEATURE_COLUMNS) == 31
    assert len(set(FEATURE_COLUMNS)) == 31
    for column in FEATURE_COLUMNS:
        assert column in windows.columns


def test_feature_matrix_has_no_missing_or_infinite_values(windows):
    matrix = feature_matrix(windows)
    assert matrix.shape[1] == 31
    assert np.isfinite(matrix.to_numpy()).all()


def test_windows_are_five_minutes_apart(windows, small_config):
    minutes = int(small_config["preprocessing.window_minutes"])
    starts = pd.to_datetime(windows["window_start"])
    assert (starts.dt.minute % minutes == 0).all()
    assert (starts.dt.second == 0).all()


def test_sparse_windows_are_dropped(windows, small_config):
    floor = int(small_config["preprocessing.min_minutes_per_window"])
    assert (windows["n_minutes"] >= floor).all()


# --- step 2: timezones ------------------------------------------------------
def test_utc_agent_is_detected_and_shifted(small_config, clean_logs):
    broken_user = clean_logs["user_id"].iloc[0]
    logs = clean_logs.copy()
    mask = logs["user_id"] == broken_user
    logs.loc[mask, "timestamp"] = logs.loc[mask, "timestamp"] - IST_OFFSET

    audit = Audit()
    repaired = step_fix_timezones(logs, small_config, audit)

    entry = audit[-1]
    assert broken_user in entry.detail["broken_agents"]
    assert entry.affected == int(mask.sum())

    # The repaired stamps match the originals exactly.
    original = clean_logs.loc[mask, "timestamp"].reset_index(drop=True)
    fixed = repaired.loc[repaired["user_id"] == broken_user, "timestamp"].reset_index(drop=True)
    pd.testing.assert_series_equal(original, fixed, check_names=False)


def test_healthy_agents_are_left_alone(small_config, clean_logs):
    audit = Audit()
    repaired = step_fix_timezones(clean_logs.copy(), small_config, audit)
    assert audit[-1].affected == 0
    pd.testing.assert_series_equal(
        clean_logs["timestamp"], repaired["timestamp"], check_names=False
    )


# --- step 3: duplicates -----------------------------------------------------
def test_duplicate_minutes_are_removed(pipeline_result):
    minutes = pipeline_result.minutes
    key = minutes.loc[:, ["user_id", "timestamp"]]
    assert not key.duplicated().any()


# --- step 4: impossible values ---------------------------------------------
def test_impossible_values_become_missing_not_clipped(small_config, clean_logs):
    logs = clean_logs.head(50).copy()
    logs["keystrokes"] = logs["keystrokes"].astype("int64")
    logs["idle_seconds"] = logs["idle_seconds"].astype("int64")
    logs.loc[logs.index[0], "keystrokes"] = 5_000      # a stuck key
    logs.loc[logs.index[1], "idle_seconds"] = 3_600    # impossible in a minute
    logs.loc[logs.index[2], "mouse_clicks"] = -4       # a counter that wrapped

    audit = Audit()
    cleaned = step_flag_impossible(logs, small_config, audit)

    assert pd.isna(cleaned.loc[cleaned.index[0], "keystrokes"])
    assert pd.isna(cleaned.loc[cleaned.index[1], "idle_seconds"])
    assert pd.isna(cleaned.loc[cleaned.index[2], "mouse_clicks"])
    # Crucially, not clipped to the boundary - that would assert an observation.
    assert cleaned.loc[cleaned.index[0], "keystrokes"] != 600
    assert audit[-1].affected == 3


def test_surviving_values_stay_in_range(minutes, small_config):
    low, high = small_config["preprocessing.idle_seconds_range"]
    idle = minutes["idle_seconds"].dropna()
    assert idle.between(low, high).all()
    keys = minutes["keystrokes"].dropna()
    assert (keys <= small_config["preprocessing.max_keystrokes_per_min"]).all()


# --- step 5/6: gaps and labels ---------------------------------------------
def test_short_gaps_filled_and_long_gaps_left_empty(pipeline_result, small_config):
    detail = pipeline_result.audit[4].detail
    limit = int(small_config["preprocessing.max_gap_fill_minutes"])
    assert detail["short_gaps"] > 0, "the fixture should contain brief agent stalls"
    assert detail["long_gaps"] > 0, "the fixture should contain agent outages"
    # Long gaps contribute rows but no activity.
    minutes = pipeline_result.minutes
    assert minutes.loc[minutes["is_gap"], "keystrokes"].isna().all()
    assert detail["filled_rows"] <= detail["short_gaps"] * limit


def test_labels_are_never_imputed(pipeline_result):
    minutes = pipeline_result.minutes
    invented = minutes.loc[minutes["is_imputed"] | minutes["is_gap"], "state_label"]
    assert invented.isna().all(), "a label was invented for a minute nobody observed"


def test_every_window_label_is_a_real_state(windows):
    labels = set(windows["state_label"].dropna().unique())
    assert labels <= set(STATES)


# --- step 7: categories -----------------------------------------------------
def test_apps_map_to_the_six_categories(minutes):
    categories = set(minutes["app_category"].dropna().astype(str).unique())
    assert categories <= set(CATEGORIES) | {"unknown"}


@pytest.mark.parametrize(
    "app, expected",
    [
        ("Visual Studio Code", "development"),
        ("Figma", "creative"),
        ("Microsoft Excel", "docs_analysis"),
        ("Google Chrome", "research"),
        ("Slack", "communication"),
        ("YouTube", "personal"),
        # outside the catalogue, resolved by keyword
        ("Sublime Text 4", "development"),
        ("Discord", "communication"),
        ("TikTok", "personal"),
        ("something entirely unknown", "unknown"),
    ],
)
def test_app_categorisation(app, expected):
    assert categorise_app(app) == expected


def test_categorise_series_matches_scalar():
    apps = pd.Series(["Slack", "Figma", "mystery app", "YouTube"])
    vectorised = list(categorise_series(apps))
    assert vectorised == [categorise_app(a) for a in apps]


# --- step 8: breaks ---------------------------------------------------------
def test_three_idle_minutes_is_a_break_and_two_is_not(small_config):
    threshold = float(small_config["preprocessing.idle_minute_threshold_seconds"])
    start = pd.Timestamp("2026-08-03 09:00")
    idle_pattern = [5, 5, threshold + 10, threshold + 10, 5,        # 2 idle - not a break
                    threshold + 10, threshold + 10, threshold + 10, 5]  # 3 idle - a break
    frame = pd.DataFrame(
        {
            "user_id": "U001",
            "timestamp": [start + pd.Timedelta(minutes=i) for i in range(len(idle_pattern))],
            "idle_seconds": idle_pattern,
        }
    )
    result = detect_breaks(frame, small_config)
    assert not result["is_break"].iloc[2:4].any(), "two idle minutes is a pause, not a break"
    assert result["is_break"].iloc[5:8].all(), "three idle minutes is a break"


def test_minutes_since_break_resets_after_a_break(minutes):
    for _key, day in minutes.groupby(["user_id", "date"], observed=True):
        day = day.sort_values("timestamp")
        during = day.loc[day["is_break"], "minutes_since_break"]
        assert (during == 0).all()


def test_longest_stretch_counts_non_break_runs():
    assert longest_stretch_without_break([False] * 7) == 7
    assert longest_stretch_without_break([False, False, True, False, False, False]) == 3
    assert longest_stretch_without_break([True, True]) == 0


# --- step 9: normalisation --------------------------------------------------
def test_normalisation_is_within_user(minutes):
    """A fast typist and a slow one should end up on a comparable scale."""
    medians = minutes.groupby("user_id", observed=True)["keystrokes_n"].median()
    assert medians.abs().max() < 0.5, "per-user centring did not work"

    raw_spread = minutes.groupby("user_id", observed=True)["keystrokes"].median()
    assert raw_spread.max() / max(raw_spread.min(), 1) > 1.5, (
        "the fixture should contain genuinely different typing speeds"
    )


# --- split ------------------------------------------------------------------
def test_split_is_by_user_and_covers_every_role(small_config, personas):
    users = pd.DataFrame([p.to_row() for p in personas])
    split = make_split(users, small_config)

    assert not set(split.train_users) & set(split.test_users), "a user appears in both splits"
    assert len(split.train_users) + len(split.test_users) == len(users)
    per_role = int(small_config["split.n_test_users_per_role"])
    for role, members in split.by_role.items():
        assert len(members) == per_role, f"role {role} is not represented in the test split"


def test_no_window_crosses_the_split(windows, small_config, personas):
    users = pd.DataFrame([p.to_row() for p in personas])
    split = make_split(users, small_config)
    train = windows.loc[windows["user_id"].isin(split.train_users), "user_id"].unique()
    assert not set(train) & set(split.test_users)


# --- schema -----------------------------------------------------------------
def test_raw_log_matches_the_published_schema(clean_logs):
    assert list(clean_logs.columns) == list(RAW_COLUMNS)


def test_pipeline_is_deterministic(small_config, raw_logs):
    first = run_pipeline(raw_logs, small_config).windows
    second = run_pipeline(raw_logs, small_config).windows
    pd.testing.assert_frame_equal(first, second)
