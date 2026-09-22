"""Models, drop labelling, storage and the end-to-end run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from focustrack.constants import DISTRACTED, FOCUSED, STATES
from focustrack.models.baseline import RuleBasedClassifier, build_rule_baseline
from focustrack.models.focus_drop import DROP_LABEL, build_drop_labels, focused_windows
from focustrack.models.focus_state import build_models, train_focus_state
from focustrack.models.importance import SIGNAL_GROUPS
from focustrack.preprocessing.features import FEATURE_COLUMNS, feature_matrix
from focustrack.preprocessing.split import make_split


@pytest.fixture(scope="module")
def split(small_config, personas):
    users = pd.DataFrame([p.to_row() for p in personas])
    return make_split(users, small_config)


# --- the rule baseline ------------------------------------------------------
def test_rule_baseline_predicts_only_valid_states(windows, base_config):
    model = build_rule_baseline(base_config).fit(feature_matrix(windows))
    predictions = model.predict(feature_matrix(windows))
    assert set(np.unique(predictions)) <= set(STATES)
    assert len(predictions) == len(windows)


def test_rule_baseline_probabilities_are_one_hot(windows, base_config):
    model = build_rule_baseline(base_config).fit(feature_matrix(windows))
    proba = model.predict_proba(feature_matrix(windows).head(50))
    assert proba.shape == (50, len(STATES))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(proba)) <= {0.0, 1.0}


def test_rule_baseline_needs_named_columns():
    model = RuleBasedClassifier()
    with pytest.raises(TypeError):
        model.predict(np.zeros((3, len(FEATURE_COLUMNS))))


# --- model construction -----------------------------------------------------
def test_all_four_candidates_are_built(base_config):
    models = build_models(base_config)
    assert list(models) == [
        "Rule-based heuristic", "Logistic Regression", "Random Forest", "Gradient Boosting"
    ]


def test_learned_models_balance_the_classes(base_config):
    """Distracted is the rare class; every learned model must be told so."""
    models = build_models(base_config)
    assert models["Random Forest"].class_weight == "balanced"
    assert models["Gradient Boosting"].class_weight == "balanced"
    assert models["Logistic Regression"].named_steps["clf"].class_weight == "balanced"


# --- training ---------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(windows, split, small_config):
    return train_focus_state(
        windows, split.train_users, split.test_users, small_config, verbose=False
    )


def test_training_scores_every_candidate(trained):
    assert len(trained.scores) == 4
    for scores in trained.scores:
        assert 0.0 <= scores.accuracy <= 1.0
        assert 0.0 <= scores.macro_f1 <= 1.0


def test_best_model_is_chosen_on_macro_f1(trained):
    best = max(trained.scores, key=lambda s: s.macro_f1)
    assert trained.best_name == best.name


def test_learned_models_beat_the_rule_baseline(trained):
    by_name = {s.name: s for s in trained.scores}
    baseline = by_name["Rule-based heuristic"]
    forest = by_name["Random Forest"]
    assert forest.macro_f1 > baseline.macro_f1
    assert forest.f1_distracted > baseline.f1_distracted, (
        "the learned model should find distraction the rules cannot see"
    )


def test_confusion_matrix_shape_and_total(trained, windows, split):
    assert trained.confusion.shape == (len(STATES), len(STATES))
    test_rows = windows.loc[
        windows["user_id"].isin(split.test_users) & windows["state_label"].notna()
    ]
    assert trained.confusion.sum() == len(test_rows)


def test_importances_cover_every_feature(trained):
    assert len(trained.importances) == len(FEATURE_COLUMNS)
    assert set(trained.importances["feature"]) == set(FEATURE_COLUMNS)


def test_signal_groups_cover_every_feature():
    grouped = {f for features in SIGNAL_GROUPS.values() for f in features}
    assert grouped == set(FEATURE_COLUMNS), (
        "every feature should belong to exactly one signal family"
    )


def test_grouped_importance_is_computed(trained):
    assert not trained.group_importances.empty
    assert set(trained.group_importances["group"]) == set(SIGNAL_GROUPS)


def test_cross_validation_is_reported_for_learned_models(trained):
    for scores in trained.scores:
        if scores.name == "Rule-based heuristic":
            assert scores.cv_macro_f1_mean is None
        else:
            assert scores.cv_macro_f1_mean is not None
            assert 0.0 <= scores.cv_macro_f1_mean <= 1.0


def test_mixed_windows_are_harder_than_clean_ones(trained):
    """Errors should concentrate at transitions, not spread evenly."""
    by_name = {s.name: s for s in trained.scores}
    forest = by_name["Random Forest"]
    assert forest.accuracy_single_state > forest.accuracy_mixed_state


# --- focus-drop labelling ---------------------------------------------------
def test_drop_label_counts_minutes_in_the_horizon(minutes, windows, small_config):
    labelled = build_drop_labels(minutes, windows, small_config)
    assert DROP_LABEL in labelled.columns
    values = labelled[DROP_LABEL].dropna().unique()
    assert set(values) <= {0.0, 1.0}


def test_windows_without_an_observable_future_are_unlabelled(minutes, windows, small_config):
    labelled = build_drop_labels(minutes, windows, small_config)
    # The last windows of each day cannot see 15 minutes ahead.
    last = labelled.sort_values("window_start").groupby(
        ["user_id", "date"], observed=True
    ).tail(1)
    assert last[DROP_LABEL].isna().any(), (
        "end-of-day windows should be unlabelled, not silently labelled zero"
    )


def test_drop_label_matches_a_hand_count(minutes, windows, small_config):
    """Recompute one window's label directly from the minute log."""
    horizon = int(small_config["models.focus_drop.horizon_minutes"])
    threshold = int(small_config["models.focus_drop.distracted_minutes_threshold"])
    window_minutes = int(small_config["preprocessing.window_minutes"])

    labelled = build_drop_labels(minutes, windows, small_config)
    candidate = labelled.loc[labelled[DROP_LABEL].notna()].iloc[len(labelled) // 3]

    start = pd.Timestamp(candidate["window_start"]) + pd.Timedelta(minutes=window_minutes)
    end = start + pd.Timedelta(minutes=horizon)
    ahead = minutes.loc[
        (minutes["user_id"] == candidate["user_id"])
        & (minutes["timestamp"] >= start)
        & (minutes["timestamp"] < end)
        & minutes["has_label"]
    ]
    expected = float((ahead["state_label"] == DISTRACTED).sum() >= threshold)
    assert candidate[DROP_LABEL] == expected


def test_focused_windows_are_the_only_ones_scored(minutes, windows, small_config):
    labelled = build_drop_labels(minutes, windows, small_config)
    pool = focused_windows(labelled)
    assert (pool["state_label"] == FOCUSED).all()
    assert pool[DROP_LABEL].notna().all()


# --- storage ----------------------------------------------------------------
def test_encrypted_store_round_trip(tmp_path, small_config):
    from focustrack.storage.db import FocusStore

    store = FocusStore(
        db_path=tmp_path / "store.db", key_path=tmp_path / "store.key", encrypt=True
    )
    rows = [
        {
            "user_id": "U001",
            "timestamp": pd.Timestamp("2026-08-03 09:00") + pd.Timedelta(minutes=i),
            "active_app": "Visual Studio Code",
            "keystrokes": 40 + i, "mouse_clicks": 3, "mouse_distance_px": 1200,
            "scroll_events": 2, "window_switches": 1, "idle_seconds": 5,
            "state_label": FOCUSED,
        }
        for i in range(5)
    ]
    assert store.record_activity(rows) == 5

    loaded = store.load_activity(user_id="U001")
    assert len(loaded) == 5
    assert loaded["keystrokes"].tolist() == [40, 41, 42, 43, 44]
    assert store.stats().activity_rows == 5


def test_stored_payload_is_not_plain_text(tmp_path):
    """The readings on disk should not be greppable."""
    import sqlite3

    from focustrack.storage.crypto import CRYPTO_AVAILABLE
    from focustrack.storage.db import FocusStore

    if not CRYPTO_AVAILABLE:
        pytest.skip("cryptography is not installed")

    db_path = tmp_path / "store.db"
    store = FocusStore(db_path=db_path, key_path=tmp_path / "store.key", encrypt=True)
    store.record_activity([{
        "user_id": "U001", "timestamp": pd.Timestamp("2026-08-03 09:00"),
        "active_app": "SecretInternalTool", "keystrokes": 1, "mouse_clicks": 0,
        "mouse_distance_px": 0, "scroll_events": 0, "window_switches": 0,
        "idle_seconds": 0, "state_label": FOCUSED,
    }])

    with sqlite3.connect(db_path) as connection:
        blob = connection.execute("SELECT payload FROM activity").fetchone()[0]
    assert b"SecretInternalTool" not in bytes(blob)


def test_wrong_key_cannot_read_the_store(tmp_path):
    from focustrack.storage.crypto import CRYPTO_AVAILABLE, DecryptionError, generate_key
    from focustrack.storage.db import FocusStore

    if not CRYPTO_AVAILABLE:
        pytest.skip("cryptography is not installed")

    db_path = tmp_path / "store.db"
    store = FocusStore(db_path=db_path, key_path=tmp_path / "a.key", encrypt=True)
    store.record_activity([{
        "user_id": "U001", "timestamp": pd.Timestamp("2026-08-03 09:00"),
        "active_app": "Slack", "keystrokes": 1, "mouse_clicks": 0,
        "mouse_distance_px": 0, "scroll_events": 0, "window_switches": 0,
        "idle_seconds": 0, "state_label": FOCUSED,
    }])

    (tmp_path / "b.key").write_bytes(generate_key())
    other = FocusStore(db_path=db_path, key_path=tmp_path / "b.key", encrypt=True)
    with pytest.raises(DecryptionError):
        other.load_activity()


# --- the collector's privacy boundary --------------------------------------
def test_collector_records_counts_not_content():
    from focustrack.agent.collector import ActivityCollector, MinuteCounts

    collector = ActivityCollector()
    for _ in range(7):
        collector._on_key("a-secret-password-character")
    counts = collector.drain()

    assert counts.keystrokes == 7
    row = counts.as_row("U001", pd.Timestamp("2026-08-03 09:00"))
    # Nothing in the row can hold what was typed.
    assert set(row) == {
        "user_id", "timestamp", "active_app", "keystrokes", "mouse_clicks",
        "mouse_distance_px", "scroll_events", "window_switches", "idle_seconds",
        "state_label",
    }
    assert all(
        not isinstance(v, str) or "secret" not in v.lower()
        for v in row.values() if v is not None
    )


def test_collector_drain_resets_the_counters():
    from focustrack.agent.collector import ActivityCollector

    collector = ActivityCollector()
    collector._on_key("x")
    collector._on_click(0, 0, None, True)
    first = collector.drain()
    second = collector.drain()

    assert first.keystrokes == 1 and first.mouse_clicks == 1
    assert second.keystrokes == 0 and second.mouse_clicks == 0
