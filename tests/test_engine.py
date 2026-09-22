"""The reminder engine and the Focus Score.

The engine's value is as much in when it stays quiet as in when it speaks, so
most of these tests are about silence: not during a meeting, not twice in a
row, not past the daily cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from focustrack.constants import BREAK, BREAK_REMINDER, DISTRACTED, FOCUS_NUDGE, FOCUSED, MEETING
from focustrack.engine.focus_score import (
    compute_focus_score,
    count_break_episodes,
    longest_run_without_break,
)
from focustrack.engine.reminders import EngineSettings, ReminderEngine, run_engine, timer_baseline


@pytest.fixture
def settings(base_config) -> EngineSettings:
    return EngineSettings.from_config(base_config)


@pytest.fixture
def engine(settings) -> ReminderEngine:
    return ReminderEngine(settings, window_minutes=5)


def _run(engine: ReminderEngine, steps, start="2026-08-03 09:00"):
    """Feed ``(state, minutes_since_break, drop_probability)`` tuples in order."""
    clock = pd.Timestamp(start)
    fired = []
    for i, (state, since, probability) in enumerate(steps):
        note = engine.step("U001", clock + pd.Timedelta(minutes=5 * i), state, since, probability)
        if note is not None:
            fired.append(note)
    return fired


# --- firing conditions ------------------------------------------------------
def test_break_suggested_when_drop_is_likely_and_break_is_overdue(engine, settings):
    fired = _run(engine, [(FOCUSED, settings.min_minutes_since_break + 5, 0.9)])
    assert len(fired) == 1
    assert fired[0].kind == BREAK_REMINDER


def test_no_break_when_the_drop_probability_is_low(engine, settings):
    fired = _run(engine, [(FOCUSED, settings.min_minutes_since_break + 5, 0.1)])
    assert fired == []


def test_no_break_too_soon_after_the_last_one(engine, settings):
    """A high-risk moment five minutes after a break is still not worth interrupting."""
    fired = _run(engine, [(FOCUSED, settings.min_minutes_since_break - 10, 0.99)])
    assert fired == []


def test_hard_limit_fires_regardless_of_the_model(engine, settings):
    fired = _run(engine, [(FOCUSED, settings.hard_break_minutes + 1, 0.0)])
    assert len(fired) == 1
    assert "without a break" in fired[0].reason


def test_nudge_after_ten_minutes_of_distraction(engine, settings):
    # Two five-minute windows is ten minutes.
    fired = _run(engine, [(DISTRACTED, 30, 0.1), (DISTRACTED, 35, 0.1)])
    assert len(fired) == 1
    assert fired[0].kind == FOCUS_NUDGE


def test_no_nudge_after_only_five_minutes(engine):
    assert _run(engine, [(DISTRACTED, 30, 0.1)]) == []


# --- silence rules ----------------------------------------------------------
def test_meetings_are_never_interrupted(engine, settings):
    fired = _run(engine, [(MEETING, settings.hard_break_minutes + 50, 0.99)])
    assert fired == [], "the engine interrupted a meeting"


def test_breaks_are_never_interrupted(engine, settings):
    fired = _run(engine, [(BREAK, settings.hard_break_minutes + 50, 0.99)])
    assert fired == []


def test_notifications_are_spaced_out(engine, settings):
    """Two firing conditions back to back should still produce one message."""
    steps = [
        (FOCUSED, settings.min_minutes_since_break + 5, 0.99),
        (DISTRACTED, settings.min_minutes_since_break + 10, 0.99),
        (DISTRACTED, settings.min_minutes_since_break + 15, 0.99),
    ]
    fired = _run(engine, steps)
    gaps = [
        (b.timestamp - a.timestamp).total_seconds() / 60.0
        for a, b in zip(fired, fired[1:])
    ]
    assert all(gap >= settings.min_gap_minutes for gap in gaps)


def test_daily_cap_on_break_reminders(engine, settings):
    # A long day of high-risk focused windows, well spaced.
    steps = [(FOCUSED, 60 + 20 * i, 0.99) for i in range(40)]
    fired = _run(engine, steps)
    breaks = [n for n in fired if n.kind == BREAK_REMINDER]
    assert len(breaks) <= settings.max_breaks_per_day


def test_daily_cap_on_nudges(engine, settings):
    steps = []
    for _ in range(30):
        steps += [(DISTRACTED, 30, 0.1), (DISTRACTED, 35, 0.1), (FOCUSED, 40, 0.1)]
    fired = _run(engine, steps)
    nudges = [n for n in fired if n.kind == FOCUS_NUDGE]
    assert len(nudges) <= settings.max_nudges_per_day


def test_hard_limit_does_not_nag_every_fifteen_minutes(engine, settings):
    """Once past the limit, an ignored reminder backs off instead of repeating."""
    steps = [(FOCUSED, settings.hard_break_minutes + 5 * i, 0.0) for i in range(12)]
    fired = _run(engine, steps)
    assert len(fired) == 1, (
        f"the engine nagged {len(fired)} times for one unbroken stretch"
    )


def test_taking_a_break_re_arms_the_hard_limit(engine, settings):
    steps = [
        (FOCUSED, settings.hard_break_minutes + 1, 0.0),   # fires
        (BREAK, 0, 0.0),                                    # the user stops
        *[(FOCUSED, 20 * i, 0.0) for i in range(1, 5)],     # back to work
        (FOCUSED, settings.hard_break_minutes + 1, 0.0),    # fires again
    ]
    fired = _run(engine, steps)
    assert len(fired) == 2


def test_day_reset_clears_the_caps(engine, settings):
    _run(engine, [(FOCUSED, settings.hard_break_minutes + 1, 0.9)])
    assert engine.state.breaks_sent == 1
    engine.state.reset()
    assert engine.state.breaks_sent == 0
    assert engine.state.last_notification is None


# --- threshold calibration --------------------------------------------------
def test_absolute_mode_uses_the_published_threshold(base_config):
    from focustrack.engine.reminders import resolve_drop_threshold

    cfg = base_config.with_overrides(**{"reminders.drop_threshold_mode": "absolute"})
    assert resolve_drop_threshold(cfg, calibrated=0.24) == pytest.approx(
        cfg["reminders.drop_probability_threshold"]
    )


def test_percentile_mode_uses_the_calibrated_threshold(base_config):
    from focustrack.engine.reminders import resolve_drop_threshold

    cfg = base_config.with_overrides(**{"reminders.drop_threshold_mode": "percentile"})
    assert resolve_drop_threshold(cfg, calibrated=0.24) == pytest.approx(0.24)


def test_percentile_mode_falls_back_when_uncalibrated(base_config):
    """Without a fitted model there is nothing to calibrate against."""
    from focustrack.engine.reminders import resolve_drop_threshold

    cfg = base_config.with_overrides(**{"reminders.drop_threshold_mode": "percentile"})
    assert resolve_drop_threshold(cfg, calibrated=None) == pytest.approx(
        cfg["reminders.drop_probability_threshold"]
    )


def test_a_lower_threshold_fires_more_often(base_config):
    """The threshold is the lever it is supposed to be."""
    steps = [(FOCUSED, 40, 0.30)]
    strict = ReminderEngine(EngineSettings.from_config(base_config, drop_threshold=0.45))
    lenient = ReminderEngine(EngineSettings.from_config(base_config, drop_threshold=0.24))
    assert _run(strict, steps) == []
    assert len(_run(lenient, steps)) == 1


# --- Focus Score ------------------------------------------------------------
def test_perfect_day_scores_one_hundred(base_config):
    states = [FOCUSED] * 12 + [BREAK] * 3 + [FOCUSED] * 12
    breaks = [s == BREAK for s in states]
    score = compute_focus_score(states, switches=0, minutes_per_state=5,
                                is_break=breaks, cfg=base_config)
    assert score.score == pytest.approx(100.0)
    assert score.focus_ratio == pytest.approx(1.0)
    assert score.break_balance == pytest.approx(1.0)
    assert score.switch_load == pytest.approx(0.0)


def test_distraction_lowers_the_score(base_config):
    focused = [FOCUSED] * 24
    half = [FOCUSED] * 12 + [DISTRACTED] * 12
    a = compute_focus_score(focused, 0, 5, None, base_config)
    b = compute_focus_score(half, 0, 5, None, base_config)
    assert b.score < a.score


def test_breaks_do_not_count_against_the_focus_ratio(base_config):
    """Resting is not a failure to focus."""
    without = compute_focus_score([FOCUSED] * 20, 0, 5, None, base_config)
    with_break = compute_focus_score(
        [FOCUSED] * 10 + [BREAK] * 4 + [FOCUSED] * 10, 0, 5, None, base_config
    )
    assert with_break.focus_ratio == pytest.approx(without.focus_ratio)


def test_a_long_unbroken_stretch_is_penalised(base_config):
    healthy = int(base_config["focus_score.healthy_stretch_minutes"])
    zero_at = int(base_config["focus_score.stretch_zero_minutes"])

    short = compute_focus_score([FOCUSED] * (healthy // 5), 0, 5, None, base_config)
    long = compute_focus_score([FOCUSED] * (zero_at // 5), 0, 5, None, base_config)

    assert short.break_balance == pytest.approx(1.0)
    assert long.break_balance == pytest.approx(0.0)
    assert long.score < short.score, "an unbroken marathon should not be a perfect day"


def test_switching_costs_points(base_config):
    calm = compute_focus_score([FOCUSED] * 24, switches=0, minutes_per_state=5,
                               is_break=None, cfg=base_config)
    frantic = compute_focus_score([FOCUSED] * 24, switches=400, minutes_per_state=5,
                                  is_break=None, cfg=base_config)
    assert frantic.score < calm.score
    assert frantic.switch_load > calm.switch_load


def test_score_stays_in_range(base_config):
    for states in ([FOCUSED] * 50, [DISTRACTED] * 50, [BREAK] * 50, []):
        score = compute_focus_score(states, 1000, 5, None, base_config)
        assert 0.0 <= score.score <= 100.0


def test_weights_sum_to_one(base_config):
    weights = (
        base_config["focus_score.w_focus_ratio"]
        + base_config["focus_score.w_switch_load"]
        + base_config["focus_score.w_break_balance"]
    )
    assert weights == pytest.approx(1.0)


def test_break_episode_counting():
    assert count_break_episodes([False, True, True, False, True]) == 2
    assert count_break_episodes([True, True, True]) == 1
    assert count_break_episodes([False, False]) == 0


# --- running over recorded days --------------------------------------------
def test_engine_replays_a_cohort_without_error(windows, base_config):
    frame = windows.copy()
    frame["drop_probability"] = 0.5
    notifications = run_engine(frame, base_config)
    assert set(notifications.columns) >= {"user_id", "timestamp", "kind", "reason"}
    if len(notifications):
        assert set(notifications["kind"]) <= {BREAK_REMINDER, FOCUS_NUDGE}


def test_engine_respects_caps_across_a_whole_cohort(windows, base_config, settings):
    frame = windows.copy()
    frame["drop_probability"] = 0.99          # the worst case for notification volume
    notifications = run_engine(frame, base_config)
    if not len(notifications):
        return
    per_day = notifications.groupby(["user_id", "date", "kind"], observed=True).size()
    for (_user, _date, kind), count in per_day.items():
        cap = settings.max_breaks_per_day if kind == BREAK_REMINDER else settings.max_nudges_per_day
        assert count <= cap


def test_timer_baseline_fires_on_the_clock(windows, base_config):
    alerts = timer_baseline(windows, base_config)
    interval = int(base_config["reminders.timer_baseline_minutes"])
    if len(alerts):
        assert (alerts["minutes_since_break"] >= interval).all()
