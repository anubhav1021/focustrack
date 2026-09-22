"""The four-state behaviour model behind the pilot dataset.

Every minute a participant is in exactly one hidden state - ``Focused``,
``Distracted``, ``Break`` or ``Meeting``. The transition hazards encode the
three effects documented in the report:

* distraction risk rises with the minutes since the last break (fatigue),
* it rises again during the 14:00-16:00 post-lunch slump,
* and it scales with a personal distractibility trait.

The emission model deliberately builds in the three overlaps that make the
classification problem hard and the rule baseline weak:

* **reading looks idle** - focused minutes with almost no keyboard input,
* **daydreaming inside work apps** - distracted minutes on a work surface,
* **personal chat looks like busy typing** - distracted minutes at full speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import (
    APP_CATALOGUE,
    BREAK,
    DISTRACTED,
    FOCUSED,
    MEETING,
    MEETING_APPS,
    RAW_COLUMNS,
)
from focustrack.data.personas import UserPersona

# --- emission "modes" -------------------------------------------------------
# A mode is the flavour of a state; it persists for a few minutes at a time so
# that reading spells and scrolling binges look like spells, not noise.
MODE_TYPING = 0      # Focused, hands on keyboard
MODE_READING = 1     # Focused, reading - looks idle
MODE_BROWSING = 2    # Distracted, scrolling social media
MODE_CHAT = 3        # Distracted, personal chat - looks like busy typing
MODE_DAYDREAM = 4    # Distracted, idling inside a work app
MODE_BREAK = 5       # away from the desk
MODE_MEETING = 6     # in a call

#: Probability of re-drawing the mode on any given minute inside a state.
MODE_RESAMPLE_P = 0.18

#: Applications grouped by category, for sampling.
_APPS_BY_CATEGORY: dict[str, list[str]] = {}
for _app, _cat in APP_CATALOGUE.items():
    _APPS_BY_CATEGORY.setdefault(_cat, []).append(_app)

PERSONAL_APPS: list[str] = _APPS_BY_CATEGORY["personal"]
#: Browsers double as research tools and as distraction surfaces.
RESEARCH_APPS: list[str] = ["Google Chrome", "Mozilla Firefox", "Microsoft Edge"]
#: Personal chat happens on a messenger, not a feed.
CHAT_APPS: list[str] = ["WhatsApp Web", "Instagram", "Slack"]


@dataclass
class DayPlan:
    """The scheduled skeleton of one workday before minute-level simulation."""

    date: pd.Timestamp
    start: pd.Timestamp
    n_minutes: int
    #: ``(start_minute, end_minute)`` offsets of each meeting.
    meetings: list[tuple[int, int]]
    #: ``(start_minute, end_minute)`` of the lunch break, if any.
    lunch: tuple[int, int] | None


# ---------------------------------------------------------------------------
# day planning
# ---------------------------------------------------------------------------
def plan_day(
    persona: UserPersona,
    date: pd.Timestamp,
    cfg: Config,
    rng: np.random.Generator,
) -> DayPlan:
    """Lay out the start time, length, meetings and lunch break of one day."""
    wd = cfg["dataset.workday"]
    start_hour = float(np.clip(rng.normal(persona.start_hour, 0.35), 5.8, 12.0))
    n_minutes = int(
        np.clip(
            rng.normal(persona.day_length_mean, wd["sd_minutes"]),
            wd["min_minutes"],
            wd["max_minutes"],
        )
    )
    start = date.normalize() + pd.Timedelta(minutes=round(start_hour * 60))

    # --- meetings ---------------------------------------------------------
    lam = float(cfg["behaviour.meetings_lambda"][persona.role]) * persona.meeting_factor
    n_meetings = int(rng.poisson(lam))
    lo, hi = cfg["behaviour.meeting_minutes"]
    meetings: list[tuple[int, int]] = []
    for _ in range(n_meetings):
        duration = int(rng.integers(lo, hi + 1))
        for _attempt in range(12):
            begin = int(rng.integers(20, max(21, n_minutes - duration - 10)))
            if all(begin >= e + 10 or begin + duration + 10 <= s for s, e in meetings):
                meetings.append((begin, begin + duration))
                break
    meetings.sort()

    # --- lunch ------------------------------------------------------------
    lunch: tuple[int, int] | None = None
    lunch_lo, lunch_hi = cfg["behaviour.lunch_window"]
    if rng.random() < float(cfg.get("behaviour.lunch_probability", 0.62)):
        target_hour = float(rng.uniform(lunch_lo, lunch_hi))
        offset = int(round((target_hour - start_hour) * 60))
        dur_lo, dur_hi = cfg["behaviour.lunch_minutes"]
        duration = int(rng.integers(dur_lo, dur_hi + 1))
        if 30 < offset < n_minutes - duration - 20:
            if all(offset >= e or offset + duration <= s for s, e in meetings):
                lunch = (offset, offset + duration)

    return DayPlan(
        date=date.normalize(),
        start=start,
        n_minutes=n_minutes,
        meetings=meetings,
        lunch=lunch,
    )


# ---------------------------------------------------------------------------
# hidden-state sequence
# ---------------------------------------------------------------------------
def _draw_distracted_mode(u: float, p_chat: float, p_daydream: float) -> int:
    """Pick the flavour of a distracted minute from the overlap probabilities."""
    if u < p_chat:
        return MODE_CHAT
    if u < p_chat + p_daydream:
        return MODE_DAYDREAM
    return MODE_BROWSING


def _simulate_states(
    persona: UserPersona,
    plan: DayPlan,
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the minute-level state machine.

    Returns ``(states, modes, minutes_since_break)`` as arrays of length
    ``plan.n_minutes``.
    """
    n = plan.n_minutes
    b = cfg.section("behaviour")

    in_meeting = np.zeros(n, dtype=bool)
    for s, e in plan.meetings:
        in_meeting[s:min(e, n)] = True

    forced_break = np.zeros(n, dtype=bool)
    if plan.lunch is not None:
        s, e = plan.lunch
        forced_break[s:min(e, n)] = True

    # Local clock hour of each minute, for the afternoon slump.
    start_hour = plan.start.hour + plan.start.minute / 60.0
    hours = start_hour + np.arange(n) / 60.0
    slump_lo, slump_hi = b["slump_window"]
    slump = np.where(
        (hours >= slump_lo) & (hours < slump_hi), float(b["slump_multiplier"]), 1.0
    )

    trait_mult = 1.0 + float(b["distractibility_multiplier"]) * persona.distractibility

    # Pre-drawn randomness - far cheaper than per-minute scalar draws.
    u_state = rng.random(n)
    u_break = rng.random(n)
    u_mode = rng.random(n)
    mode_pick = rng.random(n)
    min_break = int(cfg["preprocessing.min_break_minutes"])
    gamma_shape, gamma_scale = b.get("break_length_gamma", [2.0, 3.4])
    break_lengths = np.maximum(
        min_break,
        rng.gamma(shape=float(gamma_shape), scale=float(gamma_scale), size=n)
        .round()
        .astype(int),
    )

    p_read = float(b["overlap"]["reading_looks_idle"])
    p_daydream = float(b["overlap"]["daydream_in_work_app"])
    p_chat = float(b["overlap"]["personal_chat_burst"])
    base_hazard = float(b["focus_to_distract_base"])
    fatigue_rate = float(b["fatigue_per_minute"])
    fatigue_cap = float(b["fatigue_cap"])
    recover = float(b["distract_to_focus"])
    break_base = float(b["break_hazard_base"])
    break_rate = float(b["break_hazard_per_minute"])

    states = np.empty(n, dtype=object)
    modes = np.zeros(n, dtype=np.int8)
    since_break = np.zeros(n, dtype=np.int32)

    state = FOCUSED
    mode = MODE_TYPING
    minutes_since_break = 0
    break_left = 0

    for t in range(n):
        if in_meeting[t]:
            state, mode = MEETING, MODE_MEETING
            break_left = 0
        elif forced_break[t]:
            state, mode = BREAK, MODE_BREAK
            minutes_since_break = 0
            break_left = 0
        elif break_left > 0:
            state, mode = BREAK, MODE_BREAK
            minutes_since_break = 0
            break_left -= 1
        else:
            if state in (BREAK, MEETING):
                # Back at the desk: people resume focused.
                state, mode = FOCUSED, MODE_TYPING

            fatigue = min(fatigue_rate * minutes_since_break, fatigue_cap)
            # Pressure to take a break grows with time at the desk.
            p_break = break_base + break_rate * minutes_since_break
            if u_break[t] < p_break:
                state, mode = BREAK, MODE_BREAK
                break_left = int(break_lengths[t]) - 1
                minutes_since_break = 0
            elif state == FOCUSED:
                hazard = (base_hazard + fatigue) * trait_mult * slump[t]
                if u_state[t] < hazard:
                    state = DISTRACTED
                    mode = _draw_distracted_mode(mode_pick[t], p_chat, p_daydream)
                elif u_mode[t] < MODE_RESAMPLE_P:
                    mode = MODE_READING if mode_pick[t] < p_read else MODE_TYPING
            else:  # DISTRACTED
                if u_state[t] < recover:
                    state = FOCUSED
                    mode = MODE_READING if mode_pick[t] < p_read else MODE_TYPING
                elif u_mode[t] < MODE_RESAMPLE_P:
                    mode = _draw_distracted_mode(mode_pick[t], p_chat, p_daydream)

        states[t] = state
        modes[t] = mode
        since_break[t] = minutes_since_break
        if state != BREAK:
            minutes_since_break += 1

    return states, modes, since_break


# ---------------------------------------------------------------------------
# applications
# ---------------------------------------------------------------------------
def _sample_work_app(
    rng: np.random.Generator, cats: list[str], probs: np.ndarray
) -> str:
    category = str(rng.choice(cats, p=probs))
    return str(rng.choice(_APPS_BY_CATEGORY[category]))


def _assign_apps(
    persona: UserPersona,
    states: np.ndarray,
    modes: np.ndarray,
    plan: DayPlan,
    rng: np.random.Generator,
    p_ambiguous_browsing: float = 0.40,
) -> np.ndarray:
    """Assign an application per minute, persisting within a mode spell."""
    n = len(states)
    apps = np.empty(n, dtype=object)

    mix = persona.app_mix
    work_cats = [c for c, p in mix.items() if p > 0]
    work_p = np.array([mix[c] for c in work_cats], dtype=float)
    work_p /= work_p.sum()

    meeting_app = str(rng.choice(MEETING_APPS))
    current = _sample_work_app(rng, work_cats, work_p)

    for t in range(n):
        mode = modes[t]
        new_spell = t == 0 or modes[t] != modes[t - 1] or states[t] != states[t - 1]
        if mode == MODE_MEETING:
            current = meeting_app
        elif mode == MODE_BREAK:
            pass  # the window keeps whatever was last in front
        elif mode == MODE_BROWSING:
            if new_spell or rng.random() < 0.25:
                # Not all distraction announces itself. A good share of it
                # happens in the same browser the user researches in, which is
                # the "ambiguous browsing" the rule baseline cannot see.
                pool = (
                    RESEARCH_APPS
                    if rng.random() < p_ambiguous_browsing
                    else PERSONAL_APPS
                )
                current = str(rng.choice(pool))
        elif mode == MODE_CHAT:
            if new_spell or rng.random() < 0.20:
                current = str(rng.choice(CHAT_APPS))
        elif mode == MODE_DAYDREAM:
            # The defining overlap: a distracted minute on a work surface.
            if new_spell:
                current = _sample_work_app(rng, work_cats, work_p)
        else:  # focused - typing or reading
            if new_spell or rng.random() < 0.12:
                current = _sample_work_app(rng, work_cats, work_p)
        apps[t] = current
    return apps


# ---------------------------------------------------------------------------
# emissions
# ---------------------------------------------------------------------------
def _emit(
    persona: UserPersona,
    modes: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw the activity counts implied by each minute's mode."""
    n = len(modes)
    rate = persona.typing_rate
    mouse = persona.mouse_scale

    keystrokes = np.zeros(n, dtype=float)
    clicks = np.zeros(n, dtype=float)
    distance = np.zeros(n, dtype=float)
    scrolls = np.zeros(n, dtype=float)
    switches = np.zeros(n, dtype=float)
    idle = np.zeros(n, dtype=float)

    def where(mode: int) -> np.ndarray:
        return np.flatnonzero(modes == mode)

    # --- Focused, typing --------------------------------------------------
    idx = where(MODE_TYPING)
    if idx.size:
        duty = rng.beta(4.0, 3.0, idx.size)          # fraction of the minute typing
        keystrokes[idx] = rng.poisson(rate * duty)
        clicks[idx] = rng.poisson(8.0 * mouse, idx.size)
        distance[idx] = rng.gamma(3.0, 3000.0 * mouse, idx.size)
        scrolls[idx] = rng.poisson(4.0 * mouse, idx.size)
        switches[idx] = rng.poisson(0.8, idx.size)
        idle[idx] = np.clip(rng.gamma(2.0, 3.2, idx.size), 0, 40)

    # --- Focused, reading: looks idle --------------------------------------
    idx = where(MODE_READING)
    if idx.size:
        keystrokes[idx] = rng.poisson(rate * 0.035, idx.size)
        clicks[idx] = rng.poisson(1.6 * mouse, idx.size)
        distance[idx] = rng.gamma(2.0, 900.0 * mouse, idx.size)
        scrolls[idx] = rng.poisson(12.0 * mouse, idx.size)
        switches[idx] = rng.poisson(0.3, idx.size)
        idle[idx] = rng.uniform(18.0, 50.0, idx.size)

    # --- Distracted, browsing ----------------------------------------------
    idx = where(MODE_BROWSING)
    if idx.size:
        keystrokes[idx] = rng.poisson(rate * 0.075, idx.size)
        clicks[idx] = rng.poisson(6.0 * mouse, idx.size)
        distance[idx] = rng.gamma(3.0, 3400.0 * mouse, idx.size)
        scrolls[idx] = rng.poisson(24.0 * mouse, idx.size)
        switches[idx] = rng.poisson(2.9, idx.size)
        idle[idx] = np.clip(rng.gamma(2.0, 4.0, idx.size), 0, 45)

    # --- Distracted, personal chat: looks like busy typing ------------------
    idx = where(MODE_CHAT)
    if idx.size:
        duty = rng.beta(3.4, 3.4, idx.size)
        keystrokes[idx] = rng.poisson(rate * duty)
        clicks[idx] = rng.poisson(5.0 * mouse, idx.size)
        distance[idx] = rng.gamma(2.5, 2200.0 * mouse, idx.size)
        scrolls[idx] = rng.poisson(7.0 * mouse, idx.size)
        switches[idx] = rng.poisson(2.1, idx.size)
        idle[idx] = np.clip(rng.gamma(2.0, 3.0, idx.size), 0, 40)

    # --- Distracted, daydreaming inside a work app --------------------------
    # This is the overlap the whole problem turns on, so it is emitted to be
    # genuinely confusable with a focused reading spell: the same work app, the
    # same near-silent keyboard, the same aimless scrolling. Nothing in the
    # counts separates the two cleanly, which is the point.
    idx = where(MODE_DAYDREAM)
    if idx.size:
        keystrokes[idx] = rng.poisson(rate * 0.030, idx.size)
        clicks[idx] = rng.poisson(1.4 * mouse, idx.size)
        distance[idx] = rng.gamma(1.9, 820.0 * mouse, idx.size)
        scrolls[idx] = rng.poisson(9.0 * mouse, idx.size)
        switches[idx] = rng.poisson(0.4, idx.size)
        idle[idx] = rng.uniform(20.0, 48.0, idx.size)

    # --- Break --------------------------------------------------------------
    idx = where(MODE_BREAK)
    if idx.size:
        keystrokes[idx] = 0.0
        clicks[idx] = rng.poisson(0.05, idx.size)
        distance[idx] = rng.gamma(1.1, 60.0, idx.size)
        scrolls[idx] = 0.0
        switches[idx] = 0.0
        idle[idx] = rng.uniform(52.0, 60.0, idx.size)

    # --- Meeting ------------------------------------------------------------
    idx = where(MODE_MEETING)
    if idx.size:
        keystrokes[idx] = rng.poisson(rate * 0.055, idx.size)
        clicks[idx] = rng.poisson(1.2, idx.size)
        distance[idx] = rng.gamma(1.8, 800.0, idx.size)
        scrolls[idx] = rng.poisson(1.0, idx.size)
        switches[idx] = rng.poisson(0.5, idx.size)
        idle[idx] = rng.uniform(22.0, 47.0, idx.size)

    return {
        "keystrokes": np.rint(keystrokes).astype(np.int32),
        "mouse_clicks": np.rint(clicks).astype(np.int32),
        "mouse_distance_px": np.rint(distance).astype(np.int32),
        "scroll_events": np.rint(scrolls).astype(np.int32),
        "window_switches": np.rint(switches).astype(np.int32),
        "idle_seconds": np.clip(np.rint(idle), 0, 60).astype(np.int32),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def simulate_user_day(
    persona: UserPersona,
    date: pd.Timestamp,
    cfg: Config,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate one participant's workday as minute-level activity records."""
    plan = plan_day(persona, date, cfg, rng)
    states, modes, _ = _simulate_states(persona, plan, cfg, rng)
    apps = _assign_apps(
        persona, states, modes, plan, rng,
        p_ambiguous_browsing=float(
            cfg.get("behaviour.overlap.ambiguous_browsing", 0.40)
        ),
    )
    counts = _emit(persona, modes, rng)

    frame = pd.DataFrame(
        {
            "user_id": persona.user_id,
            "timestamp": plan.start
            + pd.to_timedelta(np.arange(plan.n_minutes), unit="m"),
            "active_app": apps.astype(str),
            **counts,
            "state_label": states.astype(str),
        }
    )
    return frame[list(RAW_COLUMNS)]


def _workdays(start: pd.Timestamp, n_days: int) -> list[pd.Timestamp]:
    """``n_days`` consecutive Mon-Fri dates starting at ``start``."""
    dates: list[pd.Timestamp] = []
    cursor = start
    while len(dates) < n_days:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor += pd.Timedelta(days=1)
    return dates


def _self_report(
    persona: UserPersona,
    day: pd.DataFrame,
    date: pd.Timestamp,
    rng: np.random.Generator,
    nonresponse: float,
) -> dict:
    """End-of-day self-rated productivity (1-10) and energy (1-5).

    The rating tracks the true focus ratio only loosely - self-report is noisy,
    which is why the Focus Score correlates with it modestly rather than
    strongly.
    """
    labels = day["state_label"].to_numpy()
    active = labels != BREAK
    focus_ratio = float((labels == FOCUSED).sum() / max(active.sum(), 1))

    if rng.random() < nonresponse:
        productivity: float | None = None
        energy: float | None = None
    else:
        raw = 3.2 + 6.0 * focus_ratio + rng.normal(0.0, 1.7)
        productivity = float(np.clip(round(raw), 1, 10))
        e_raw = 1.5 + 3.0 * focus_ratio - 0.004 * len(day) + rng.normal(0.0, 0.9)
        energy = float(np.clip(round(e_raw), 1, 5))

    return {
        "user_id": persona.user_id,
        "date": date.normalize().date().isoformat(),
        "productivity_rating": productivity,
        "energy_rating": energy,
    }


def simulate_cohort(
    personas: list[UserPersona],
    cfg: Config,
    rng: np.random.Generator | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the whole cohort.

    Returns the clean minute-level log and the end-of-day self-report survey.
    Corruption (timezones, duplicates, impossible values, dropouts) is applied
    separately by :func:`focustrack.data.corrupt.corrupt_logs`, so the clean
    log stays available as a reference for the preprocessing audit.
    """
    rng = rng or np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg["dataset.start_date"])
    n_days = int(cfg["dataset.n_workdays"])
    present_p = float(cfg["dataset.day_present_prob"])
    nonresponse = float(cfg["dataset.survey_nonresponse_rate"])

    dates = _workdays(start, n_days)
    day_frames: list[pd.DataFrame] = []
    survey_rows: list[dict] = []

    for i, persona in enumerate(personas):
        # A per-user stream keeps each participant reproducible on their own.
        user_rng = np.random.default_rng([cfg.seed, i])
        for date in dates:
            if user_rng.random() > present_p:
                continue  # leave, or the agent was never started
            day = simulate_user_day(persona, date, cfg, user_rng)
            day_frames.append(day)
            survey_rows.append(_self_report(persona, day, date, user_rng, nonresponse))
        if progress:
            print(f"  simulated {persona.user_id} ({persona.role})", flush=True)

    logs = pd.concat(day_frames, ignore_index=True)
    logs = logs.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    survey = pd.DataFrame(survey_rows)
    return logs, survey
