"""The end-of-day self-report.

Two questions, once a day. They are what turn the agent's counts into labelled
data for retraining, and they are the only place the user is asked to type
anything at all.

Deliberately short: a long form gets skipped, and a skipped form is worse than
a two-question one. Non-response is recorded as missing rather than as a
neutral score, because "did not answer" and "felt average" are different
things and imputing one as the other would bias every model downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type, datetime
from typing import Any

from focustrack.constants import BREAK, DISTRACTED, FOCUSED, MEETING, STATES
from focustrack.storage.db import FocusStore

PRODUCTIVITY_RANGE = (1, 10)
ENERGY_RANGE = (1, 5)


@dataclass
class SelfReport:
    """One day's answers."""

    user_id: str
    date: date_type
    productivity_rating: float | None
    energy_rating: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "date": self.date.isoformat(),
            "productivity_rating": self.productivity_rating,
            "energy_rating": self.energy_rating,
        }


def prompt_self_report(
    user_id: str,
    day: date_type | None = None,
    store: FocusStore | None = None,
    input_fn: Any = input,
) -> SelfReport:
    """Ask the two questions and store the answers.

    An empty answer is accepted and recorded as missing.
    """
    day = day or datetime.now().date()
    print(f"\nFocusTrack - end of day, {day:%A %d %B}")
    print("Two quick questions. Press Enter to skip either one.\n")

    productivity = _ask_number(
        "  How productive did today feel? (1-10)  ", PRODUCTIVITY_RANGE, input_fn
    )
    energy = _ask_number(
        "  How is your energy right now? (1-5)    ", ENERGY_RANGE, input_fn
    )

    report = SelfReport(user_id, day, productivity, energy)
    if store is not None:
        store.record_self_report(user_id, day, productivity, energy)
        print("\n  Saved. Thanks - this is what keeps the model honest.\n")
    return report


def _ask_number(
    prompt: str, bounds: tuple[int, int], input_fn: Any
) -> float | None:
    low, high = bounds
    for _attempt in range(3):
        try:
            raw = input_fn(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        text = (raw or "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            print(f"    please enter a number between {low} and {high}, or Enter to skip")
            continue
        if low <= value <= high:
            return value
        print(f"    that is outside {low}-{high} - try again, or Enter to skip")
    return None


# ---------------------------------------------------------------------------
# labelling recent windows
# ---------------------------------------------------------------------------
STATE_PROMPT = "\n".join(
    f"    {i + 1}. {state}" for i, state in enumerate(STATES)
)


def prompt_window_label(window_start: Any, input_fn: Any = input) -> str | None:
    """Ask what the user was actually doing in one window.

    Used sparingly - a handful of windows a day - to build real labels for
    retraining without turning the product into a survey.
    """
    print(f"\n  What were you doing around {window_start:%H:%M}?")
    print(STATE_PROMPT)
    try:
        raw = input_fn("    choice (Enter to skip): ")
    except (EOFError, KeyboardInterrupt):
        return None
    text = (raw or "").strip()
    if not text:
        return None
    if text.isdigit() and 1 <= int(text) <= len(STATES):
        return STATES[int(text) - 1]
    matches = [s for s in STATES if s.lower().startswith(text.lower())]
    return matches[0] if len(matches) == 1 else None
