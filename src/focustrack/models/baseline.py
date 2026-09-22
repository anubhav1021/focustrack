"""The rule-based heuristic the learned models are measured against.

This is the system a sensible engineer writes before reaching for a model:
quiet means a break, a communication app means a meeting, a social app means
distraction, everything else is work. It is a genuinely useful baseline - it
gets most windows right, because most windows are ordinary focused work.

What it cannot do is recognise distraction that does not announce itself. It
misses daydreaming inside a work app, personal chat that types as fast as
work, and ambiguous browsing, which is exactly where its F1 on ``Distracted``
collapses and why the learned models earn their place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from focustrack.config import Config
from focustrack.constants import BREAK, DISTRACTED, FOCUSED, MEETING, STATES

#: Thresholds chosen by a grid search over the **training** users only, so the
#: baseline is as strong as it can honestly be made before the learned models
#: are compared against it. Beating a deliberately weak baseline proves nothing.
DEFAULT_RULES: dict[str, float] = {
    "break_minutes_min": 3.0,       # a detected break inside the window
    "idle_frac_break": 0.80,        # or almost every minute idle
    "communication_frac": 0.85,     # almost entirely in a communication app...
    "meeting_idle_min": 25.0,       # ...and mostly listening, not typing
    "meeting_typing_max": 0.0,      # at or below this user's median typing rate
    "personal_frac": 0.30,          # a meaningful share of time on personal apps
}


class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    """A transparent four-rule heuristic, in the scikit-learn estimator shape.

    Implementing it as an estimator means it goes through exactly the same
    evaluation, cross-validation and reporting code as the learned models,
    so the comparison is like for like.
    """

    def __init__(self, rules: dict[str, float] | None = None) -> None:
        self.rules = dict(DEFAULT_RULES) if rules is None else dict(rules)

    # -- sklearn plumbing ----------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any = None) -> "RuleBasedClassifier":
        """Nothing is learned; the thresholds are chosen by hand."""
        self.classes_ = np.array(STATES)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._as_frame(X)
        r = self.rules

        prediction = np.full(len(frame), FOCUSED, dtype=object)

        # Rule 3: mostly on a personal app -> distracted.
        prediction[frame["app_frac_personal"].to_numpy() >= r["personal_frac"]] = DISTRACTED

        # Rule 2: parked in a communication app, listening rather than typing.
        meeting = (
            (frame["app_frac_communication"].to_numpy() >= r["communication_frac"])
            & (frame["idle_mean"].to_numpy() >= r["meeting_idle_min"])
            & (frame["keystrokes_mean"].to_numpy() <= r["meeting_typing_max"])
        )
        prediction[meeting] = MEETING

        # Rule 1: quiet enough to be away from the desk. Checked last so it
        # overrides the others - nobody is distracted while they are absent.
        away = (frame["break_minutes"].to_numpy() >= r["break_minutes_min"]) | (
            frame["idle_frac"].to_numpy() >= r["idle_frac_break"]
        )
        prediction[away] = BREAK
        return prediction

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """One-hot probabilities - the rules express no uncertainty."""
        prediction = self.predict(X)
        proba = np.zeros((len(prediction), len(STATES)), dtype=float)
        index = {s: i for i, s in enumerate(STATES)}
        for row, state in enumerate(prediction):
            proba[row, index[state]] = 1.0
        return proba

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError(
            "RuleBasedClassifier reads features by name; pass a DataFrame."
        )

    def describe(self) -> str:
        r = self.rules
        return "\n".join(
            [
                f"1. break_minutes >= {r['break_minutes_min']:.0f} "
                f"or idle_frac >= {r['idle_frac_break']:.2f}        -> {BREAK}",
                f"2. communication >= {r['communication_frac']:.2f} "
                f"and idle >= {r['meeting_idle_min']:.0f}s and typing <= "
                f"{r['meeting_typing_max']:.2f} -> {MEETING}",
                f"3. personal apps >= {r['personal_frac']:.2f}"
                f"                              -> {DISTRACTED}",
                f"4. otherwise"
                f"                                          -> {FOCUSED}",
            ]
        )


def build_rule_baseline(cfg: Config) -> RuleBasedClassifier:
    """Construct the baseline, allowing ``config.yaml`` to override thresholds."""
    overrides = cfg.get("models.rule_baseline", None) or {}
    rules = {**DEFAULT_RULES, **overrides}
    return RuleBasedClassifier(rules=rules)
