"""Permutation importance, per feature and per signal family.

Per-feature permutation importance has a well-known blind spot: when two
features carry the same signal, permuting either one alone barely hurts,
because the other still tells the model what it needs. The feature set here is
full of such pairs - ``keystrokes_mean`` and ``keystrokes_max``,
``idle_mean`` and ``idle_max``, and every ``prev_`` / ``hist15_`` echo of a
current-window signal.

Read alone, the per-feature chart therefore *understates* exactly the signals
the system depends on most. So both are computed: the per-feature ranking, and
a grouped ranking that permutes a whole family at once and shows what the
model would lose if that signal disappeared entirely.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from focustrack.config import Config

#: Families of features that carry the same underlying signal.
SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "typing rate": (
        "keystrokes_mean",
        "keystrokes_max",
        "prev_keystrokes_mean",
        "hist15_keystrokes_mean",
    ),
    "idle signal": (
        "idle_mean",
        "idle_max",
        "idle_frac",
        "prev_idle_mean",
        "hist15_idle_frac",
    ),
    "mouse and scroll": (
        "mouse_clicks_mean",
        "mouse_distance_mean",
        "scroll_events_mean",
    ),
    "switching load": (
        "window_switches_sum",
        "app_switch_rate",
        "n_unique_apps",
        "prev_switch_sum",
        "activity_index_std",
    ),
    "application mix": (
        "app_frac_development",
        "app_frac_creative",
        "app_frac_docs_analysis",
        "app_frac_research",
        "app_frac_communication",
        "app_frac_personal",
        "hist15_personal_frac",
    ),
    "break signals": (
        "minutes_since_break",
        "break_minutes",
    ),
    "time of day": (
        "hour_sin",
        "hour_cos",
        "minutes_since_day_start",
        "day_progress",
        "is_afternoon_slump",
    ),
}


def per_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: Config,
    scoring: str = "f1_macro",
    n_repeats: int | None = None,
) -> pd.DataFrame:
    """Standard permutation importance, one feature at a time."""
    repeats = int(
        n_repeats if n_repeats is not None else cfg.get("models.permutation_repeats", 5)
    )
    result = permutation_importance(
        model, X, y, n_repeats=repeats, random_state=cfg.seed, scoring=scoring, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def grouped_importance(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: Config,
    groups: dict[str, Sequence[str]] | None = None,
    n_repeats: int | None = None,
) -> pd.DataFrame:
    """Permute each signal family as a unit and measure the macro-F1 lost.

    Shuffling every member of a family together removes the signal outright,
    so the drop is what the model genuinely depends on rather than what one
    column happens to hold uniquely.
    """
    groups = groups or SIGNAL_GROUPS
    repeats = int(
        n_repeats if n_repeats is not None else cfg.get("models.permutation_repeats", 5)
    )
    rng = np.random.default_rng(cfg.seed)

    baseline = f1_score(y, model.predict(X), average="macro", zero_division=0)

    rows: list[dict[str, Any]] = []
    for name, features in groups.items():
        present = [f for f in features if f in X.columns]
        if not present:
            continue
        drops: list[float] = []
        for _ in range(repeats):
            shuffled = X.copy()
            # One permutation order for the whole family, so the columns stay
            # consistent with each other and only their link to the label breaks.
            order = rng.permutation(len(shuffled))
            for feature in present:
                shuffled[feature] = X[feature].to_numpy()[order]
            score = f1_score(
                y, model.predict(shuffled), average="macro", zero_division=0
            )
            drops.append(baseline - score)
        rows.append(
            {
                "group": name,
                "n_features": len(present),
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
