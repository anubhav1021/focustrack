"""Evaluating the reminder engine against a fixed timer.

Comparing an adaptive reminder to a timer on raw hit-rate is unfair in both
directions: whichever fires more often will look better on recall and worse on
precision. So the comparison here **fixes the alert budget**. The timer is run
first, and the model is then allowed exactly as many alerts, spent on the
windows it considers riskiest.

Under an equal budget the question becomes the only one that matters: given
the same number of interruptions a day, which set of interruptions lands on
the moments that were actually about to go wrong?
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from focustrack.config import Config
from focustrack.models.focus_drop import DROP_LABEL


@dataclass
class AlertScores:
    """Precision and recall of one alerting strategy at a fixed budget."""

    strategy: str
    n_alerts: int
    precision: float
    recall: float
    n_drops: int
    n_opportunities: int

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["precision"] = round(float(self.precision), 4)
        out["recall"] = round(float(self.recall), 4)
        return out


@dataclass
class ReminderEvaluation:
    """The full comparison, including both ROC-AUCs and the budgeted alerts."""

    model_roc_auc: float
    timer_roc_auc: float
    model_alerts: AlertScores
    timer_alerts: AlertScores
    budget: int
    notifications_per_day: dict[str, float]
    timer_interval_minutes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "roc_auc": {
                "focus_drop_model": round(float(self.model_roc_auc), 4),
                "time_since_break_only": round(float(self.timer_roc_auc), 4),
            },
            "equal_budget_alerts": {
                "budget": self.budget,
                "recall_ceiling": round(float(self.recall_ceiling), 4),
                "timer_interval_minutes": self.timer_interval_minutes,
                "model": self.model_alerts.to_dict(),
                "timer": self.timer_alerts.to_dict(),
            },
            "engine_volume": self.notifications_per_day,
        }

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Strategy": self.timer_alerts.strategy,
                    "Alerts": self.timer_alerts.n_alerts,
                    "Precision": round(self.timer_alerts.precision, 3),
                    "Recall": round(self.timer_alerts.recall, 3),
                    "ROC-AUC": round(float(self.timer_roc_auc), 3),
                },
                {
                    "Strategy": self.model_alerts.strategy,
                    "Alerts": self.model_alerts.n_alerts,
                    "Precision": round(self.model_alerts.precision, 3),
                    "Recall": round(self.model_alerts.recall, 3),
                    "ROC-AUC": round(float(self.model_roc_auc), 3),
                },
            ]
        )

    @property
    def recall_ceiling(self) -> float:
        """The best recall any strategy could reach on this alert budget.

        With fewer alerts than there are drops, most drops cannot be caught by
        anyone, so raw recall says as much about the budget as about the model.
        """
        drops = self.model_alerts.n_drops
        return min(1.0, self.budget / drops) if drops else float("nan")

    def describe(self) -> str:
        gain_p = self.model_alerts.precision - self.timer_alerts.precision
        gain_r = self.model_alerts.recall - self.timer_alerts.recall
        verdict = (
            "the model beats the timer on both"
            if gain_p > 0 and gain_r > 0
            else "the model does not clearly beat the timer"
        )
        ceiling = self.recall_ceiling
        share = (
            self.model_alerts.recall / ceiling if ceiling and ceiling == ceiling else float("nan")
        )
        return (
            f"  at an equal budget of {self.budget} alerts, {verdict}:\n"
            f"    precision {self.timer_alerts.precision:.1%} -> "
            f"{self.model_alerts.precision:.1%}  ({gain_p:+.1%})\n"
            f"    recall    {self.timer_alerts.recall:.1%} -> "
            f"{self.model_alerts.recall:.1%}  ({gain_r:+.1%})\n"
            f"  the budget covers {self.budget} of {self.model_alerts.n_drops} drops, so no\n"
            f"  strategy could exceed {ceiling:.1%} recall; the model reaches "
            f"{share:.0%} of that ceiling."
        )


def evaluate_reminders(
    test_frame: pd.DataFrame,
    cfg: Config,
    notifications_per_day: dict[str, float] | None = None,
    probability_column: str = "drop_probability",
) -> ReminderEvaluation:
    """Compare the drop model with a fixed timer at an equal alert budget.

    ``test_frame`` must hold one row per focused window of the unseen test
    users, with the drop label, the predicted probability and the
    minutes-since-break control.
    """
    interval = int(cfg["reminders.timer_baseline_minutes"])
    truth = test_frame[DROP_LABEL].to_numpy(dtype=float)
    probability = test_frame[probability_column].to_numpy(dtype=float)
    since_break = test_frame["minutes_since_break"].to_numpy(dtype=float)

    n_drops = int(truth.sum())
    n_opportunities = len(truth)

    # --- the timer's own alerts, one per crossing of the interval ---------
    timer_mask = _timer_alert_mask(test_frame, interval)
    budget = int(timer_mask.sum())

    timer_alerts = AlertScores(
        strategy=f"fixed {interval}-minute timer",
        n_alerts=budget,
        precision=_precision(truth, timer_mask),
        recall=_recall(truth, timer_mask),
        n_drops=n_drops,
        n_opportunities=n_opportunities,
    )

    # --- the model, spending exactly the same budget ----------------------
    model_mask = _top_k_mask(probability, budget)
    model_alerts = AlertScores(
        strategy="focus-drop model",
        n_alerts=int(model_mask.sum()),
        precision=_precision(truth, model_mask),
        recall=_recall(truth, model_mask),
        n_drops=n_drops,
        n_opportunities=n_opportunities,
    )

    return ReminderEvaluation(
        model_roc_auc=_safe_auc(truth, probability),
        timer_roc_auc=_safe_auc(truth, since_break),
        model_alerts=model_alerts,
        timer_alerts=timer_alerts,
        budget=budget,
        notifications_per_day=notifications_per_day or {},
        timer_interval_minutes=interval,
    )


# ---------------------------------------------------------------------------
def _timer_alert_mask(frame: pd.DataFrame, interval: int) -> np.ndarray:
    """Where a fixed timer would fire: each time the interval elapses again."""
    mask = np.zeros(len(frame), dtype=bool)
    position = {index: i for i, index in enumerate(frame.index)}

    for _key, day in frame.groupby(["user_id", "date"], sort=False, observed=True):
        day = day.sort_values("window_start")
        fired_at = -np.inf
        for index, since in zip(day.index, day["minutes_since_break"].to_numpy(dtype=float)):
            if since >= interval and since - fired_at >= interval:
                mask[position[index]] = True
                fired_at = since
    return mask


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Alert on the ``k`` highest-scoring windows."""
    mask = np.zeros(len(scores), dtype=bool)
    if k <= 0 or len(scores) == 0:
        return mask
    k = min(k, len(scores))
    mask[np.argsort(-scores, kind="stable")[:k]] = True
    return mask


def _precision(truth: np.ndarray, alerts: np.ndarray) -> float:
    fired = int(alerts.sum())
    return float(truth[alerts].sum() / fired) if fired else 0.0


def _recall(truth: np.ndarray, alerts: np.ndarray) -> float:
    total = float(truth.sum())
    return float(truth[alerts].sum() / total) if total else 0.0


def _safe_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(truth)) < 2:
        return float("nan")
    return float(roc_auc_score(truth, scores))
