"""The focus-drop predictor.

Given a user who is focused **right now**, estimate the probability that they
will be distracted for five or more of the next fifteen minutes. That
probability is what the reminder engine consults before deciding to interrupt
someone, so it has to be calibrated as well as discriminative - a threshold of
0.45 only means something if the numbers behind it do.

The honest control is *time since the last break on its own*. A fixed timer
already knows that number, so any model has to beat it to be worth shipping,
and both are reported side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from focustrack.config import Config
from focustrack.constants import DISTRACTED, FOCUSED
from focustrack.preprocessing.features import FEATURE_COLUMNS, feature_matrix

DROP_LABEL = "focus_drop"


@dataclass
class FocusDropResult:
    """Scores for the drop predictor and its time-since-break control."""

    model: Any
    roc_auc: float
    baseline_roc_auc: float
    average_precision: float
    baseline_average_precision: float
    brier: float
    positive_rate: float
    n_train: int
    n_test: int
    test_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    importances: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_minutes": self.horizon_minutes,
            "distracted_minutes_threshold": self.threshold_minutes,
            "roc_auc": round(self.roc_auc, 4),
            "baseline_roc_auc_time_since_break": round(self.baseline_roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "baseline_average_precision": round(self.baseline_average_precision, 4),
            "brier_score": round(self.brier, 4),
            "positive_rate": round(self.positive_rate, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "calibrated_threshold": round(self.calibrated_threshold, 4),
            "top_features": self.importances.head(10).to_dict("records"),
        }

    horizon_minutes: int = 15
    threshold_minutes: int = 5
    #: Threshold that would fire on the configured share of training windows.
    calibrated_threshold: float = 0.0


# ---------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------
def build_drop_labels(
    minutes: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Attach the look-ahead label to every window.

    For each window the label counts *minutes* labelled ``Distracted`` in the
    fifteen minutes that follow the window, not windows - five scattered
    distracted minutes are a real drop even when no single window is majority
    distracted.

    Windows too close to the end of a day have no observable future and are
    marked with a missing label rather than a zero, which would quietly teach
    the model that days end calmly.
    """
    horizon = int(cfg["models.focus_drop.horizon_minutes"])
    threshold = int(cfg["models.focus_drop.distracted_minutes_threshold"])
    window_minutes = int(cfg["preprocessing.window_minutes"])

    obs = minutes.loc[minutes["has_label"], ["user_id", "timestamp", "state_label"]].copy()
    obs["is_distracted"] = (obs["state_label"] == DISTRACTED).astype(np.int8)

    out = windows.copy()
    out["window_start"] = pd.to_datetime(out["window_start"])

    labels = np.full(len(out), np.nan)
    counts = np.full(len(out), np.nan)

    for user, index in out.groupby("user_id", sort=False, observed=True).indices.items():
        user_obs = obs.loc[obs["user_id"] == user]
        if user_obs.empty:
            continue
        # A minute-indexed series of the distraction flag, so the look-ahead is
        # a slice rather than a scan.
        series = (
            user_obs.set_index("timestamp")["is_distracted"]
            .sort_index()
            .astype(float)
        )
        stamps = series.index.to_numpy()
        values = series.to_numpy()

        idx = np.sort(index)
        starts = out["window_start"].to_numpy()[idx]
        # The future begins when the window ends.
        lookahead_start = starts + np.timedelta64(window_minutes, "m")
        lookahead_end = lookahead_start + np.timedelta64(horizon, "m")

        left = np.searchsorted(stamps, lookahead_start, side="left")
        right = np.searchsorted(stamps, lookahead_end, side="left")
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        observed = right - left
        distracted = cumulative[right] - cumulative[left]

        # Require most of the horizon to be observed before trusting a label.
        enough = observed >= horizon * 0.6
        counts[idx] = np.where(enough, distracted, np.nan)
        labels[idx] = np.where(enough, (distracted >= threshold).astype(float), np.nan)

    out["distracted_minutes_ahead"] = counts
    out[DROP_LABEL] = labels
    return out


def focused_windows(windows: pd.DataFrame, use_predicted: bool = False) -> pd.DataFrame:
    """Restrict to windows where the user is focused and a label is available.

    The question the model answers is conditional - *given that you are focused
    now* - so windows in any other state are not part of it.
    """
    state_col = "predicted_state" if use_predicted else "state_label"
    mask = (windows[state_col] == FOCUSED) & windows[DROP_LABEL].notna()
    return windows.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_focus_drop(
    windows: pd.DataFrame,
    train_users: list[str],
    test_users: list[str],
    cfg: Config,
    verbose: bool = True,
) -> FocusDropResult:
    """Fit the gradient-boosting drop predictor and score it against the control."""
    cfg_drop = cfg.section("models")["focus_drop"]
    pool = focused_windows(windows)
    train = pool.loc[pool["user_id"].isin(train_users)].reset_index(drop=True)
    test = pool.loc[pool["user_id"].isin(test_users)].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("no focused windows available for the drop predictor")

    X_train, y_train = feature_matrix(train), train[DROP_LABEL].to_numpy()
    X_test, y_test = feature_matrix(test), test[DROP_LABEL].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=int(cfg_drop["max_iter"]),
        learning_rate=float(cfg_drop["learning_rate"]),
        max_depth=int(cfg_drop["max_depth"]),
        max_leaf_nodes=int(cfg_drop.get("max_leaf_nodes", 31)),
        early_stopping=bool(cfg_drop.get("early_stopping", False)),
        random_state=cfg.seed,
    )
    if verbose:
        print(
            f"    fitting focus-drop predictor on {len(train):,} focused windows "
            f"({y_train.mean():.1%} positive) ...",
            flush=True,
        )
    model.fit(X_train, y_train)

    # Calibrate on the training distribution - using the test set to pick an
    # operating point would leak exactly the information being measured.
    train_probability = model.predict_proba(X_train)[:, 1]
    percentile = float(cfg.get("reminders.drop_threshold_percentile", 0.92))
    calibrated = float(np.quantile(train_probability, percentile))

    probability = model.predict_proba(X_test)[:, 1]
    # The control a fixed timer already implements: the longer since a break,
    # the likelier a drop.
    control = test["minutes_since_break"].to_numpy(dtype=float)

    result = FocusDropResult(
        model=model,
        roc_auc=_safe_auc(y_test, probability),
        baseline_roc_auc=_safe_auc(y_test, control),
        average_precision=_safe_ap(y_test, probability),
        baseline_average_precision=_safe_ap(y_test, control),
        brier=float(brier_score_loss(y_test, probability)),
        positive_rate=float(np.mean(y_test)),
        n_train=len(train),
        n_test=len(test),
        horizon_minutes=int(cfg_drop["horizon_minutes"]),
        threshold_minutes=int(cfg_drop["distracted_minutes_threshold"]),
        calibrated_threshold=calibrated,
    )

    frame = test.loc[
        :, [c for c in ("user_id", "date", "window_start", "minutes_since_break", DROP_LABEL)
            if c in test.columns]
    ].copy()
    frame["drop_probability"] = probability
    result.test_frame = frame

    # Histogram boosting exposes no impurity importances, and permutation
    # importance is the more honest measure anyway: it is scored on held-out
    # users, in the metric the model is judged by.
    permuted = permutation_importance(
        model, X_test, y_test,
        n_repeats=int(cfg.get("models.permutation_repeats", 5)),
        random_state=cfg.seed, scoring="roc_auc", n_jobs=1,
    )
    result.importances = (
        pd.DataFrame(
            {
                "feature": list(X_train.columns),
                "importance": permuted.importances_mean,
                "std": permuted.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return result


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def predict_drop_probability(model: Any, windows: pd.DataFrame) -> np.ndarray:
    """Score any window frame with a fitted drop predictor."""
    return model.predict_proba(feature_matrix(windows))[:, 1]
