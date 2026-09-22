"""The focus-state classifier.

Predicts ``Focused`` / ``Distracted`` / ``Break`` / ``Meeting`` for every
five-minute window. Four candidates are compared - the rule baseline, logistic
regression, a random forest and gradient boosting - and the winner is chosen
on **macro-F1**, not accuracy.

That choice matters. ``Distracted`` is about a tenth of all windows, so a
model that never predicts it at all still scores near 90% accuracy while being
useless for the one thing the product exists to do. Macro-F1 gives the rare
class equal weight, and every classifier is fitted with balanced class weights
for the same reason.

Cross-validation is grouped by participant, matching the by-user test split:
a fold never contains windows from a user it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from focustrack.config import Config
from focustrack.constants import DISTRACTED, STATES
from focustrack.models.baseline import build_rule_baseline
from focustrack.models.importance import grouped_importance, per_feature_importance
from focustrack.preprocessing.features import FEATURE_COLUMNS, feature_matrix

#: The order models are reported in.
MODEL_NAMES: tuple[str, ...] = (
    "Rule-based heuristic",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
)


@dataclass
class ModelScores:
    """Test-set scores for one candidate model."""

    name: str
    accuracy: float
    macro_f1: float
    f1_distracted: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    cv_macro_f1_mean: float | None = None
    cv_macro_f1_std: float | None = None
    #: Accuracy split by whether the window contained one state or several.
    accuracy_single_state: float | None = None
    accuracy_mixed_state: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "f1_distracted": round(self.f1_distracted, 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "cv_macro_f1_mean": (
                None if self.cv_macro_f1_mean is None else round(self.cv_macro_f1_mean, 4)
            ),
            "cv_macro_f1_std": (
                None if self.cv_macro_f1_std is None else round(self.cv_macro_f1_std, 4)
            ),
            "accuracy_single_state": (
                None if self.accuracy_single_state is None
                else round(self.accuracy_single_state, 4)
            ),
            "accuracy_mixed_state": (
                None if self.accuracy_mixed_state is None
                else round(self.accuracy_mixed_state, 4)
            ),
        }

    def cv_text(self) -> str:
        if self.cv_macro_f1_mean is None:
            return "-"
        return f"{self.cv_macro_f1_mean:.3f} +/- {self.cv_macro_f1_std:.3f}"


@dataclass
class FocusStateResult:
    """Everything the training run produced."""

    scores: list[ModelScores]
    best_name: str
    best_model: Any
    confusion: np.ndarray
    labels: tuple[str, ...]
    importances: pd.DataFrame
    group_importances: pd.DataFrame
    test_predictions: pd.DataFrame

    def table(self) -> pd.DataFrame:
        """Table 2 of the report."""
        return pd.DataFrame(
            [
                {
                    "Model (unseen test users)": s.name,
                    "Accuracy": round(s.accuracy, 3),
                    "Macro-F1": round(s.macro_f1, 3),
                    "F1 Distracted": round(s.f1_distracted, 3),
                    "5-fold CV macro-F1": s.cv_text(),
                }
                for s in self.scores
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.best_name,
            "selection_metric": "macro_f1",
            "labels": list(self.labels),
            "scores": [s.to_dict() for s in self.scores],
            "confusion_matrix": self.confusion.tolist(),
            "top_features": self.importances.head(12).to_dict("records"),
            "signal_groups": self.group_importances.to_dict("records"),
        }


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def build_models(cfg: Config) -> dict[str, Any]:
    """The four candidates, configured from ``config.yaml``."""
    m = cfg.section("models")["focus_state"]
    weight = m.get("class_weight", "balanced")
    seed = cfg.seed

    lr_cfg = m["logistic_regression"]
    rf_cfg = m["random_forest"]
    gb_cfg = m["gradient_boosting"]

    return {
        "Rule-based heuristic": build_rule_baseline(cfg),
        # Scaling matters for the linear model and for nothing else here.
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(lr_cfg["C"]),
                        max_iter=int(lr_cfg["max_iter"]),
                        class_weight=weight,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=int(rf_cfg["n_estimators"]),
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            max_features=rf_cfg["max_features"],
            class_weight=weight,
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
            random_state=seed,
        ),
        # Histogram-binned boosting. Four-class boosting fits one tree per
        # class per iteration, so binning the features is what keeps the whole
        # comparison re-runnable in seconds rather than the better part of an
        # hour - the algorithm is Friedman's either way.
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=int(gb_cfg["max_iter"]),
            learning_rate=float(gb_cfg["learning_rate"]),
            max_depth=int(gb_cfg["max_depth"]),
            max_leaf_nodes=int(gb_cfg.get("max_leaf_nodes", 31)),
            early_stopping=bool(gb_cfg.get("early_stopping", False)),
            class_weight=weight,
            random_state=seed,
        ),
    }


def _fit(name: str, model: Any, X: pd.DataFrame, y: np.ndarray) -> Any:
    """Fit one model. Every estimator here balances the classes itself."""
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_focus_state(
    windows: pd.DataFrame,
    train_users: list[str],
    test_users: list[str],
    cfg: Config,
    verbose: bool = True,
) -> FocusStateResult:
    """Fit and compare the four candidates, then pick the best on macro-F1."""
    labelled = windows.loc[windows["state_label"].notna()].reset_index(drop=True)
    train = labelled.loc[labelled["user_id"].isin(train_users)].reset_index(drop=True)
    test = labelled.loc[labelled["user_id"].isin(test_users)].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("train or test split is empty - check the user split")

    X_train, y_train = feature_matrix(train), train["state_label"].to_numpy()
    X_test, y_test = feature_matrix(test), test["state_label"].to_numpy()
    groups = train["user_id"].to_numpy()

    models = build_models(cfg)
    scores: list[ModelScores] = []
    fitted: dict[str, Any] = {}

    for name in MODEL_NAMES:
        model = models[name]
        if verbose:
            print(f"    fitting {name} ...", flush=True)
        _fit(name, model, X_train, y_train)
        fitted[name] = model

        predicted = model.predict(X_test)
        per_class = f1_score(y_test, predicted, labels=list(STATES), average=None, zero_division=0)
        entry = ModelScores(
            name=name,
            accuracy=float(accuracy_score(y_test, predicted)),
            macro_f1=float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            f1_distracted=float(
                f1_score(y_test, predicted, labels=[DISTRACTED], average="macro", zero_division=0)
            ),
            per_class_f1={s: float(v) for s, v in zip(STATES, per_class)},
        )

        # Where the errors live: clean windows versus transition windows.
        if "n_distinct_states" in test.columns:
            single = test["n_distinct_states"].to_numpy() <= 1
            if single.any():
                entry.accuracy_single_state = float(
                    accuracy_score(y_test[single], predicted[single])
                )
            if (~single).any():
                entry.accuracy_mixed_state = float(
                    accuracy_score(y_test[~single], predicted[~single])
                )

        # The rule baseline learns nothing, so cross-validating it is noise.
        if name != "Rule-based heuristic":
            mean, std = cross_validate_grouped(name, cfg, X_train, y_train, groups)
            entry.cv_macro_f1_mean, entry.cv_macro_f1_std = mean, std

        scores.append(entry)

    best = max(scores, key=lambda s: s.macro_f1)
    best_model = fitted[best.name]
    if verbose:
        print(f"    best on macro-F1: {best.name} ({best.macro_f1:.3f})", flush=True)

    predicted = best_model.predict(X_test)
    confusion = confusion_matrix(y_test, predicted, labels=list(STATES))
    if verbose:
        print("    scoring permutation importance ...", flush=True)
    importances = compute_importances(best_model, X_test, y_test, cfg)
    group_importances = grouped_importance(best_model, X_test, y_test, cfg)

    test_predictions = test.loc[
        :, [c for c in ("user_id", "date", "window_start", "state_label", "n_distinct_states")
            if c in test.columns]
    ].copy()
    test_predictions["predicted_state"] = predicted
    proba = _safe_proba(best_model, X_test)
    if proba is not None:
        for i, state in enumerate(getattr(best_model, "classes_", STATES)):
            test_predictions[f"p_{state}"] = proba[:, i]

    return FocusStateResult(
        scores=scores,
        best_name=best.name,
        best_model=best_model,
        confusion=confusion,
        labels=STATES,
        importances=importances,
        group_importances=group_importances,
        test_predictions=test_predictions,
    )


def cross_validate_grouped(
    name: str,
    cfg: Config,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float]:
    """Grouped 5-fold CV macro-F1, so no fold shares a user with its training part."""
    n_splits = int(cfg["split.cv_folds"])
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))

    fold_scores: list[float] = []
    for train_idx, valid_idx in splitter.split(X, y, groups=groups):
        model = build_models(cfg)[name]
        _fit(name, model, X.iloc[train_idx], y[train_idx])
        predicted = model.predict(X.iloc[valid_idx])
        fold_scores.append(
            float(f1_score(y[valid_idx], predicted, average="macro", zero_division=0))
        )
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def compute_importances(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: Config,
    n_repeats: int | None = None,
) -> pd.DataFrame:
    """Permutation importance on the test set, scored by macro-F1.

    Permutation is used rather than a tree's built-in impurity importance
    because it measures what the model actually loses without a feature, on
    users it has never seen, in the metric the model was selected on.

    Correlated features split the credit between them here; see
    :func:`focustrack.models.importance.grouped_importance` for the companion
    view that permutes whole signal families.
    """
    return per_feature_importance(
        model, X, y, cfg, scoring="f1_macro", n_repeats=n_repeats
    )


def _safe_proba(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    try:
        return model.predict_proba(X)
    except (AttributeError, NotImplementedError):
        return None


def classification_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true, y_pred, labels=list(STATES), zero_division=0, digits=3
    )
