"""Assemble the results: tables, figures and a written summary.

This module produces the section-5 deliverables - the model comparison, the
reminder evaluation, the daily analytics - and writes them to ``reports/`` as
both JSON (for anything downstream) and Markdown (for people).

It also carries :data:`MILESTONE2_REFERENCE`, the numbers published in the
Milestone 2 report, and prints this run's results beside them. Where the two
diverge the divergence is shown rather than smoothed over: the pipeline
reports what it measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.engine.analytics import Correlation
from focustrack.evaluation.reminder_eval import ReminderEvaluation
from focustrack.io_utils import write_csv, write_json, write_text
from focustrack.models.focus_state import FocusStateResult

RESULTS_FILE = "results.json"
SUMMARY_FILE = "results.md"
DAILY_FILE = "daily_summary.csv"
NOTIFICATIONS_FILE = "notifications.csv"

#: Table 2 and section 5 of the Milestone 2 report, for side-by-side reading.
MILESTONE2_REFERENCE: dict[str, Any] = {
    "focus_state": {
        "Rule-based heuristic": {"accuracy": 0.883, "macro_f1": 0.747, "f1_distracted": 0.195},
        "Logistic Regression": {"accuracy": 0.956, "macro_f1": 0.938, "f1_distracted": 0.804,
                                "cv": "0.939 +/- 0.006"},
        "Random Forest": {"accuracy": 0.966, "macro_f1": 0.946, "f1_distracted": 0.831,
                          "cv": "0.949 +/- 0.003"},
        "Gradient Boosting": {"accuracy": 0.961, "macro_f1": 0.943, "f1_distracted": 0.819,
                              "cv": "0.946 +/- 0.004"},
    },
    "window_accuracy": {"single_state": 0.997, "mixed_state": 0.834},
    "focus_drop": {"roc_auc": 0.61, "baseline_roc_auc": 0.58,
                   "precision": 0.19, "baseline_precision": 0.16,
                   "recall": 0.38, "baseline_recall": 0.31},
    "engine_volume": {"break_reminders_per_day": 3.0, "focus_nudges_per_day": 1.8},
    "analytics": {"focused_hours_per_day": 5.5, "distracted_minutes_per_day": 51,
                  "breaks_per_day": 5.2, "longest_stretch_minutes": 167,
                  "focus_score_mae": 1.2},
    "correlations": {"focus_score_r": 0.23, "focus_score_p": 0.002,
                     "hours_logged_r": 0.14, "hours_logged_p": 0.06},
    "dataset": {"minute_records": 384515, "user_days": 768, "windows": 76318,
                "applications": 37, "features": 31},
}


@dataclass
class EvaluationBundle:
    """Everything one full run produced."""

    dataset: dict[str, Any]
    preprocessing: dict[str, Any]
    focus_state: FocusStateResult
    reminders: ReminderEvaluation
    drop_model: dict[str, Any]
    analytics: dict[str, Any]
    correlations: list[Correlation]
    score_agreement: dict[str, float]
    daily: pd.DataFrame
    notifications: pd.DataFrame
    figures: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "preprocessing": self.preprocessing,
            "focus_state": self.focus_state.to_dict(),
            "focus_drop": self.drop_model,
            "reminders": self.reminders.to_dict(),
            "analytics": self.analytics,
            "focus_score_agreement": self.score_agreement,
            "correlations": [c.to_dict() for c in self.correlations],
            "figures": {k: str(v) for k, v in self.figures.items()},
            "milestone2_reference": MILESTONE2_REFERENCE,
        }


# ---------------------------------------------------------------------------
# comparison against the published numbers
# ---------------------------------------------------------------------------
def comparison_table(bundle: EvaluationBundle) -> pd.DataFrame:
    """This run's headline numbers beside the ones in the Milestone 2 report."""
    ref = MILESTONE2_REFERENCE
    rows: list[dict[str, Any]] = []

    def add(metric: str, run: float | None, reported: float | None, fmt: str = "{:.3f}") -> None:
        rows.append(
            {
                "Metric": metric,
                "This run": "-" if run is None else fmt.format(run),
                "Milestone 2 report": "-" if reported is None else fmt.format(reported),
                "Delta": (
                    "-" if run is None or reported is None else f"{run - reported:+.3f}"
                ),
            }
        )

    add("minute records", bundle.dataset.get("n_minute_records"),
        ref["dataset"]["minute_records"], "{:,.0f}")
    add("user-days", bundle.dataset.get("n_user_days"), ref["dataset"]["user_days"], "{:,.0f}")
    add("5-minute windows", bundle.preprocessing.get("windows"),
        ref["dataset"]["windows"], "{:,.0f}")

    by_name = {s.name: s for s in bundle.focus_state.scores}
    for name, reported in ref["focus_state"].items():
        scores = by_name.get(name)
        if scores is None:
            continue
        add(f"{name} - accuracy", scores.accuracy, reported["accuracy"])
        add(f"{name} - macro-F1", scores.macro_f1, reported["macro_f1"])
        add(f"{name} - F1 Distracted", scores.f1_distracted, reported["f1_distracted"])

    best = by_name.get(bundle.focus_state.best_name)
    if best is not None:
        add("best model - accuracy, single-state windows",
            best.accuracy_single_state, ref["window_accuracy"]["single_state"])
        add("best model - accuracy, mixed windows",
            best.accuracy_mixed_state, ref["window_accuracy"]["mixed_state"])

    add("focus-drop ROC-AUC", bundle.reminders.model_roc_auc, ref["focus_drop"]["roc_auc"])
    add("time-since-break ROC-AUC", bundle.reminders.timer_roc_auc,
        ref["focus_drop"]["baseline_roc_auc"])
    add("alert precision - model", bundle.reminders.model_alerts.precision,
        ref["focus_drop"]["precision"])
    add("alert precision - timer", bundle.reminders.timer_alerts.precision,
        ref["focus_drop"]["baseline_precision"])
    add("alert recall - model", bundle.reminders.model_alerts.recall,
        ref["focus_drop"]["recall"])
    add("alert recall - timer", bundle.reminders.timer_alerts.recall,
        ref["focus_drop"]["baseline_recall"])

    volume = bundle.reminders.notifications_per_day
    add("break reminders per day", volume.get("break_reminders_per_day"),
        ref["engine_volume"]["break_reminders_per_day"], "{:.2f}")
    add("focus nudges per day", volume.get("focus_nudges_per_day"),
        ref["engine_volume"]["focus_nudges_per_day"], "{:.2f}")

    analytics = bundle.analytics
    add("focused hours per day", analytics.get("focused_hours_per_day"),
        ref["analytics"]["focused_hours_per_day"], "{:.2f}")
    add("distracted minutes per day", analytics.get("distracted_minutes_per_day"),
        ref["analytics"]["distracted_minutes_per_day"], "{:.1f}")
    add("breaks per day", analytics.get("breaks_per_day"),
        ref["analytics"]["breaks_per_day"], "{:.2f}")
    add("longest stretch without a break (min)", analytics.get("longest_stretch_minutes"),
        ref["analytics"]["longest_stretch_minutes"], "{:.1f}")
    add("Focus Score error, predicted vs true",
        bundle.score_agreement.get("mean_absolute_error"),
        ref["analytics"]["focus_score_mae"], "{:.2f}")

    by_correlation = {c.name: c for c in bundle.correlations}
    focus_r = by_correlation.get("Focus Score vs self-rated productivity")
    hours_r = by_correlation.get("hours logged vs self-rated productivity")
    add("r(Focus Score, self-rated productivity)",
        None if focus_r is None else focus_r.r, ref["correlations"]["focus_score_r"])
    add("r(hours logged, self-rated productivity)",
        None if hours_r is None else hours_r.r, ref["correlations"]["hours_logged_r"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def save_results(bundle: EvaluationBundle, cfg: Config) -> dict[str, Path]:
    """Write the JSON results, the Markdown summary and the daily tables."""
    reports = cfg.path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    paths = {
        "results": write_json(bundle.to_dict(), reports / RESULTS_FILE),
        "summary": write_text(markdown_summary(bundle, cfg), reports / SUMMARY_FILE),
        "daily": write_csv(bundle.daily, reports / DAILY_FILE),
    }
    if len(bundle.notifications):
        paths["notifications"] = write_csv(
            bundle.notifications, reports / NOTIFICATIONS_FILE
        )
    return paths


def markdown_summary(bundle: EvaluationBundle, cfg: Config) -> str:
    """A readable write-up of the run."""
    fs = bundle.focus_state
    lines: list[str] = [
        "# FocusTrack - results",
        "",
        "Generated by `focustrack all`. Every number below is computed by this "
        "repository from the dataset in `data/`; nothing is copied from the report.",
        "",
        "## 1. Dataset",
        "",
        f"- participants: **{bundle.dataset.get('n_users')}** "
        f"across {len(cfg['dataset.role_counts'])} roles",
        f"- minute records: **{bundle.dataset.get('n_minute_records'):,}** "
        f"over {bundle.dataset.get('n_user_days')} user-days",
        f"- applications: **{bundle.dataset.get('n_applications')}** "
        f"in 6 categories",
        f"- survey response rate: "
        f"**{bundle.dataset.get('survey_response_rate', 0):.1%}**",
        "",
        "## 2. Preprocessing",
        "",
        f"The 11-step pipeline produced **{bundle.preprocessing.get('windows'):,}** "
        f"five-minute windows described by "
        f"**{bundle.preprocessing.get('n_features')}** features.",
        "",
        "| # | step | rows in | rows out | affected |",
        "|---|------|--------:|---------:|---------:|",
    ]
    for step in bundle.preprocessing.get("steps", []):
        lines.append(
            f"| {step['step']} | {step['name']} | {step['rows_in']:,} | "
            f"{step['rows_out']:,} | {step['affected']:,} |"
        )

    lines += [
        "",
        "## 3. Focus-state classification",
        "",
        _frame_to_markdown(fs.table()),
        "",
        f"Selected on macro-F1: **{fs.best_name}**.",
        "",
        "Accuracy splits sharply by how clean a window is - almost every error "
        "is a transition window containing more than one state:",
        "",
    ]
    for scores in fs.scores:
        if scores.accuracy_single_state is None:
            continue
        lines.append(
            f"- {scores.name}: **{scores.accuracy_single_state:.3f}** on "
            f"single-state windows vs **{scores.accuracy_mixed_state:.3f}** on mixed ones"
        )

    lines += [
        "",
        "### Feature importance",
        "",
        "Per-feature permutation importance splits credit between correlated "
        "features, so the grouped view below it permutes whole signal families "
        "at once and is the better guide to what the model depends on.",
        "",
        _frame_to_markdown(fs.importances.head(10).round(4)),
        "",
        _frame_to_markdown(fs.group_importances.round(4)),
        "",
        "## 4. Focus-drop prediction and reminders",
        "",
        _frame_to_markdown(bundle.reminders.table()),
        "",
        bundle.reminders.describe(),
        "",
        f"Running the full engine produced "
        f"**{bundle.reminders.notifications_per_day.get('break_reminders_per_day', 0):.1f}** "
        f"break reminders and "
        f"**{bundle.reminders.notifications_per_day.get('focus_nudges_per_day', 0):.1f}** "
        f"focus nudges per user-day.",
        "",
        "## 5. Daily analytics",
        "",
    ]
    for key, value in bundle.analytics.items():
        lines.append(f"- {key.replace('_', ' ')}: **{value}**")

    lines += ["", "### Focus Score, predicted vs true", ""]
    for key, value in bundle.score_agreement.items():
        lines.append(f"- {key.replace('_', ' ')}: **{value}**")

    lines += ["", "### Correlation with self-report", ""]
    for correlation in bundle.correlations:
        lines.append(f"- {correlation.describe()}")

    lines += [
        "",
        "## 6. This run vs the Milestone 2 report",
        "",
        "The report's published figures come from a different simulated cohort, "
        "so exact agreement is not expected and differences are shown as measured.",
        "",
        _frame_to_markdown(comparison_table(bundle)),
        "",
        "## Figures",
        "",
    ]
    for name, path in bundle.figures.items():
        lines.append(f"- `{name}`: `{Path(path).as_posix()}`")

    lines += [
        "",
        "---",
        "",
        "*All data in this run is simulated. Scores on real volunteer data are "
        "expected to be lower.*",
        "",
    ]
    return "\n".join(lines)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a frame as a Markdown table without needing `tabulate`."""
    if frame.empty:
        return "_(no rows)_"
    columns = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(_cell(v) for v in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _cell(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.4g}"
    return str(value)
