"""Evaluation: metrics, the reminder comparison, figures and the write-up."""

from focustrack.evaluation.figures import (
    figure_architecture,
    figure_daily_rhythm,
    figure_example_day,
    figure_model_diagnostics,
    figure_pipeline,
)
from focustrack.evaluation.reminder_eval import (
    AlertScores,
    ReminderEvaluation,
    evaluate_reminders,
)
from focustrack.evaluation.report import (
    MILESTONE2_REFERENCE,
    EvaluationBundle,
    comparison_table,
    markdown_summary,
    save_results,
)

__all__ = [
    "AlertScores",
    "EvaluationBundle",
    "MILESTONE2_REFERENCE",
    "ReminderEvaluation",
    "comparison_table",
    "evaluate_reminders",
    "figure_architecture",
    "figure_daily_rhythm",
    "figure_example_day",
    "figure_model_diagnostics",
    "figure_pipeline",
    "markdown_summary",
    "save_results",
]
