"""Daily analytics.

Turns the window table into the numbers a user actually sees: where the day
went, how fragmented it was, how long they went without stopping, and whether
the Focus Score says anything self-reported productivity agrees with.

The last question is the honest one to ask of a product like this. Hours
logged is the metric every time tracker already reports, so the Focus Score
has to relate to how the day *felt* better than raw hours do, or it is just a
prettier stopwatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from focustrack.config import Config
from focustrack.constants import BREAK, CATEGORIES, DISTRACTED, FOCUSED, MEETING
from focustrack.engine.focus_score import count_break_episodes, longest_run_without_break, score_day


@dataclass
class Correlation:
    """A correlation with its p-value and sample size."""

    name: str
    r: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "r": round(float(self.r), 4),
            "p_value": round(float(self.p_value), 5),
            "n": int(self.n),
            "significant_at_05": bool(self.significant),
        }

    def describe(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.name:38} r = {self.r:+.3f}, p = {self.p_value:.4f}, "
            f"n = {self.n}  ({verdict})"
        )


def daily_summary(
    windows: pd.DataFrame,
    cfg: Config,
    state_column: str = "state_label",
) -> pd.DataFrame:
    """One row per participant-day: where the time went, plus the Focus Score."""
    window_minutes = int(cfg["preprocessing.window_minutes"])
    rows: list[dict[str, Any]] = []

    for (user, date), day in windows.groupby(["user_id", "date"], sort=True, observed=True):
        day = day.sort_values("window_start")
        states = day[state_column].to_numpy()
        breaks = day["break_minutes"].to_numpy() > 0

        row: dict[str, Any] = {
            "user_id": user,
            "date": pd.Timestamp(date).normalize(),
            "tracked_minutes": len(day) * window_minutes,
            "focused_minutes": int((states == FOCUSED).sum() * window_minutes),
            "distracted_minutes": int((states == DISTRACTED).sum() * window_minutes),
            "meeting_minutes": int((states == MEETING).sum() * window_minutes),
            "break_minutes": int(day["break_minutes"].sum()),
            "n_breaks": count_break_episodes(breaks),
            "longest_stretch_minutes": longest_run_without_break(breaks, window_minutes),
            "window_switches": float(day["window_switches_sum"].sum()),
        }
        row["focused_hours"] = round(row["focused_minutes"] / 60.0, 2)
        row["tracked_hours"] = round(row["tracked_minutes"] / 60.0, 2)

        # Where the time went, by application category.
        for category in CATEGORIES:
            column = f"app_frac_{category}"
            if column in day.columns:
                row[f"minutes_{category}"] = int(
                    round(day[column].sum() * window_minutes)
                )

        score = score_day(day, cfg, state_column=state_column)
        row["focus_score"] = round(score.score, 2)
        row["focus_ratio"] = round(score.focus_ratio, 4)
        row["switch_load"] = round(score.switch_load, 4)
        row["break_balance"] = round(score.break_balance, 4)
        rows.append(row)

    return pd.DataFrame(rows)


def cohort_averages(summary: pd.DataFrame) -> dict[str, float]:
    """The average-day figures quoted in section 5 of the report."""
    if summary.empty:
        return {}
    return {
        "user_days": int(len(summary)),
        "tracked_hours_per_day": round(float(summary["tracked_hours"].mean()), 2),
        "focused_hours_per_day": round(float(summary["focused_hours"].mean()), 2),
        "distracted_minutes_per_day": round(float(summary["distracted_minutes"].mean()), 1),
        "meeting_minutes_per_day": round(float(summary["meeting_minutes"].mean()), 1),
        "break_minutes_per_day": round(float(summary["break_minutes"].mean()), 1),
        "breaks_per_day": round(float(summary["n_breaks"].mean()), 2),
        "longest_stretch_minutes": round(float(summary["longest_stretch_minutes"].mean()), 1),
        "mean_focus_score": round(float(summary["focus_score"].mean()), 2),
    }


def score_agreement(
    true_summary: pd.DataFrame,
    predicted_summary: pd.DataFrame,
) -> dict[str, float]:
    """How far the Focus Score computed from predictions drifts from the truth.

    This is the number that decides whether the score can be shown to a user at
    all: a score built on predicted states is only worth displaying if it lands
    close to the score their real day would have earned.
    """
    merged = true_summary.merge(
        predicted_summary,
        on=["user_id", "date"],
        suffixes=("_true", "_pred"),
    )
    if merged.empty:
        return {}
    difference = merged["focus_score_pred"] - merged["focus_score_true"]
    return {
        "n_user_days": int(len(merged)),
        "mean_absolute_error": round(float(difference.abs().mean()), 3),
        "mean_error": round(float(difference.mean()), 3),
        "max_absolute_error": round(float(difference.abs().max()), 3),
        "correlation": round(
            float(merged["focus_score_pred"].corr(merged["focus_score_true"])), 4
        ),
    }


def survey_correlations(
    summary: pd.DataFrame,
    survey: pd.DataFrame,
) -> list[Correlation]:
    """Correlate the Focus Score - and plain hours logged - with self-report."""
    survey = survey.copy()
    survey["date"] = pd.to_datetime(survey["date"]).dt.normalize()
    merged = summary.merge(survey, on=["user_id", "date"], how="inner")
    merged = merged.loc[merged["productivity_rating"].notna()]
    if len(merged) < 3:
        return []

    results: list[Correlation] = []
    for column, name in (
        ("focus_score", "Focus Score vs self-rated productivity"),
        ("tracked_hours", "hours logged vs self-rated productivity"),
        ("focus_ratio", "focus ratio vs self-rated productivity"),
    ):
        if column not in merged.columns:
            continue
        r, p = stats.pearsonr(merged[column], merged["productivity_rating"])
        results.append(Correlation(name=name, r=float(r), p_value=float(p), n=len(merged)))

    if "energy_rating" in merged.columns:
        energy = merged.loc[merged["energy_rating"].notna()]
        if len(energy) >= 3:
            r, p = stats.pearsonr(energy["focus_score"], energy["energy_rating"])
            results.append(
                Correlation("Focus Score vs self-rated energy", float(r), float(p), len(energy))
            )
    return results


def hourly_profile(
    windows: pd.DataFrame,
    cfg: Config,
    state_column: str = "state_label",
) -> pd.DataFrame:
    """Share of each state by hour of the day - where the slump shows up."""
    work = windows.copy()
    work["hour"] = pd.to_datetime(work["window_start"]).dt.hour
    counts = (
        work.groupby(["hour", state_column], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0).reset_index()
    shares.columns.name = None
    return shares


def per_user_summary(summary: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily summary up to one row per participant."""
    aggregated = (
        summary.groupby("user_id", observed=True)
        .agg(
            days=("date", "nunique"),
            focused_hours=("focused_hours", "mean"),
            distracted_minutes=("distracted_minutes", "mean"),
            breaks_per_day=("n_breaks", "mean"),
            longest_stretch=("longest_stretch_minutes", "mean"),
            focus_score=("focus_score", "mean"),
        )
        .reset_index()
    )
    if "user_id" in users.columns:
        aggregated = aggregated.merge(users, on="user_id", how="left")
    return aggregated.round(2)
