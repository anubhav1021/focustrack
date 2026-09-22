"""The end-to-end run: data -> pipeline -> models -> engine -> results.

One function per stage, plus :func:`run_all` which chains them. The CLI is a
thin wrapper over this module, so the whole study is reproducible from a
script or a notebook without going through argument parsing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from focustrack.config import Config
from focustrack.data.generate import (
    DatasetSummary,
    generate_dataset,
    load_raw,
    load_survey,
    load_users,
)
from focustrack.engine.analytics import (
    cohort_averages,
    daily_summary,
    hourly_profile,
    score_agreement,
    survey_correlations,
)
from focustrack.engine.reminders import (
    notifications_per_day,
    resolve_drop_threshold,
    run_engine,
)
from focustrack.evaluation.figures import (
    figure_architecture,
    figure_daily_rhythm,
    figure_example_day,
    figure_model_diagnostics,
    figure_pipeline,
)
from focustrack.evaluation.report import EvaluationBundle, save_results
from focustrack.evaluation.reminder_eval import evaluate_reminders
from focustrack.models.focus_drop import DROP_LABEL, build_drop_labels, train_focus_drop
from focustrack.models.focus_state import FocusStateResult, train_focus_state
from focustrack.models.registry import load_bundle, save_bundle
from focustrack.preprocessing.features import feature_matrix
from focustrack.io_utils import write_parquet
from focustrack.preprocessing.pipeline import (
    WINDOWS_FILE,
    PipelineResult,
    load_minutes,
    load_windows,
    run_pipeline,
    save_result,
)
from focustrack.preprocessing.split import UserSplit, load_split, make_split, save_split


def _banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 58 - len(title)), flush=True)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_generate(cfg: Config, verbose: bool = True) -> DatasetSummary:
    """Simulate the cohort and write the dataset files."""
    if verbose:
        _banner("1/5  generating the dataset")
    started = time.perf_counter()
    summary = generate_dataset(cfg)
    if verbose:
        print(summary.describe())
        print(f"  elapsed           {time.perf_counter() - started:.1f}s")
    return summary


def stage_preprocess(cfg: Config, verbose: bool = True) -> tuple[PipelineResult, UserSplit]:
    """Run the 11-step pipeline and make the by-user split."""
    if verbose:
        _banner("2/5  preprocessing")
    started = time.perf_counter()
    raw = load_raw(cfg)
    result = run_pipeline(raw, cfg, verbose=False)
    save_result(result, cfg)

    split = make_split(load_users(cfg), cfg)
    save_split(split, cfg)

    if verbose:
        print(result.audit.describe())
        print()
        print(split.describe())
        print(f"  elapsed           {time.perf_counter() - started:.1f}s")
    return result, split


def stage_train(
    cfg: Config,
    windows: pd.DataFrame | None = None,
    minutes: pd.DataFrame | None = None,
    split: UserSplit | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit the focus-state classifier and the focus-drop predictor."""
    if verbose:
        _banner("3/5  training models")
    started = time.perf_counter()

    windows = load_windows(cfg) if windows is None else windows
    minutes = load_minutes(cfg) if minutes is None else minutes
    split = load_split(cfg) if split is None else split

    state_result = train_focus_state(
        windows, split.train_users, split.test_users, cfg, verbose=verbose
    )

    labelled = build_drop_labels(minutes, windows, cfg)
    drop_result = train_focus_drop(
        labelled, split.train_users, split.test_users, cfg, verbose=verbose
    )

    save_bundle(
        state_result.best_model,
        drop_result.model,
        cfg,
        metadata={
            "focus_state_model": state_result.best_name,
            "focus_state_macro_f1": round(
                max(s.macro_f1 for s in state_result.scores), 4
            ),
            "focus_drop_roc_auc": round(drop_result.roc_auc, 4),
            "drop_threshold_calibrated": round(drop_result.calibrated_threshold, 4),
            "train_users": split.train_users,
            "test_users": split.test_users,
        },
    )

    if verbose:
        print()
        print(state_result.table().to_string(index=False))
        print(
            f"\n  focus-drop ROC-AUC  {drop_result.roc_auc:.3f} "
            f"(time-since-break alone: {drop_result.baseline_roc_auc:.3f})"
        )
        print(f"  elapsed             {time.perf_counter() - started:.1f}s")

    return {
        "state_result": state_result,
        "drop_result": drop_result,
        "labelled_windows": labelled,
        "split": split,
        "windows": windows,
        "minutes": minutes,
    }


def stage_evaluate(
    cfg: Config,
    trained: dict[str, Any],
    dataset: DatasetSummary | None = None,
    preprocessing: dict[str, Any] | None = None,
    verbose: bool = True,
) -> EvaluationBundle:
    """Run the engine, compute analytics, draw the figures and write results."""
    if verbose:
        _banner("4/5  evaluation and analytics")
    started = time.perf_counter()

    state_result: FocusStateResult = trained["state_result"]
    drop_result = trained["drop_result"]
    split: UserSplit = trained["split"]
    labelled: pd.DataFrame = trained["labelled_windows"]

    # --- score every window, so the engine can be replayed end to end -----
    scored = labelled.copy()
    bundle = load_bundle(cfg)
    features = feature_matrix(scored)
    scored["predicted_state"] = bundle.focus_state.predict(features)
    scored["drop_probability"] = bundle.focus_drop.predict_proba(features)[:, 1]

    # Persist the scored table. It is what the dashboard reads, and shipping
    # predictions rather than a 100 MB forest is what makes the dashboard
    # deployable.
    write_parquet(
        scored.assign(state_label=scored["state_label"].astype("string")),
        cfg.path("processed") / WINDOWS_FILE,
    )

    test = scored.loc[scored["user_id"].isin(split.test_users)].reset_index(drop=True)

    # --- reminders on unseen users ----------------------------------------
    drop_threshold = resolve_drop_threshold(cfg, drop_result.calibrated_threshold)
    notifications = run_engine(
        test, cfg,
        state_column="predicted_state",
        probability_column="drop_probability",
        drop_threshold=drop_threshold,
    )
    n_user_days = int(test.groupby(["user_id", "date"], observed=True).ngroups)
    volume = notifications_per_day(notifications, n_user_days)

    drop_eval_frame = test.loc[
        (test["state_label"] == "Focused") & test[DROP_LABEL].notna()
    ].reset_index(drop=True)

    # How often the model's probability actually clears the engine's threshold.
    # A threshold no window reaches makes the adaptive path dead code, so this
    # belongs in the results rather than being left to be discovered later.
    volume["share_of_windows_above_threshold"] = round(
        float((drop_eval_frame["drop_probability"] >= drop_threshold).mean()), 4
    )
    volume["drop_threshold_mode"] = str(
        cfg.get("reminders.drop_threshold_mode", "absolute")
    )
    volume["drop_probability_threshold"] = round(drop_threshold, 4)
    volume["drop_threshold_calibrated"] = round(drop_result.calibrated_threshold, 4)
    volume["drop_probability_p95"] = round(
        float(drop_eval_frame["drop_probability"].quantile(0.95)), 4
    )

    reminder_eval = evaluate_reminders(drop_eval_frame, cfg, notifications_per_day=volume)

    # --- analytics ---------------------------------------------------------
    true_daily = daily_summary(test, cfg, state_column="state_label")
    predicted_daily = daily_summary(test, cfg, state_column="predicted_state")
    agreement = score_agreement(true_daily, predicted_daily)

    survey = load_survey(cfg)
    correlations = survey_correlations(true_daily, survey)
    averages = cohort_averages(true_daily)
    profile = hourly_profile(test, cfg, state_column="state_label")

    # --- figures -----------------------------------------------------------
    if verbose:
        print("  drawing figures ...", flush=True)
    preprocessing = preprocessing or {}
    figures = {
        "figure1_preprocessing": figure_pipeline(preprocessing.get("steps", []), cfg),
        "figure2_architecture": figure_architecture(cfg),
        "figure3_diagnostics": figure_model_diagnostics(
            state_result.confusion, state_result.importances,
            state_result.best_name, cfg, state_result.group_importances,
        ),
        "figure5_daily_rhythm": figure_daily_rhythm(profile, true_daily, cfg),
    }

    example = _pick_example_day(test, notifications)
    if example is not None:
        day, day_notifications, suffix = example
        figures["figure4_example_day"] = figure_example_day(
            day, day_notifications, cfg, title_suffix=suffix
        )

    evaluation = EvaluationBundle(
        dataset=dataset.to_dict() if dataset else {},
        preprocessing=preprocessing,
        focus_state=state_result,
        reminders=reminder_eval,
        drop_model=drop_result.to_dict(),
        analytics=averages,
        correlations=correlations,
        score_agreement=agreement,
        daily=predicted_daily.merge(
            true_daily[["user_id", "date", "focus_score"]],
            on=["user_id", "date"], suffixes=("", "_true"),
        ),
        notifications=notifications,
        figures=figures,
    )
    paths = save_results(evaluation, cfg)

    if verbose:
        print()
        print(reminder_eval.table().to_string(index=False))
        print()
        print(reminder_eval.describe())
        print()
        print("  average day across the unseen test users:")
        for key, value in averages.items():
            print(f"    {key:32} {value}")
        print()
        for correlation in correlations:
            print(f"    {correlation.describe()}")
        print()
        print(f"  results written to {paths['summary']}")
        print(f"  elapsed            {time.perf_counter() - started:.1f}s")

    return evaluation


def _pick_example_day(
    test: pd.DataFrame, notifications: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str] | None:
    """Choose an illustrative day: the one where the most reminders fired."""
    if test.empty:
        return None
    if len(notifications):
        counts = notifications.groupby(["user_id", "date"], observed=True).size()
        user, date = counts.idxmax()
    else:
        user, date = next(iter(test.groupby(["user_id", "date"], observed=True).groups))

    day = test.loc[(test["user_id"] == user) & (test["date"] == date)].copy()
    day_notifications = (
        notifications.loc[
            (notifications["user_id"] == user) & (notifications["date"] == date)
        ]
        if len(notifications)
        else pd.DataFrame()
    )
    suffix = f" - {user}, {pd.Timestamp(date):%A %d %B}"
    return day, day_notifications, suffix


# ---------------------------------------------------------------------------
def run_all(cfg: Config, verbose: bool = True, skip_generate: bool = False) -> EvaluationBundle:
    """Run every stage, from simulating the cohort to writing the results."""
    started = time.perf_counter()
    cfg.ensure_dirs()

    dataset = None if skip_generate else stage_generate(cfg, verbose)
    result, split = stage_preprocess(cfg, verbose)
    trained = stage_train(
        cfg, windows=result.windows, minutes=result.minutes, split=split, verbose=verbose
    )
    evaluation = stage_evaluate(
        cfg, trained, dataset=dataset, preprocessing=result.summary(), verbose=verbose
    )

    if verbose:
        _banner("5/5  done")
        print(f"  total elapsed     {time.perf_counter() - started:.1f}s")
        print(f"  figures           {cfg.path('figures')}")
        print(f"  results           {cfg.path('reports') / 'results.md'}")
        print("\n  next: focustrack dashboard\n")
    return evaluation
