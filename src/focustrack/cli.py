"""The ``focustrack`` command line.

    focustrack all              run everything: data, pipeline, models, results
    focustrack generate         simulate the cohort and write the dataset
    focustrack preprocess       run the 11-step pipeline
    focustrack train            fit the classifier and the drop predictor
    focustrack evaluate         analytics, reminder comparison and figures
    focustrack agent            run the live desktop agent
    focustrack report           end-of-day self-report
    focustrack dashboard        open the Streamlit dashboard
    focustrack store            inspect the local encrypted store
    focustrack info             show the configuration and what is installed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from focustrack import __version__
from focustrack.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="focustrack",
        description="FocusTrack - tracking work hours, breaks and focus for remote workers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"FocusTrack {__version__}")
    parser.add_argument(
        "-c", "--config", type=Path, default=None,
        help="path to config.yaml (default: the one beside the project root)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only print warnings and errors",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_all = sub.add_parser("all", help="run the whole study end to end")
    run_all.add_argument(
        "--skip-generate", action="store_true",
        help="reuse the dataset already in data/raw",
    )
    run_all.add_argument(
        "--quick", action="store_true",
        help="a small, fast run (10 users, 5 days) for smoke-testing",
    )

    generate = sub.add_parser("generate", help="simulate the cohort and write the dataset")
    generate.add_argument("--users", type=int, default=None, help="override participant count")
    generate.add_argument("--days", type=int, default=None, help="override workday count")

    sub.add_parser("preprocess", help="run the 11-step preprocessing pipeline")
    sub.add_parser("train", help="fit the focus-state and focus-drop models")
    sub.add_parser("evaluate", help="analytics, reminder comparison and figures")

    agent = sub.add_parser("agent", help="run the live desktop agent")
    agent.add_argument("--user", default="local", help="identifier for this user")
    agent.add_argument(
        "--minutes", type=float, default=None,
        help="stop after this many minutes (default: run until Ctrl+C)",
    )
    agent.add_argument(
        "--console", action="store_true",
        help="print reminders instead of sending desktop notifications",
    )

    report = sub.add_parser("report", help="answer the end-of-day self-report")
    report.add_argument("--user", default="local")

    dashboard = sub.add_parser("dashboard", help="open the Streamlit dashboard")
    dashboard.add_argument("--port", type=int, default=8501)

    store = sub.add_parser("store", help="inspect the local encrypted store")
    store.add_argument(
        "--purge-before", default=None,
        help="delete activity before this date, e.g. 2026-01-01",
    )

    sub.add_parser("info", help="show configuration and installed capabilities")
    return parser


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_all(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.pipeline_runner import run_all

    if args.quick:
        cfg = cfg.with_overrides(
            **{
                "dataset.role_counts": {
                    "developer": 2, "designer": 2, "writer": 2, "analyst": 2, "support": 2
                },
                "dataset.n_workdays": 5,
                "split.n_test_users_per_role": 1,
                "split.cv_folds": 3,
                "models.focus_state.random_forest.n_estimators": 120,
                "models.focus_state.gradient_boosting.max_iter": 120,
                "models.focus_drop.max_iter": 120,
                "models.permutation_repeats": 2,
            }
        )
        print("running in --quick mode: 10 users, 5 days, smaller models\n")

    run_all(cfg, verbose=not args.quiet, skip_generate=args.skip_generate)
    return 0


def cmd_generate(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.pipeline_runner import stage_generate

    overrides: dict[str, Any] = {}
    if args.days is not None:
        overrides["dataset.n_workdays"] = args.days
    if args.users is not None:
        per_role, remainder = divmod(args.users, 5)
        counts = {r: per_role for r in ("developer", "designer", "writer", "analyst", "support")}
        counts["developer"] += remainder
        overrides["dataset.role_counts"] = counts
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    cfg.ensure_dirs()
    stage_generate(cfg, verbose=not args.quiet)
    return 0


def cmd_preprocess(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.pipeline_runner import stage_preprocess

    cfg.ensure_dirs()
    stage_preprocess(cfg, verbose=not args.quiet)
    return 0


def cmd_train(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.pipeline_runner import stage_train

    cfg.ensure_dirs()
    stage_train(cfg, verbose=not args.quiet)
    return 0


def cmd_evaluate(args: argparse.Namespace, cfg: Config) -> int:
    import json

    from focustrack.pipeline_runner import stage_evaluate, stage_train
    from focustrack.preprocessing.pipeline import AUDIT_FILE

    cfg.ensure_dirs()
    trained = stage_train(cfg, verbose=not args.quiet)

    audit_path = cfg.path("reports") / AUDIT_FILE
    preprocessing: dict[str, Any] = {}
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as fh:
            preprocessing = json.load(fh)

    manifest = cfg.path("raw") / "dataset_manifest.json"
    dataset = None
    if manifest.exists():
        from focustrack.data.generate import DatasetSummary

        with manifest.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        dataset = DatasetSummary(
            n_users=payload.get("n_users", 0),
            n_workdays=payload.get("n_workdays", 0),
            n_user_days=payload.get("n_user_days", 0),
            n_minute_records=payload.get("n_minute_records", 0),
            n_applications=payload.get("n_applications", 0),
            state_distribution=payload.get("state_distribution", {}),
            survey_response_rate=payload.get("survey_response_rate", 0.0),
            corruption=payload.get("corruption", {}),
            elapsed_seconds=payload.get("elapsed_seconds", 0.0),
        )

    stage_evaluate(
        cfg, trained, dataset=dataset, preprocessing=preprocessing, verbose=not args.quiet
    )
    return 0


def cmd_agent(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.agent.notifier import ConsoleNotifier
    from focustrack.agent.service import FocusTrackAgent

    agent = FocusTrackAgent(
        cfg,
        user_id=args.user,
        notifier=ConsoleNotifier() if args.console else None,
    )
    agent.run(duration_minutes=args.minutes, verbose=not args.quiet)
    return 0


def cmd_report(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.agent.self_report import prompt_self_report
    from focustrack.storage.db import FocusStore

    prompt_self_report(args.user, store=FocusStore.from_config(cfg))
    return 0


def cmd_dashboard(args: argparse.Namespace, cfg: Config) -> int:
    import subprocess

    app = Path(__file__).parent / "dashboard" / "app.py"
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "streamlit is not installed.\n"
            "  pip install streamlit\n"
            f"then run:  streamlit run {app}",
            file=sys.stderr,
        )
        return 1

    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port", str(args.port),
    ]
    print(f"starting the dashboard on http://localhost:{args.port} ...")
    return subprocess.call(command)


def cmd_store(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.storage.db import FocusStore

    store = FocusStore.from_config(cfg)
    if args.purge_before:
        removed = store.purge_before(args.purge_before)
        print(f"deleted {removed:,} activity rows before {args.purge_before}")
    print(f"store: {store.db_path}")
    print(store.stats().describe())
    return 0


def cmd_info(args: argparse.Namespace, cfg: Config) -> int:
    from focustrack.agent.collector import ActivityCollector
    from focustrack.constants import APP_CATALOGUE, CATEGORIES, STATES
    from focustrack.models.registry import models_available
    from focustrack.preprocessing.features import FEATURE_GROUPS, N_FEATURES
    from focustrack.storage.crypto import CRYPTO_AVAILABLE

    print(f"FocusTrack {__version__}")
    print(f"  config            {cfg.source}")
    print(f"  project root      {cfg.root}")
    print()
    print("  states            " + ", ".join(STATES))
    print(f"  applications      {len(APP_CATALOGUE)} in {len(CATEGORIES)} categories")
    print(f"  features          {N_FEATURES}")
    for group, features in FEATURE_GROUPS.items():
        print(f"    {group:28} {len(features)}")
    print()
    print("  data")
    for name in ("raw", "interim", "processed", "models", "reports"):
        path = cfg.path(name)
        files = len(list(path.glob("*"))) if path.exists() else 0
        print(f"    {name:12} {'exists' if path.exists() else 'missing':8} "
              f"{files} files  {path}")
    print()
    print(f"  models fitted     {'yes' if models_available(cfg) else 'no'}")
    print(f"  encryption        {'available' if CRYPTO_AVAILABLE else 'NOT installed'}")
    print("  agent capabilities")
    print(ActivityCollector.describe_capabilities())
    return 0


COMMANDS = {
    "all": cmd_all,
    "generate": cmd_generate,
    "preprocess": cmd_preprocess,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "agent": cmd_agent,
    "report": cmd_report,
    "dashboard": cmd_dashboard,
    "store": cmd_store,
    "info": cmd_info,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    handler = COMMANDS[args.command]
    try:
        return handler(args, cfg)
    except FileNotFoundError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
