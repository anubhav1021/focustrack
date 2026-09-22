"""Layer 5 - the FocusTrack dashboard.

    focustrack dashboard
    # or:  streamlit run src/focustrack/dashboard/app.py

Five tabs, each answering one question:

  Today      where did the day go, and what is the Focus Score?
  Trends     is this getting better or worse over time?
  Reminders  what did the engine say, and was it right to?
  Model      how well does the classifier actually work?
  Data       what is recorded, and what is deliberately not?
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run path/to/app.py` without the package installed.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from focustrack import __version__
from focustrack.config import load_config
from focustrack.constants import BREAK_REMINDER, CATEGORIES, STATES
from focustrack.dashboard.data import (
    DashboardData,
    category_minutes,
    load_dashboard_data,
    state_minutes,
)
from focustrack.dashboard.theme import (
    ACCENT,
    INK,
    MUTED,
    STATE_COLOURS,
    WARN,
    app_mix_chart,
    hourly_chart,
    score_band,
    score_trend_chart,
    state_split_chart,
    timeline_chart,
)

st.set_page_config(
    page_title="FocusTrack",
    page_icon="focus",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="loading your activity ...")
def _load():
    cfg = load_config()
    data = load_dashboard_data(cfg)
    return data, cfg


def main() -> None:
    st.title("FocusTrack")
    st.caption(
        "Work hours, breaks and focus for remote workers - with adaptive "
        "reminders and a daily Focus Score."
    )

    try:
        data, cfg = _load()
    except FileNotFoundError:
        _show_setup_help()
        return

    if data.windows.empty:
        _show_setup_help()
        return

    user_id, day = _sidebar(data, cfg)
    scoped = data.for_user(user_id)

    tabs = st.tabs(["Today", "Trends", "Reminders", "Model", "Data"])
    with tabs[0]:
        _tab_today(scoped, cfg, day)
    with tabs[1]:
        _tab_trends(scoped, cfg)
    with tabs[2]:
        _tab_reminders(scoped, cfg)
    with tabs[3]:
        _tab_model(data, cfg)
    with tabs[4]:
        _tab_data(data, cfg)


# ---------------------------------------------------------------------------
def _sidebar(data: DashboardData, cfg) -> tuple[str, pd.Timestamp]:
    with st.sidebar:
        st.subheader("View")
        user_id = st.selectbox("Participant", data.user_ids, index=0)

        days = sorted(
            pd.to_datetime(
                data.windows.loc[data.windows["user_id"] == user_id, "date"]
            ).dt.normalize().unique()
        )
        day = st.selectbox(
            "Day",
            days,
            index=len(days) - 1,
            format_func=lambda d: pd.Timestamp(d).strftime("%A %d %B"),
        )

        if not data.users.empty and "user_id" in data.users.columns:
            row = data.users.loc[data.users["user_id"] == user_id]
            if not row.empty:
                person = row.iloc[0]
                st.markdown(
                    f"**Role** {person['role']}  \n"
                    f"**Experience** {person['years_experience']:.0f} years  \n"
                    f"**Chronotype** {person['chronotype']}"
                )

        st.divider()
        source = (
            "model predictions" if data.state_column == "predicted_state"
            else "recorded labels"
        )
        st.caption(f"States shown from **{source}**.")
        st.caption(f"FocusTrack {__version__}")

        with st.expander("What is recorded"):
            st.markdown(
                "Counts only: keystrokes, clicks, pointer distance, scrolls, "
                "window switches, idle seconds and the foreground application "
                "name.\n\n"
                "**Never** keystroke content, window titles, screenshots or "
                "network traffic. Everything stays on this machine."
            )
    return user_id, pd.Timestamp(day)


# ---------------------------------------------------------------------------
def _tab_today(data: DashboardData, cfg, day: pd.Timestamp) -> None:
    window_minutes = int(cfg["preprocessing.window_minutes"])
    windows = data.windows
    day_windows = windows.loc[windows["date"] == day].sort_values("window_start")
    if day_windows.empty:
        st.info("No activity recorded for this day.")
        return

    summary = data.daily.loc[data.daily["date"] == day]
    row = summary.iloc[0] if not summary.empty else None

    # --- the headline -----------------------------------------------------
    if row is not None:
        colour, verdict = score_band(float(row["focus_score"]))
        left, right = st.columns([1, 2.4])
        with left:
            st.markdown(
                f"<div style='border-left:6px solid {colour};padding:0 0 0 16px'>"
                f"<div style='font-size:64px;font-weight:700;line-height:1;"
                f"color:{colour}'>{row['focus_score']:.0f}</div>"
                f"<div style='color:{MUTED};font-size:14px'>Focus Score - {verdict}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with right:
            a, b, c, d = st.columns(4)
            a.metric("Focused", f"{row['focused_minutes'] / 60:.1f} h")
            b.metric("Distracted", f"{row['distracted_minutes']:.0f} min")
            c.metric("Breaks", f"{row['n_breaks']:.0f}")
            d.metric(
                "Longest stretch",
                f"{row['longest_stretch_minutes']:.0f} min",
                delta=(
                    None
                    if row["longest_stretch_minutes"]
                    <= cfg["focus_score.healthy_stretch_minutes"]
                    else f"+{row['longest_stretch_minutes'] - cfg['focus_score.healthy_stretch_minutes']:.0f} over"
                ),
                delta_color="inverse",
            )

        st.caption(
            f"R (focus ratio) {row['focus_ratio']:.2f}  ·  "
            f"S (switch load) {row['switch_load']:.2f}  ·  "
            f"B (break balance) {row['break_balance']:.2f}"
            + (
                f"  ·  score from recorded labels: {row['focus_score_actual']:.0f}"
                if "focus_score_actual" in row and pd.notna(row.get("focus_score_actual"))
                else ""
            )
        )

    st.divider()

    # --- the day, minute by minute ----------------------------------------
    st.subheader("Your day")
    st.altair_chart(
        timeline_chart(day_windows, data.state_column), use_container_width=True
    )

    notifications = data.notifications
    if len(notifications):
        today_notes = notifications.loc[notifications["date"] == day]
        if len(today_notes):
            st.caption("Reminders sent")
            for note in today_notes.sort_values("timestamp").itertuples():
                icon = "break" if note.kind == BREAK_REMINDER else "nudge"
                st.markdown(
                    f"- `{pd.Timestamp(note.timestamp):%H:%M}` **{icon}** - "
                    f"{getattr(note, 'reason', '')}"
                )

    left, right = st.columns(2)
    with left:
        st.subheader("Where the time went")
        st.altair_chart(
            state_split_chart(state_minutes(day_windows, data.state_column, window_minutes)),
            use_container_width=True,
        )
        categories = category_minutes(day_windows, window_minutes)
        if not categories.empty:
            st.altair_chart(app_mix_chart(categories), use_container_width=True)
    with right:
        st.subheader("How focus drifted")
        if "drop_probability" in day_windows.columns:
            chart_frame = day_windows.loc[:, ["window_start", "drop_probability"]]
            st.line_chart(
                chart_frame.set_index("window_start"),
                color=ACCENT, height=280,
            )
            st.caption(
                f"Predicted probability that focus drops within the next "
                f"{cfg['models.focus_drop.horizon_minutes']} minutes. The engine "
                f"considers a break at {cfg['reminders.drop_probability_threshold']:.2f}."
            )
        else:
            st.info("Train the models to see focus-drop predictions.")


# ---------------------------------------------------------------------------
def _tab_trends(data: DashboardData, cfg) -> None:
    daily = data.daily.sort_values("date")
    if daily.empty:
        st.info("No days recorded yet.")
        return

    st.subheader("Focus Score over time")
    st.altair_chart(score_trend_chart(daily), use_container_width=True)

    a, b, c, d = st.columns(4)
    a.metric("Mean Focus Score", f"{daily['focus_score'].mean():.0f}")
    b.metric("Focused per day", f"{daily['focused_minutes'].mean() / 60:.1f} h")
    c.metric("Breaks per day", f"{daily['n_breaks'].mean():.1f}")
    d.metric("Longest stretch", f"{daily['longest_stretch_minutes'].mean():.0f} min")

    st.divider()
    st.subheader("Where the day goes, hour by hour")
    from focustrack.engine.analytics import hourly_profile

    profile = hourly_profile(data.windows, cfg, state_column=data.state_column)
    st.altair_chart(
        hourly_chart(profile, tuple(cfg["behaviour.slump_window"])),
        use_container_width=True,
    )
    st.caption(
        "The shaded hours are the 14:00-16:00 post-lunch slump, when "
        "distraction risk is highest."
    )

    st.divider()
    st.subheader("Daily detail")
    columns = [
        c for c in (
            "date", "tracked_hours", "focused_hours", "distracted_minutes",
            "meeting_minutes", "n_breaks", "longest_stretch_minutes", "focus_score",
        ) if c in daily.columns
    ]
    st.dataframe(
        daily.loc[:, columns].sort_values("date", ascending=False),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
def _tab_reminders(data: DashboardData, cfg) -> None:
    st.subheader("What the engine decided")
    rules = cfg.section("reminders")
    st.markdown(
        f"""
The engine suggests a break when you are **focused**, the predicted chance of
losing focus within {cfg['models.focus_drop.horizon_minutes']} minutes is at
least **{rules['drop_probability_threshold']}**, and at least
**{rules['min_minutes_since_break']} minutes** have passed since your last
break - or unconditionally after **{rules['hard_break_minutes']} minutes** at
the desk. It nudges after **{rules['distraction_nudge_minutes']} minutes** of
continuous distraction.

It stays quiet during {" and ".join(rules['protected_states']).lower()}s, waits
at least **{rules['min_gap_between_notifications']} minutes** between messages,
and stops after **{rules['max_breaks_per_day']} break reminders** or
**{rules['max_nudges_per_day']} nudges** in a day.
"""
    )

    notifications = data.notifications
    if not len(notifications):
        st.info(
            "No reminders recorded. Run `focustrack all` to replay the engine "
            "over the dataset, or `focustrack agent` to run it live."
        )
        return

    n_days = data.daily["date"].nunique() or 1
    breaks = int((notifications["kind"] == BREAK_REMINDER).sum())
    nudges = len(notifications) - breaks

    a, b, c = st.columns(3)
    a.metric("Break reminders per day", f"{breaks / n_days:.1f}")
    b.metric("Focus nudges per day", f"{nudges / n_days:.1f}")
    c.metric("Total sent", f"{len(notifications)}")

    st.divider()
    results = data.results.get("reminders", {})
    if results:
        st.subheader("Against a fixed timer")
        budget = results.get("equal_budget_alerts", {})
        model, timer = budget.get("model", {}), budget.get("timer", {})
        if model and timer:
            comparison = pd.DataFrame(
                [
                    {
                        "Strategy": timer.get("strategy"),
                        "Alerts": timer.get("n_alerts"),
                        "Precision": timer.get("precision"),
                        "Recall": timer.get("recall"),
                        "ROC-AUC": results.get("roc_auc", {}).get("time_since_break_only"),
                    },
                    {
                        "Strategy": model.get("strategy"),
                        "Alerts": model.get("n_alerts"),
                        "Precision": model.get("precision"),
                        "Recall": model.get("recall"),
                        "ROC-AUC": results.get("roc_auc", {}).get("focus_drop_model"),
                    },
                ]
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.caption(
                "Both strategies are given the same number of alerts, so the "
                "comparison is about *which* moments they pick, not how often "
                "they interrupt."
            )

    st.subheader("Every reminder sent")
    columns = [
        c for c in ("timestamp", "kind", "reason", "minutes_since_break", "drop_probability")
        if c in notifications.columns
    ]
    st.dataframe(
        notifications.loc[:, columns].sort_values("timestamp", ascending=False),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
def _tab_model(data: DashboardData, cfg) -> None:
    results = data.results
    if not results:
        st.info("Run `focustrack all` to produce model results.")
        return

    focus_state = results.get("focus_state", {})
    scores = focus_state.get("scores", [])
    if scores:
        st.subheader("Focus-state classification")
        st.caption(
            "Scored on 10 participants the models never saw during training. "
            "Selected on macro-F1, because Distracted is the rare class and "
            "accuracy would reward ignoring it."
        )
        table = pd.DataFrame(
            [
                {
                    "Model": s["name"],
                    "Accuracy": s["accuracy"],
                    "Macro-F1": s["macro_f1"],
                    "F1 Distracted": s["f1_distracted"],
                    "CV macro-F1": (
                        "-" if s.get("cv_macro_f1_mean") is None
                        else f"{s['cv_macro_f1_mean']:.3f} ± {s['cv_macro_f1_std']:.3f}"
                    ),
                    "Single-state acc.": s.get("accuracy_single_state"),
                    "Mixed-window acc.": s.get("accuracy_mixed_state"),
                }
                for s in scores
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            f"Selected: **{focus_state.get('best_model')}**. Almost every error "
            "is a transition window that contained more than one state."
        )

    groups = focus_state.get("signal_groups", [])
    if groups:
        st.subheader("What the model relies on")
        frame = pd.DataFrame(groups)
        st.bar_chart(
            frame.set_index("group")["importance"], color=ACCENT, horizontal=True
        )
        st.caption(
            "Whole signal families are permuted together. Permuting single "
            "features understates correlated signals, because the others still "
            "carry the information."
        )

    figures = results.get("figures", {})
    for key, caption in (
        ("figure3_diagnostics", "Confusion matrix and feature importance"),
        ("figure1_preprocessing", "The 11-step preprocessing pipeline"),
        ("figure2_architecture", "System architecture"),
    ):
        path = figures.get(key)
        if path and Path(path).exists():
            st.image(path, caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
def _tab_data(data: DashboardData, cfg) -> None:
    st.subheader("What FocusTrack records")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Recorded, once a minute**\n\n"
            "- keystroke *count*\n"
            "- mouse clicks, pointer distance, scroll events\n"
            "- window switches\n"
            "- idle seconds\n"
            "- foreground application *name*\n"
        )
    with right:
        st.markdown(
            "**Never recorded**\n\n"
            "- which keys were pressed\n"
            "- window or document titles\n"
            "- URLs or page contents\n"
            "- screenshots\n"
            "- network traffic\n"
        )
    st.caption(
        "The minute records are encrypted at rest with a key that stays on "
        "this machine. Nothing is uploaded."
    )

    st.divider()
    dataset = data.results.get("dataset", {})
    preprocessing = data.results.get("preprocessing", {})
    if dataset or preprocessing:
        a, b, c, d = st.columns(4)
        a.metric("Participants", dataset.get("n_users", "-"))
        b.metric("Minute records", f"{dataset.get('n_minute_records', 0):,}")
        c.metric("5-minute windows", f"{preprocessing.get('windows', 0):,}")
        d.metric("Features", preprocessing.get("n_features", "-"))

    steps = preprocessing.get("steps", [])
    if steps:
        st.subheader("The processing pipeline")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": s["step"],
                        "Step": s["name"],
                        "Rows in": s["rows_in"],
                        "Rows out": s["rows_out"],
                        "Affected": s["affected"],
                    }
                    for s in steps
                ]
            ),
            use_container_width=True, hide_index=True,
        )

    st.subheader("The feature set")
    from focustrack.preprocessing.features import describe_features

    st.dataframe(describe_features(), use_container_width=True, hide_index=True)

    st.subheader("Recent windows")
    columns = [
        c for c in ("user_id", "window_start", "state_label", "predicted_state",
                    "drop_probability", "minutes_since_break", "idle_mean",
                    "keystrokes_mean")
        if c in data.windows.columns
    ]
    st.dataframe(
        data.windows.loc[:, columns].tail(200).iloc[::-1],
        use_container_width=True, hide_index=True,
    )


def _show_setup_help() -> None:
    st.warning("No processed data found yet.")
    st.markdown(
        """
Run the pipeline first:

```bash
focustrack all
```

That simulates the cohort, runs the 11-step preprocessing pipeline, fits both
models and writes the results this dashboard reads. Then reload this page.

To track your own work instead:

```bash
focustrack agent
```
"""
    )


main()
