"""Shared colours and small chart builders for the dashboard.

The palette matches :mod:`focustrack.evaluation.figures`, so a state is the
same colour whether it appears in a report figure or on screen. Charts are
built with Altair, which ships with Streamlit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from focustrack.constants import BREAK, DISTRACTED, FOCUSED, MEETING, STATES

# --- palette, shared with the report figures --------------------------------
STATE_COLOURS: dict[str, str] = {
    FOCUSED: "#2f6f4e",
    DISTRACTED: "#c2622d",
    BREAK: "#4a6fa5",
    MEETING: "#7a5ea8",
}
ACCENT = "#2f6f4e"
WARN = "#c2622d"
INK = "#1d2433"
MUTED = "#6b7688"

STATE_ORDER = list(STATES)
STATE_RANGE = [STATE_COLOURS[s] for s in STATE_ORDER]

#: Pixel heights of the two band-style charts. Bars drawn with x/x2 and no y
#: encoding have no vertical extent, so these are applied explicitly.
TIMELINE_HEIGHT = 64
SPLIT_BAR_HEIGHT = 48


def score_band(score: float) -> tuple[str, str]:
    """A colour and a short verdict for a Focus Score."""
    if score >= 75:
        return ACCENT, "a strong day"
    if score >= 60:
        return "#5c8a4a", "a solid day"
    if score >= 45:
        return "#c9a227", "a fragmented day"
    return WARN, "a hard day"


def timeline_chart(day: pd.DataFrame, state_column: str = "predicted_state") -> Any:
    """A horizontal band of the day, one mark per five-minute window."""
    import altair as alt

    frame = day.loc[:, ["window_start", state_column]].copy()
    frame["end"] = pd.to_datetime(frame["window_start"]) + pd.Timedelta(minutes=5)
    frame = frame.rename(columns={state_column: "state"})

    # A bar encoded only on x/x2 has no vertical extent and renders invisible.
    # A one-category band scale collapses to a sub-pixel band here, so the
    # height is set in pixels instead: y runs from the top of the plot down.
    colour = alt.Color(
        "state:N",
        scale=alt.Scale(domain=STATE_ORDER, range=STATE_RANGE),
        legend=alt.Legend(title="state", orient="bottom"),
    )
    return (
        alt.Chart(frame)
        # Adjacent windows land on fractional pixel boundaries, which leaves a
        # pale seam between every bar and makes a solid stretch of focus look
        # striped. A 1px stroke in the bar's own colour closes it.
        .mark_bar(strokeWidth=1)
        .encode(
            x=alt.X("window_start:T", title="time of day"),
            x2="end:T",
            y=alt.value(0),
            y2=alt.value(TIMELINE_HEIGHT),
            color=colour,
            stroke=alt.Stroke(
                "state:N",
                scale=alt.Scale(domain=STATE_ORDER, range=STATE_RANGE),
                legend=None,
            ),
            tooltip=["window_start:T", "state:N"],
        )
        .properties(height=TIMELINE_HEIGHT)
    )


def state_split_chart(counts: pd.DataFrame) -> Any:
    """Minutes per state, one bar each.

    A single stacked bar would show the proportion more directly, but stacking
    needs a grouping field on the other axis and this chart has none, so the
    four states get their own rows - which also matches the application-mix
    chart beneath it.
    """
    import altair as alt

    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("minutes:Q", title="minutes"),
            y=alt.Y("state:N", sort=STATE_ORDER, title=None),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(domain=STATE_ORDER, range=STATE_RANGE),
                legend=None,
            ),
            tooltip=["state:N", "minutes:Q"],
        )
        .properties(height=150)
    )


def score_trend_chart(daily: pd.DataFrame) -> Any:
    """Focus Score over time, with the 'healthy' band shaded."""
    import altair as alt

    base = alt.Chart(daily)
    band = (
        alt.Chart(pd.DataFrame({"low": [60], "high": [100]}))
        .mark_rect(opacity=0.06, color=ACCENT)
        .encode(y="low:Q", y2="high:Q")
    )
    line = base.mark_line(point=True, color=ACCENT, strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("focus_score:Q", title="Focus Score", scale=alt.Scale(domain=[0, 100])),
        tooltip=["date:T", alt.Tooltip("focus_score:Q", format=".1f")],
    )
    return (band + line).properties(height=240)


def hourly_chart(profile: pd.DataFrame, slump: tuple[float, float] | None = None) -> Any:
    """Share of each state by hour of day."""
    import altair as alt

    present = [s for s in STATE_ORDER if s in profile.columns]
    long = profile.melt(
        id_vars="hour", value_vars=present, var_name="state", value_name="share"
    )
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("hour:O", title="hour of day"),
            y=alt.Y("share:Q", title="share of windows", stack="normalize"),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(domain=STATE_ORDER, range=STATE_RANGE),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["hour:O", "state:N", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=260)
    )
    if slump is not None:
        marker = (
            alt.Chart(pd.DataFrame({"hour": [int(h) for h in range(int(slump[0]), int(slump[1]))]}))
            .mark_rule(color=WARN, opacity=0.35, strokeWidth=12)
            .encode(x="hour:O")
        )
        return marker + chart
    return chart


def app_mix_chart(minutes_by_category: pd.DataFrame) -> Any:
    """Where the time went, by application category."""
    import altair as alt

    return (
        alt.Chart(minutes_by_category)
        .mark_bar(color=ACCENT, opacity=0.9)
        .encode(
            y=alt.Y("category:N", sort="-x", title=None),
            x=alt.X("minutes:Q", title="minutes"),
            tooltip=["category:N", "minutes:Q"],
        )
        .properties(height=190)
    )
