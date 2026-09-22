"""The four figures from the report.

Figure 1  the preprocessing pipeline and the effect of each step
Figure 2  the five-layer system architecture
Figure 3  (a) confusion matrix on unseen users, (b) permutation importance
Figure 4  one unseen test user's day: actual vs predicted states, and the
          reminders the engine fired

All four are drawn with matplotlib only, on one shared palette, so the set
reads as one system and regenerates identically on any machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from focustrack.config import Config
from focustrack.constants import BREAK, BREAK_REMINDER, DISTRACTED, FOCUSED, MEETING, STATES
from focustrack.preprocessing.features import label as feature_label

# --- one palette for every figure -------------------------------------------
INK = "#1d2433"
MUTED = "#6b7688"
GRID = "#dde2ea"
PANEL = "#f6f7f9"

STATE_COLOURS: dict[str, str] = {
    FOCUSED: "#2f6f4e",       # green - working
    DISTRACTED: "#c2622d",    # amber - drifting
    BREAK: "#4a6fa5",         # blue - away
    MEETING: "#7a5ea8",       # violet - in a call
}
ACCENT = "#2f6f4e"
WARN = "#c2622d"

DPI = 150


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1 - preprocessing pipeline
# ---------------------------------------------------------------------------
def figure_pipeline(audit: Sequence[dict[str, Any]], cfg: Config) -> Path:
    """The eleven steps and how many rows each one touched."""
    _style()
    steps = list(audit)
    names = [f"{s['step']}. {s['name']}" for s in steps]
    affected = [s["affected"] for s in steps]
    rows_out = [s["rows_out"] for s in steps]

    fig, (ax_rows, ax_hits) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )

    # --- left: how the row count changes through the pipeline -------------
    y = np.arange(len(steps))
    ax_rows.plot(rows_out, y, marker="o", color=ACCENT, linewidth=2, markersize=6)
    ax_rows.set_yticks(y, names, fontsize=9)
    ax_rows.invert_yaxis()
    ax_rows.set_xlabel("rows after the step")
    ax_rows.set_title("Rows carried forward", loc="left")
    ax_rows.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:,.0f}")
    )
    # The window step collapses minutes into windows - mark the change of unit.
    window_step = next((i for i, s in enumerate(steps) if "window" in s["name"]), None)
    if window_step is not None:
        ax_rows.axhline(window_step - 0.5, color=MUTED, linestyle=":", linewidth=1)
        # Anchor the note on the left, where the collapsed window counts leave
        # space - the right-hand side is occupied by the minute-level line.
        ax_rows.text(
            min(rows_out) * 0.97, window_step - 0.62,
            "minutes above, 5-minute windows below",
            ha="left", va="bottom", fontsize=8, color=MUTED, style="italic",
        )

    # --- right: what each step actually repaired --------------------------
    bars = ax_hits.barh(y, affected, color=ACCENT, alpha=0.85, height=0.62)
    for i, step in enumerate(steps):
        if step["step"] in (5, 6):      # the steps that deliberately leave data alone
            bars[i].set_color(WARN)
    ax_hits.set_yticks(y, [""] * len(steps))
    ax_hits.invert_yaxis()
    ax_hits.set_xscale("symlog")
    ax_hits.set_xlabel("rows affected (log scale)")
    ax_hits.set_title("What each step changed", loc="left")
    for i, value in enumerate(affected):
        ax_hits.text(
            max(value, 0.6) * 1.35, i, f"{value:,}",
            va="center", fontsize=8.5, color=MUTED,
        )

    fig.suptitle(
        "Figure 1. Preprocessing pipeline and the effect of each step",
        x=0.02, ha="left", fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, cfg.path("figures") / "figure1_preprocessing_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2 - architecture
# ---------------------------------------------------------------------------
LAYERS: tuple[tuple[str, str, str], ...] = (
    ("1. Collection", "desktop agent", "pynput + psutil - counts only,\nnever content"),
    ("2. Storage", "encrypted local SQLite", "minute records stay on\nthe user's machine"),
    ("3. Processing", "the 11-step engine", "runs every 5 minutes on\nthe latest window"),
    ("4. Intelligence", "focus state + focus drop", "what is happening now,\nand what happens next"),
    ("5. Interface", "reminders + dashboard", "adaptive nudges and the\ndaily Focus Score"),
)


def figure_architecture(cfg: Config) -> Path:
    """The five layers and what flows between them."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 10 * len(LAYERS))
    ax.set_ylim(0, 10)
    ax.axis("off")

    width, height = 16.0, 5.2
    gap = (10 * len(LAYERS) - width * len(LAYERS)) / max(len(LAYERS) - 1, 1)
    colours = [STATE_COLOURS[BREAK], STATE_COLOURS[BREAK], ACCENT, STATE_COLOURS[MEETING], WARN]

    for i, (title, subtitle, detail) in enumerate(LAYERS):
        x = i * (width + gap)
        ax.add_patch(
            Rectangle((x, 2.4), width, height, facecolor=PANEL,
                      edgecolor=colours[i], linewidth=1.8, zorder=2)
        )
        ax.text(x + width / 2, 6.9, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=colours[i], zorder=3)
        ax.text(x + width / 2, 5.9, subtitle, ha="center", va="top",
                fontsize=9.5, color=INK, zorder=3)
        ax.text(x + width / 2, 4.7, detail, ha="center", va="top",
                fontsize=8.2, color=MUTED, zorder=3)

        if i < len(LAYERS) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.6, 5.0), (x + width + gap - 0.6, 5.0),
                    arrowstyle="-|>", mutation_scale=16,
                    color=MUTED, linewidth=1.4, zorder=1,
                )
            )

    ax.text(
        0, 1.2,
        "Every layer runs on the user's own machine. Nothing leaves the device, "
        "and only activity counts are ever recorded.",
        fontsize=8.8, color=MUTED, style="italic",
    )
    fig.suptitle("Figure 2. System architecture", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, cfg.path("figures") / "figure2_architecture.png")


# ---------------------------------------------------------------------------
# Figure 3 - confusion matrix and feature importance
# ---------------------------------------------------------------------------
def figure_model_diagnostics(
    confusion: np.ndarray,
    importances: pd.DataFrame,
    model_name: str,
    cfg: Config,
    group_importances: pd.DataFrame | None = None,
) -> Path:
    """(a) confusion matrix on unseen users, (b) permutation importance."""
    _style()
    has_groups = group_importances is not None and not group_importances.empty
    ncols = 3 if has_groups else 2
    widths = [1.0, 1.05, 0.8] if has_groups else [1.0, 1.15]
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols + 1, 5.4),
                            gridspec_kw={"width_ratios": widths})

    # --- (a) confusion matrix, row-normalised ----------------------------
    ax = axes[0]
    matrix = np.asarray(confusion, dtype=float)
    shares = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    image = ax.imshow(shares, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(STATES)), STATES, rotation=30, ha="right")
    ax.set_yticks(range(len(STATES)), STATES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("(a) Confusion matrix, unseen users", loc="left", fontsize=11)
    ax.grid(False)
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            ax.text(
                j, i, f"{shares[i, j]:.2f}\n{int(matrix[i, j]):,}",
                ha="center", va="center", fontsize=8.5,
                color="white" if shares[i, j] > 0.55 else INK,
            )
    fig.colorbar(image, ax=ax, fraction=0.045, label="share of actual class")

    # --- (b) per-feature permutation importance --------------------------
    ax = axes[1]
    top = importances.head(12).iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y, top["importance"], xerr=top.get("std"), color=ACCENT,
            alpha=0.85, height=0.68, error_kw={"ecolor": MUTED, "elinewidth": 0.9})
    ax.set_yticks(y, [feature_label(f) for f in top["feature"]], fontsize=8.5)
    ax.set_xlabel("drop in macro-F1 when permuted")
    ax.set_title("(b) Top features by permutation importance", loc="left", fontsize=11)

    # --- (c) grouped importance ------------------------------------------
    if has_groups:
        ax = axes[2]
        groups = group_importances.iloc[::-1]
        y = np.arange(len(groups))
        ax.barh(y, groups["importance"], xerr=groups.get("std"),
                color=STATE_COLOURS[MEETING], alpha=0.85, height=0.62,
                error_kw={"ecolor": MUTED, "elinewidth": 0.9})
        ax.set_yticks(y, groups["group"], fontsize=8.5)
        ax.set_xlabel("drop in macro-F1")
        ax.set_title("(c) By signal family", loc="left", fontsize=11)

    fig.suptitle(
        f"Figure 3. {model_name} on 10 unseen test users",
        x=0.02, ha="left", fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, cfg.path("figures") / "figure3_model_diagnostics.png")


# ---------------------------------------------------------------------------
# Figure 4 - one day
# ---------------------------------------------------------------------------
def figure_example_day(
    day: pd.DataFrame,
    notifications: pd.DataFrame,
    cfg: Config,
    title_suffix: str = "",
) -> Path:
    """One unseen user's day: actual vs predicted states, and reminders fired."""
    _style()
    day = day.sort_values("window_start").reset_index(drop=True)
    times = pd.to_datetime(day["window_start"])
    window_minutes = int(cfg["preprocessing.window_minutes"])

    fig, (ax_states, ax_prob) = plt.subplots(
        2, 1, figsize=(13, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )

    # --- top: two timelines, actual over predicted -----------------------
    width = pd.Timedelta(minutes=window_minutes)
    for row, (column, name) in enumerate(
        ((("state_label"), "actual"), (("predicted_state"), "predicted"))
    ):
        if column not in day.columns:
            continue
        for time, state in zip(times, day[column]):
            ax_states.add_patch(
                Rectangle(
                    (matplotlib.dates.date2num(time), row * 1.1),
                    width / pd.Timedelta(days=1), 0.9,
                    facecolor=STATE_COLOURS.get(str(state), MUTED),
                    edgecolor="none",
                )
            )
        ax_states.text(
            matplotlib.dates.date2num(times.iloc[0]) - 0.006,
            row * 1.1 + 0.45, name, ha="right", va="center",
            fontsize=9.5, color=MUTED,
        )

    ax_states.set_ylim(-0.25, 2.3)
    ax_states.set_yticks([])
    ax_states.grid(False)
    ax_states.set_title("Actual and predicted state, five minutes at a time",
                        loc="left", fontsize=11)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STATE_COLOURS[s], label=s) for s in STATES
    ]
    ax_states.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02),
                     ncol=4, frameon=False, fontsize=9)

    # --- bottom: the drop probability and where reminders fired ----------
    if "drop_probability" in day.columns:
        ax_prob.plot(times, day["drop_probability"], color=ACCENT,
                     linewidth=1.6, label="P(focus drops in next 15 min)")
        threshold = float(cfg["reminders.drop_probability_threshold"])
        ax_prob.axhline(threshold, color=WARN, linestyle="--", linewidth=1.2,
                        label=f"threshold {threshold:.2f}")
        ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("drop probability")

    if notifications is not None and len(notifications):
        stamps = pd.to_datetime(notifications["timestamp"])
        for stamp, kind in zip(stamps, notifications["kind"]):
            colour = WARN if kind == BREAK_REMINDER else STATE_COLOURS[MEETING]
            ax_prob.axvline(stamp, color=colour, linewidth=1.6, alpha=0.85)
            ax_prob.plot(
                [stamp], [1.02], marker="v", color=colour, markersize=8,
                clip_on=False,
            )
        ax_prob.plot([], [], color=WARN, linewidth=1.6, label="break reminder")
        ax_prob.plot([], [], color=STATE_COLOURS[MEETING], linewidth=1.6,
                     label="focus nudge")

    ax_prob.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=2)
    ax_prob.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax_prob.set_xlabel("time of day")

    fig.suptitle(
        f"Figure 4. One unseen test user's day{title_suffix}",
        x=0.02, ha="left", fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, cfg.path("figures") / "figure4_example_day.png")


# ---------------------------------------------------------------------------
# an extra: how the day looks across the cohort
# ---------------------------------------------------------------------------
def figure_daily_rhythm(
    profile: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: Config,
) -> Path:
    """State mix by hour, and the distribution of Focus Scores."""
    _style()
    fig, (ax_hours, ax_scores) = plt.subplots(1, 2, figsize=(13, 4.8))

    present = [s for s in STATES if s in profile.columns]
    bottom = np.zeros(len(profile))
    for state in present:
        values = profile[state].to_numpy(dtype=float)
        ax_hours.bar(profile["hour"], values, bottom=bottom, width=0.82,
                     color=STATE_COLOURS[state], label=state)
        bottom += values

    slump_lo, slump_hi = cfg["behaviour.slump_window"]
    ax_hours.axvspan(slump_lo - 0.5, slump_hi - 0.5, color=WARN, alpha=0.10, zorder=0)
    ax_hours.text((slump_lo + slump_hi) / 2 - 0.5, 1.02, "afternoon slump",
                  ha="center", fontsize=8.5, color=WARN)
    ax_hours.set_xlabel("hour of day")
    ax_hours.set_ylabel("share of windows")
    ax_hours.set_ylim(0, 1)
    ax_hours.set_title("Where the day goes, hour by hour", loc="left", fontsize=11)
    ax_hours.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center")

    ax_scores.hist(summary["focus_score"], bins=30, color=ACCENT, alpha=0.85)
    mean = float(summary["focus_score"].mean())
    ax_scores.axvline(mean, color=WARN, linestyle="--", linewidth=1.4,
                      label=f"mean {mean:.1f}")
    ax_scores.set_xlabel("Focus Score")
    ax_scores.set_ylabel("user-days")
    ax_scores.set_title("Focus Score across all user-days", loc="left", fontsize=11)
    ax_scores.legend(frameon=False, fontsize=9)

    fig.suptitle("Figure 5. Daily rhythm and Focus Score distribution",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, cfg.path("figures") / "figure5_daily_rhythm.png")
