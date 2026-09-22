"""Fixed vocabularies: states, roles, the 37-application catalogue, schemas.

These are structural facts about the system rather than tunable parameters,
so they live in code instead of ``config.yaml``.
"""

from __future__ import annotations

from typing import Final

# --- the four hidden states -------------------------------------------------
FOCUSED: Final = "Focused"
DISTRACTED: Final = "Distracted"
BREAK: Final = "Break"
MEETING: Final = "Meeting"

STATES: Final[tuple[str, ...]] = (FOCUSED, DISTRACTED, BREAK, MEETING)
STATE_INDEX: Final[dict[str, int]] = {s: i for i, s in enumerate(STATES)}

#: States the reminder engine treats as "the user is working".
ACTIVE_STATES: Final[tuple[str, ...]] = (FOCUSED, DISTRACTED, MEETING)

# --- roles and chronotypes --------------------------------------------------
ROLES: Final[tuple[str, ...]] = ("developer", "designer", "writer", "analyst", "support")
CHRONOTYPES: Final[tuple[str, ...]] = ("morning", "intermediate", "evening")

# --- the six application categories ----------------------------------------
CATEGORIES: Final[tuple[str, ...]] = (
    "development",
    "creative",
    "docs_analysis",
    "research",
    "communication",
    "personal",
)

#: Categories that count as genuine work surfaces.
WORK_CATEGORIES: Final[tuple[str, ...]] = (
    "development",
    "creative",
    "docs_analysis",
    "research",
    "communication",
)

# --- the 37-application catalogue (application name -> category) -----------
APP_CATALOGUE: Final[dict[str, str]] = {
    # development (7)
    "Visual Studio Code": "development",
    "PyCharm": "development",
    "IntelliJ IDEA": "development",
    "Windows Terminal": "development",
    "Docker Desktop": "development",
    "Postman": "development",
    "DBeaver": "development",
    # creative (5)
    "Figma": "creative",
    "Adobe Photoshop": "creative",
    "Adobe Illustrator": "creative",
    "Canva": "creative",
    "Blender": "creative",
    # docs_analysis (8)
    "Microsoft Word": "docs_analysis",
    "Microsoft Excel": "docs_analysis",
    "Microsoft PowerPoint": "docs_analysis",
    "Google Docs": "docs_analysis",
    "Notion": "docs_analysis",
    "Jira": "docs_analysis",
    "Tableau": "docs_analysis",
    "Jupyter Notebook": "docs_analysis",
    # research (5)
    "Google Chrome": "research",
    "Mozilla Firefox": "research",
    "Microsoft Edge": "research",
    "Stack Overflow": "research",
    "Confluence": "research",
    # communication (6)
    "Slack": "communication",
    "Microsoft Teams": "communication",
    "Zoom": "communication",
    "Google Meet": "communication",
    "Microsoft Outlook": "communication",
    "Zendesk": "communication",
    # personal (6)
    "YouTube": "personal",
    "Instagram": "personal",
    "X (Twitter)": "personal",
    "Reddit": "personal",
    "WhatsApp Web": "personal",
    "Spotify": "personal",
}

APPS: Final[tuple[str, ...]] = tuple(APP_CATALOGUE)
N_APPS: Final[int] = len(APPS)          # 37
UNKNOWN_CATEGORY: Final = "unknown"

#: Applications that carry a live meeting.
MEETING_APPS: Final[tuple[str, ...]] = ("Zoom", "Google Meet", "Microsoft Teams")

# --- raw log schema ---------------------------------------------------------
#: Columns of ``raw_activity_logs.csv``. Counts only - never content.
RAW_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "timestamp",
    "active_app",
    "keystrokes",
    "mouse_clicks",
    "mouse_distance_px",
    "scroll_events",
    "window_switches",
    "idle_seconds",
    "state_label",
)

ACTIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "keystrokes",
    "mouse_clicks",
    "mouse_distance_px",
    "scroll_events",
    "window_switches",
    "idle_seconds",
)

#: Activity counts normalised within each user (see report section 3).
NORMALISED_COLUMNS: Final[tuple[str, ...]] = (
    "keystrokes",
    "mouse_clicks",
    "mouse_distance_px",
    "scroll_events",
)

USERS_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "role",
    "years_experience",
    "chronotype",
)

SURVEY_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "date",
    "productivity_rating",   # 1-10
    "energy_rating",         # 1-5
)

# --- notification kinds -----------------------------------------------------
BREAK_REMINDER: Final = "break_reminder"
FOCUS_NUDGE: Final = "focus_nudge"
