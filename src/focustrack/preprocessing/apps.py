"""Group applications into the six categories used by the feature set.

The catalogue in :mod:`focustrack.constants` covers the 37 applications in the
pilot dataset. A live agent will meet applications that are not in it, so
unknown names fall through to a keyword heuristic before being labelled
``unknown``.
"""

from __future__ import annotations

import pandas as pd

from focustrack.constants import APP_CATALOGUE, CATEGORIES, UNKNOWN_CATEGORY

#: Substring heuristics for applications outside the catalogue, tried in order.
#: Lower-cased substring -> category.
_KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    # development
    ("code", "development"), ("studio", "development"), ("pycharm", "development"),
    ("intellij", "development"), ("webstorm", "development"), ("vim", "development"),
    ("emacs", "development"), ("terminal", "development"), ("iterm", "development"),
    ("sublime", "development"), ("atom", "development"), ("neovim", "development"),
    ("eclipse", "development"), ("rider", "development"), ("goland", "development"),
    ("powershell", "development"), ("cmd.exe", "development"), ("docker", "development"),
    ("git", "development"), ("postman", "development"), ("sql", "development"),
    ("dbeaver", "development"), ("xcode", "development"),
    # creative
    ("figma", "creative"), ("photoshop", "creative"), ("illustrator", "creative"),
    ("sketch", "creative"), ("canva", "creative"), ("blender", "creative"),
    ("premiere", "creative"), ("affinity", "creative"), ("indesign", "creative"),
    # communication
    ("slack", "communication"), ("teams", "communication"), ("zoom", "communication"),
    ("meet", "communication"), ("outlook", "communication"), ("gmail", "communication"),
    ("mail", "communication"), ("webex", "communication"), ("zendesk", "communication"),
    ("freshdesk", "communication"), ("discord", "communication"),
    # personal
    ("youtube", "personal"), ("instagram", "personal"), ("twitter", "personal"),
    ("reddit", "personal"), ("whatsapp", "personal"), ("spotify", "personal"),
    ("netflix", "personal"), ("tiktok", "personal"), ("facebook", "personal"),
    ("steam", "personal"), ("twitch", "personal"),
    # docs and analysis
    ("word", "docs_analysis"), ("excel", "docs_analysis"), ("powerpoint", "docs_analysis"),
    ("docs", "docs_analysis"), ("sheets", "docs_analysis"), ("notion", "docs_analysis"),
    ("obsidian", "docs_analysis"), ("jira", "docs_analysis"), ("tableau", "docs_analysis"),
    ("jupyter", "docs_analysis"), ("power bi", "docs_analysis"), ("rstudio", "docs_analysis"),
    ("pdf", "docs_analysis"), ("acrobat", "docs_analysis"),
    # research / browsing
    ("chrome", "research"), ("firefox", "research"), ("edge", "research"),
    ("safari", "research"), ("brave", "research"), ("stack overflow", "research"),
    ("confluence", "research"), ("wikipedia", "research"),
)


def categorise_app(app: str) -> str:
    """Return the category of a single application name."""
    if not isinstance(app, str) or not app.strip():
        return UNKNOWN_CATEGORY
    exact = APP_CATALOGUE.get(app)
    if exact is not None:
        return exact
    lowered = app.lower()
    for needle, category in _KEYWORD_RULES:
        if needle in lowered:
            return category
    return UNKNOWN_CATEGORY


def categorise_series(apps: pd.Series) -> pd.Series:
    """Vectorised categorisation of an ``active_app`` column."""
    # The catalogue covers the overwhelming majority; only map the rest.
    mapped = apps.map(APP_CATALOGUE)
    missing = mapped.isna()
    if missing.any():
        unique_unknown = apps[missing].dropna().unique()
        lookup = {name: categorise_app(name) for name in unique_unknown}
        mapped = mapped.astype(object)
        mapped[missing] = apps[missing].map(lookup)
    return pd.Categorical(
        mapped.fillna(UNKNOWN_CATEGORY),
        categories=[*CATEGORIES, UNKNOWN_CATEGORY],
    )
