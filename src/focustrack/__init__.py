"""FocusTrack - Remote Work Productivity Booster.

Tracks work hours, breaks and focus for remote workers, predicts the focus
state every 5 minutes, fires adaptive break reminders and focus nudges, and
turns the result into daily analytics and a Focus Score.

The package is organised as the five layers of the Milestone 2 architecture:

    focustrack.agent          layer 1 - collection (desktop agent)
    focustrack.storage        layer 2 - encrypted local SQLite store
    focustrack.preprocessing  layer 3 - the 11-step processing engine
    focustrack.models         layer 4 - intelligence (classifier + predictor)
    focustrack.engine         layer 4/5 - reminders, Focus Score, analytics
    focustrack.dashboard      layer 5 - Streamlit dashboard

Only activity *counts* are ever recorded - never keystroke content, window
titles beyond the application name, or screen contents.
"""

__version__ = "0.2.0"

from focustrack.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
