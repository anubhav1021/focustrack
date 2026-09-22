"""Layer 1 - the desktop agent.

Counts only. No keystroke content, no window titles, no screenshots.
"""

from focustrack.agent.collector import (
    INPUT_AVAILABLE,
    PSUTIL_AVAILABLE,
    ActivityCollector,
    MinuteCounts,
    current_application,
)
from focustrack.agent.notifier import (
    ConsoleNotifier,
    DesktopNotifier,
    Notifier,
    build_notifier,
)
from focustrack.agent.self_report import SelfReport, prompt_self_report, prompt_window_label
from focustrack.agent.service import AgentStatus, FocusTrackAgent

__all__ = [
    "INPUT_AVAILABLE",
    "PSUTIL_AVAILABLE",
    "ActivityCollector",
    "AgentStatus",
    "ConsoleNotifier",
    "DesktopNotifier",
    "FocusTrackAgent",
    "MinuteCounts",
    "Notifier",
    "SelfReport",
    "build_notifier",
    "current_application",
    "prompt_self_report",
    "prompt_window_label",
]
