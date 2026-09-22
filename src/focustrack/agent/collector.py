"""Layer 1 - the desktop activity collector.

Once a minute this records how much happened, and nothing about what happened:

* how many keys were pressed - never **which** keys,
* how many clicks, how far the pointer travelled, how many scroll events,
* how many times the foreground window changed,
* how many seconds passed with no input at all,
* the name of the foreground application - not its window title.

No keystroke content, no window titles, no screenshots, no network traffic.
That boundary is the reason the product is installable on a personal machine,
so it is enforced here at the point of collection rather than downstream: the
data that would need protecting is never captured in the first place.

``pynput`` and ``psutil`` are optional. Without them the collector still runs
and reports what it can, marking the rest as unavailable rather than guessing.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# --- optional dependencies --------------------------------------------------
try:
    from pynput import keyboard as _keyboard, mouse as _mouse
    INPUT_AVAILABLE = True
except Exception:                                            # pragma: no cover
    _keyboard = _mouse = None                                # type: ignore[assignment]
    INPUT_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:                                            # pragma: no cover
    psutil = None                                            # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

UNKNOWN_APP = "unknown"


@dataclass
class MinuteCounts:
    """One minute of activity, in the raw-log schema."""

    keystrokes: int = 0
    mouse_clicks: int = 0
    mouse_distance_px: int = 0
    scroll_events: int = 0
    window_switches: int = 0
    idle_seconds: int = 0
    active_app: str = UNKNOWN_APP

    def as_row(self, user_id: str, timestamp: datetime) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "timestamp": timestamp,
            "active_app": self.active_app,
            "keystrokes": self.keystrokes,
            "mouse_clicks": self.mouse_clicks,
            "mouse_distance_px": self.mouse_distance_px,
            "scroll_events": self.scroll_events,
            "window_switches": self.window_switches,
            "idle_seconds": self.idle_seconds,
            "state_label": None,          # only a self-report can fill this in
        }


class ActivityCollector:
    """Counts input events and foreground-app changes for the current minute.

    Thread-safe: the input listeners fire on their own threads, and
    :meth:`drain` swaps the counters out under a lock so a sample is never
    partly from two different minutes.
    """

    def __init__(self, idle_threshold_seconds: float = 1.0) -> None:
        self.idle_threshold_seconds = idle_threshold_seconds
        self._lock = threading.Lock()
        self._counts = MinuteCounts()
        self._last_pointer: tuple[int, int] | None = None
        self._last_event_at = time.monotonic()
        self._window_opened_at = time.monotonic()
        self._idle_accumulated = 0.0
        self._current_app = UNKNOWN_APP
        self._listeners: list[Any] = []
        self._running = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ActivityCollector":
        """Begin listening. Safe to call when pynput is unavailable."""
        if self._running:
            return self
        self._running = True
        self._window_opened_at = time.monotonic()
        self._last_event_at = time.monotonic()

        if INPUT_AVAILABLE:
            keyboard_listener = _keyboard.Listener(on_press=self._on_key)
            mouse_listener = _mouse.Listener(
                on_click=self._on_click, on_move=self._on_move, on_scroll=self._on_scroll
            )
            for listener in (keyboard_listener, mouse_listener):
                listener.daemon = True
                listener.start()
                self._listeners.append(listener)
        return self

    def stop(self) -> None:
        for listener in self._listeners:
            try:
                listener.stop()
            except Exception:                                # pragma: no cover
                pass
        self._listeners.clear()
        self._running = False

    def __enter__(self) -> "ActivityCollector":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- input callbacks ---------------------------------------------------
    # Each callback records only that *an* event happened. The key itself is
    # never inspected, stored, or passed on.
    def _on_key(self, _key: Any) -> None:
        with self._lock:
            self._counts.keystrokes += 1
            self._mark_active()

    def _on_click(self, _x: int, _y: int, _button: Any, pressed: bool) -> None:
        if not pressed:
            return
        with self._lock:
            self._counts.mouse_clicks += 1
            self._mark_active()

    def _on_scroll(self, _x: int, _y: int, _dx: int, _dy: int) -> None:
        with self._lock:
            self._counts.scroll_events += 1
            self._mark_active()

    def _on_move(self, x: int, y: int) -> None:
        with self._lock:
            if self._last_pointer is not None:
                dx = x - self._last_pointer[0]
                dy = y - self._last_pointer[1]
                self._counts.mouse_distance_px += int(math.hypot(dx, dy))
            self._last_pointer = (x, y)
            self._mark_active()

    def _mark_active(self) -> None:
        """Close off any idle stretch that just ended. Call under the lock."""
        now = time.monotonic()
        gap = now - self._last_event_at
        if gap >= self.idle_threshold_seconds:
            self._idle_accumulated += gap
        self._last_event_at = now

    # -- foreground application -------------------------------------------
    def poll_foreground_app(self) -> str:
        """Record the foreground application name, counting any change."""
        name = current_application()
        with self._lock:
            if name != self._current_app and self._current_app != UNKNOWN_APP:
                self._counts.window_switches += 1
            self._current_app = name
        return name

    # -- sampling ----------------------------------------------------------
    def drain(self) -> MinuteCounts:
        """Return the counts since the last drain and start a fresh minute."""
        now = time.monotonic()
        with self._lock:
            idle = self._idle_accumulated
            trailing = now - self._last_event_at
            if trailing >= self.idle_threshold_seconds:
                idle += trailing

            elapsed = max(now - self._window_opened_at, 1e-6)
            counts = self._counts
            counts.active_app = self._current_app
            counts.idle_seconds = int(round(min(idle, elapsed, 60.0)))

            self._counts = MinuteCounts()
            self._idle_accumulated = 0.0
            self._window_opened_at = now
            self._last_event_at = now
            self._last_pointer = None
        return counts

    # -- introspection -----------------------------------------------------
    @staticmethod
    def capability_report() -> dict[str, bool]:
        return {
            "input_events": INPUT_AVAILABLE,
            "foreground_app": PSUTIL_AVAILABLE or _platform_window_api_available(),
        }

    @staticmethod
    def describe_capabilities() -> str:
        report = ActivityCollector.capability_report()
        lines = []
        for name, available in report.items():
            status = "available" if available else "NOT available"
            lines.append(f"  {name:18} {status}")
        if not report["input_events"]:
            lines.append("  -> install pynput to count keyboard and mouse events")
        if not report["foreground_app"]:
            lines.append("  -> install psutil to identify the foreground application")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# foreground application, per platform
# ---------------------------------------------------------------------------
def _platform_window_api_available() -> bool:
    try:
        import ctypes  # noqa: F401
        return True
    except Exception:                                        # pragma: no cover
        return False


def current_application() -> str:
    """Best-effort name of the foreground application.

    Only the executable or process name is read - never the window title,
    which would leak document names, URLs and message contents.
    """
    name = _foreground_windows()
    if name:
        return name
    return _foreground_psutil() or UNKNOWN_APP


def _foreground_windows() -> str | None:
    """Windows: resolve the foreground window to its owning process name."""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:                                        # pragma: no cover
        return None
    if not hasattr(ctypes, "windll"):
        return None

    try:
        user32 = ctypes.windll.user32
        handle = user32.GetForegroundWindow()
        if not handle:
            return None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        if not pid.value or not PSUTIL_AVAILABLE:
            return None
        return _friendly_name(psutil.Process(pid.value).name())
    except Exception:                                        # pragma: no cover
        return None


def _foreground_psutil() -> str | None:
    """Fallback: the busiest non-system process the user is running."""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        best: tuple[float, str] | None = None
        for process in psutil.process_iter(["name", "cpu_percent"]):
            info = process.info
            name = info.get("name") or ""
            if not name or name.lower() in _SYSTEM_PROCESSES:
                continue
            usage = float(info.get("cpu_percent") or 0.0)
            if best is None or usage > best[0]:
                best = (usage, name)
        return _friendly_name(best[1]) if best else None
    except Exception:                                        # pragma: no cover
        return None


_SYSTEM_PROCESSES = {
    "system", "system idle process", "registry", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "dwm.exe", "fontdrvhost.exe",
    "kernel_task", "launchd", "systemd",
}

#: Map executable names onto the catalogue names the pipeline knows.
_EXECUTABLE_NAMES: dict[str, str] = {
    "code.exe": "Visual Studio Code", "code": "Visual Studio Code",
    "pycharm64.exe": "PyCharm", "pycharm": "PyCharm",
    "idea64.exe": "IntelliJ IDEA",
    "windowsterminal.exe": "Windows Terminal", "wt.exe": "Windows Terminal",
    "docker desktop.exe": "Docker Desktop",
    "postman.exe": "Postman",
    "dbeaver.exe": "DBeaver",
    "figma.exe": "Figma",
    "photoshop.exe": "Adobe Photoshop",
    "illustrator.exe": "Adobe Illustrator",
    "blender.exe": "Blender",
    "winword.exe": "Microsoft Word",
    "excel.exe": "Microsoft Excel",
    "powerpnt.exe": "Microsoft PowerPoint",
    "notion.exe": "Notion",
    "chrome.exe": "Google Chrome", "google chrome": "Google Chrome",
    "firefox.exe": "Mozilla Firefox", "firefox": "Mozilla Firefox",
    "msedge.exe": "Microsoft Edge",
    "slack.exe": "Slack", "slack": "Slack",
    "ms-teams.exe": "Microsoft Teams", "teams.exe": "Microsoft Teams",
    "zoom.exe": "Zoom", "zoom.us": "Zoom",
    "outlook.exe": "Microsoft Outlook",
    "spotify.exe": "Spotify", "spotify": "Spotify",
}


def _friendly_name(executable: str) -> str:
    """Map an executable name to a catalogue application name where possible."""
    if not executable:
        return UNKNOWN_APP
    lowered = executable.lower()
    if lowered in _EXECUTABLE_NAMES:
        return _EXECUTABLE_NAMES[lowered]
    # Otherwise keep the process name, minus the extension, for categorisation.
    return executable.rsplit(".", 1)[0] if lowered.endswith(".exe") else executable
