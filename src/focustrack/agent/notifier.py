"""Delivering reminders to the user.

Tries a real desktop notification first and falls back to the console, so the
agent is usable on a headless machine and in tests without pretending a
notification was shown when it was not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from focustrack.constants import BREAK_REMINDER
from focustrack.engine.reminders import Notification

APP_NAME = "FocusTrack"


class Notifier(Protocol):
    """Anything that can show a reminder."""

    def send(self, notification: Notification) -> bool: ...


@dataclass
class ConsoleNotifier:
    """Prints to stdout. The dependable fallback."""

    sent: list[Notification] = field(default_factory=list)

    def send(self, notification: Notification) -> bool:
        icon = "[break]" if notification.kind == BREAK_REMINDER else "[focus]"
        stamp = notification.timestamp.strftime("%H:%M")
        print(f"\n  {icon} {stamp}  {notification.message}")
        print(f"          why: {notification.reason}\n", flush=True)
        self.sent.append(notification)
        return True


@dataclass
class DesktopNotifier:
    """A real system notification, falling back to the console if unavailable."""

    fallback: ConsoleNotifier = field(default_factory=ConsoleNotifier)
    sent: list[Notification] = field(default_factory=list)
    _backend: Callable[[str, str], bool] | None = field(default=None, init=False)
    _resolved: bool = field(default=False, init=False)

    def send(self, notification: Notification) -> bool:
        title = (
            "Time for a break"
            if notification.kind == BREAK_REMINDER
            else "Still on track?"
        )
        backend = self._resolve()
        delivered = backend(title, notification.message) if backend else False
        if not delivered:
            self.fallback.send(notification)
        self.sent.append(notification)
        return delivered

    def _resolve(self) -> Callable[[str, str], bool] | None:
        if self._resolved:
            return self._backend
        self._resolved = True
        self._backend = _find_backend()
        return self._backend

    @property
    def backend_name(self) -> str:
        self._resolve()
        return "desktop" if self._backend else "console"


def _find_backend() -> Callable[[str, str], bool] | None:
    """Return the first working desktop-notification backend, if any."""
    try:
        from plyer import notification as plyer_notification

        def send_plyer(title: str, message: str) -> bool:
            try:
                plyer_notification.notify(
                    title=title, message=message, app_name=APP_NAME, timeout=12
                )
                return True
            except Exception:
                return False

        return send_plyer
    except Exception:
        pass

    try:
        import subprocess
        import shutil

        if shutil.which("notify-send"):

            def send_libnotify(title: str, message: str) -> bool:
                try:
                    subprocess.run(
                        ["notify-send", "-a", APP_NAME, title, message],
                        check=True, capture_output=True, timeout=5,
                    )
                    return True
                except Exception:
                    return False

            return send_libnotify
    except Exception:
        pass

    return None


def build_notifier(prefer_desktop: bool = True) -> Notifier:
    """The notifier the agent should use on this machine."""
    return DesktopNotifier() if prefer_desktop else ConsoleNotifier()
