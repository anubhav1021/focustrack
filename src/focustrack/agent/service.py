"""The live agent: all five layers, running on the user's machine.

Every minute a sample is taken and written to the encrypted store. Every five
minutes the processing engine rebuilds the latest window from the stored
minutes, the models score it, and the reminder engine decides whether to say
anything.

The five-minute cycle deliberately re-runs the same pipeline used for
training rather than a streaming shortcut, so a window the agent scores is
built exactly the way the windows the model learned from were built. A
separate "fast path" here would be the most likely place for training and
serving to quietly drift apart.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from focustrack.config import Config
from focustrack.constants import FOCUSED
from focustrack.engine.focus_score import score_day
from focustrack.engine.reminders import (
    EngineSettings,
    Notification,
    ReminderEngine,
    resolve_drop_threshold,
)
from focustrack.models.registry import ModelBundle, load_bundle, models_available
from focustrack.preprocessing.features import feature_matrix
from focustrack.preprocessing.pipeline import run_pipeline
from focustrack.storage.db import FocusStore
from focustrack.agent.collector import ActivityCollector
from focustrack.agent.notifier import Notifier, build_notifier

#: How much recent history the five-minute cycle rebuilds. Enough to give the
#: history features their previous 15 minutes and the break detector room to
#: see a full break.
CONTEXT_MINUTES = 180


@dataclass
class AgentStatus:
    """What the agent has done so far this session."""

    started_at: datetime
    samples_taken: int = 0
    cycles_run: int = 0
    notifications_sent: int = 0
    last_state: str | None = None
    last_drop_probability: float | None = None
    last_focus_score: float | None = None
    errors: list[str] = field(default_factory=list)

    def describe(self) -> str:
        uptime = datetime.now() - self.started_at
        minutes = int(uptime.total_seconds() // 60)
        lines = [
            f"  uptime            {minutes} min",
            f"  samples taken     {self.samples_taken}",
            f"  cycles run        {self.cycles_run}",
            f"  reminders sent    {self.notifications_sent}",
        ]
        if self.last_state:
            lines.append(f"  current state     {self.last_state}")
        if self.last_drop_probability is not None:
            lines.append(f"  drop probability  {self.last_drop_probability:.2f}")
        if self.last_focus_score is not None:
            lines.append(f"  focus score today {self.last_focus_score:.0f}/100")
        if self.errors:
            lines.append(f"  errors            {len(self.errors)} (see log)")
        return "\n".join(lines)


class FocusTrackAgent:
    """The running product: collect, store, process, predict, remind."""

    def __init__(
        self,
        cfg: Config,
        user_id: str = "local",
        store: FocusStore | None = None,
        models: ModelBundle | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.cfg = cfg
        self.user_id = user_id
        self.store = store or FocusStore.from_config(cfg)
        self.notifier = notifier or build_notifier()
        self.collector = ActivityCollector()
        self._models = models
        # The engine uses whatever threshold the fitted model was calibrated
        # for, so the live agent behaves the way the evaluation said it would.
        self.engine = ReminderEngine(
            EngineSettings.from_config(
                cfg, drop_threshold=resolve_drop_threshold(cfg, self._calibrated())
            ),
            int(cfg["preprocessing.window_minutes"]),
        )
        self.status = AgentStatus(started_at=datetime.now())
        self._current_day: Any = None
        self._stop = False

    def _calibrated(self) -> float | None:
        """The calibrated threshold recorded when the models were fitted."""
        bundle = self.models
        if bundle is None:
            return None
        value = bundle.metadata.get("drop_threshold_calibrated")
        return float(value) if value else None

    # -- models ------------------------------------------------------------
    @property
    def models(self) -> ModelBundle | None:
        """The fitted models, loaded lazily.

        The agent still collects and stores without them; it simply cannot
        predict or remind, and says so rather than failing.
        """
        if self._models is None and models_available(self.cfg):
            self._models = load_bundle(self.cfg)
        return self._models

    # -- the two cadences --------------------------------------------------
    def sample_minute(self, now: datetime | None = None) -> dict[str, Any]:
        """Take one minute's counts and store them."""
        now = now or datetime.now()
        self.collector.poll_foreground_app()
        counts = self.collector.drain()
        row = counts.as_row(self.user_id, now.replace(second=0, microsecond=0))
        self.store.record_activity([row])
        self.status.samples_taken += 1
        self._roll_day(now)
        return row

    def run_cycle(self, now: datetime | None = None) -> Notification | None:
        """The five-minute cycle: rebuild, predict, decide."""
        now = now or datetime.now()
        self.status.cycles_run += 1

        windows = self.recent_windows(now)
        if windows is None or windows.empty:
            return None

        latest = windows.iloc[[-1]]
        bundle = self.models
        if bundle is None:
            self.status.errors.append("no fitted models - run `focustrack train`")
            return None

        features = feature_matrix(latest)
        state = str(bundle.focus_state.predict(features)[0])
        drop_probability = float(bundle.focus_drop.predict_proba(features)[0, 1])
        minutes_since_break = float(latest["minutes_since_break"].iloc[0])

        self.status.last_state = state
        self.status.last_drop_probability = drop_probability

        self.store.record_prediction(
            self.user_id,
            latest["window_start"].iloc[0],
            {
                "predicted_state": state,
                "drop_probability": drop_probability,
                "minutes_since_break": minutes_since_break,
            },
        )

        # The Focus Score so far today, for the dashboard and the status line.
        scored = windows.copy()
        scored["predicted_state"] = bundle.focus_state.predict(feature_matrix(scored))
        today = scored.loc[scored["date"] == scored["date"].iloc[-1]]
        if not today.empty:
            self.status.last_focus_score = score_day(
                today, self.cfg, state_column="predicted_state"
            ).score

        notification = self.engine.step(
            user_id=self.user_id,
            timestamp=pd.Timestamp(latest["window_start"].iloc[0]),
            state=state,
            minutes_since_break=minutes_since_break,
            drop_probability=drop_probability if state == FOCUSED else drop_probability,
        )
        if notification is not None:
            self.notifier.send(notification)
            self.store.record_notification(
                self.user_id, notification.timestamp, notification.kind,
                {
                    "reason": notification.reason,
                    "message": notification.message,
                    "minutes_since_break": notification.minutes_since_break,
                    "drop_probability": notification.drop_probability,
                },
            )
            self.status.notifications_sent += 1
        return notification

    def recent_windows(self, now: datetime | None = None) -> pd.DataFrame | None:
        """Rebuild the recent window table from the stored minutes."""
        now = now or datetime.now()
        since = now - timedelta(minutes=CONTEXT_MINUTES)
        minutes = self.store.load_activity(user_id=self.user_id, since=since)
        if minutes.empty:
            return None
        try:
            result = run_pipeline(minutes, self.cfg)
        except Exception as exc:                              # pragma: no cover
            self.status.errors.append(f"pipeline: {exc}")
            return None
        return result.windows if not result.windows.empty else None

    # -- day boundaries ----------------------------------------------------
    def _roll_day(self, now: datetime) -> None:
        """Reset the engine's daily caps when the date changes."""
        today = now.date()
        if self._current_day is None:
            self._current_day = today
        elif today != self._current_day:
            self.engine.state.reset()
            self._current_day = today

    # -- the loop ----------------------------------------------------------
    def run(self, duration_minutes: float | None = None, verbose: bool = True) -> AgentStatus:
        """Run until interrupted, or for ``duration_minutes``."""
        sample_seconds = float(self.cfg["agent.sample_seconds"])
        cycle_every = int(self.cfg["agent.process_every_minutes"])

        self.collector.start()
        self._install_signal_handlers()
        if verbose:
            print(f"FocusTrack agent running as '{self.user_id}'.")
            print(ActivityCollector.describe_capabilities())
            print(f"  store             {self.store.db_path}")
            print(f"  sampling every    {sample_seconds:.0f}s")
            print(f"  processing every  {cycle_every} min")
            if self.models is None:
                print("  models            none fitted yet - collecting only")
            print("  press Ctrl+C to stop\n", flush=True)

        deadline = (
            time.monotonic() + duration_minutes * 60.0
            if duration_minutes is not None
            else None
        )
        try:
            while not self._stop:
                slept = self._sleep_until_next_sample(sample_seconds, deadline)
                if not slept:
                    break
                self.sample_minute()
                if self.status.samples_taken % cycle_every == 0:
                    self.run_cycle()
                if verbose and self.status.samples_taken % cycle_every == 0:
                    print(
                        f"  [{datetime.now():%H:%M}] "
                        f"state={self.status.last_state or '-'} "
                        f"p(drop)={self.status.last_drop_probability or float('nan'):.2f} "
                        f"score={self.status.last_focus_score or float('nan'):.0f}",
                        flush=True,
                    )
        except KeyboardInterrupt:                             # pragma: no cover
            pass
        finally:
            self.collector.stop()
            if verbose:
                print("\nagent stopped.")
                print(self.status.describe())
        return self.status

    def _sleep_until_next_sample(self, seconds: float, deadline: float | None) -> bool:
        """Sleep in short slices so Ctrl+C is responsive. False means stop."""
        target = time.monotonic() + seconds
        while time.monotonic() < target:
            if self._stop:
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(min(0.25, max(target - time.monotonic(), 0.0)))
        return not (deadline is not None and time.monotonic() >= deadline)

    def _install_signal_handlers(self) -> None:
        def handle(_signum: int, _frame: Any) -> None:        # pragma: no cover
            self._stop = True

        for name in ("SIGINT", "SIGTERM"):
            handler = getattr(signal, name, None)
            if handler is not None:
                try:
                    signal.signal(handler, handle)
                except (ValueError, OSError):                 # pragma: no cover
                    pass

    def stop(self) -> None:
        self._stop = True
