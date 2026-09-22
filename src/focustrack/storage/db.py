"""Layer 2 - the encrypted local SQLite store.

Four tables, all on the user's own machine:

``activity``       one row per minute of activity counts
``self_report``    end-of-day productivity and energy ratings
``predictions``    what the models said about each five-minute window
``notifications``  what the reminder engine sent, and why

The timestamp and user columns are stored in the clear so the database can be
queried by time without decrypting everything; the readings themselves are
encrypted per row. Nothing here is ever uploaded.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from focustrack.config import Config
from focustrack.constants import RAW_COLUMNS
from focustrack.storage.crypto import ValueCipher, load_or_create_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS activity (
    user_id    TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    payload    BLOB NOT NULL,
    PRIMARY KEY (user_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_activity_time ON activity (timestamp);

CREATE TABLE IF NOT EXISTS self_report (
    user_id    TEXT NOT NULL,
    date       TEXT NOT NULL,
    payload    BLOB NOT NULL,
    PRIMARY KEY (user_id, date)
);

CREATE TABLE IF NOT EXISTS predictions (
    user_id      TEXT NOT NULL,
    window_start TEXT NOT NULL,
    payload      BLOB NOT NULL,
    PRIMARY KEY (user_id, window_start)
);

CREATE TABLE IF NOT EXISTS notifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notifications_time ON notifications (timestamp);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class StoreStats:
    """Row counts, for the dashboard and the CLI."""

    activity_rows: int
    self_reports: int
    predictions: int
    notifications: int
    first_timestamp: str | None
    last_timestamp: str | None
    encryption: str

    def describe(self) -> str:
        span = (
            f"{self.first_timestamp} .. {self.last_timestamp}"
            if self.first_timestamp
            else "empty"
        )
        return (
            f"  activity rows   {self.activity_rows:,}\n"
            f"  self reports    {self.self_reports:,}\n"
            f"  predictions     {self.predictions:,}\n"
            f"  notifications   {self.notifications:,}\n"
            f"  covering        {span}\n"
            f"  encryption      {self.encryption}"
        )


class FocusStore:
    """The local encrypted database the agent writes to."""

    def __init__(
        self,
        db_path: str | Path,
        key_path: str | Path | None = None,
        encrypt: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        key = load_or_create_key(key_path) if (encrypt and key_path) else None
        self.cipher = ValueCipher(key, enabled=encrypt)
        self._init_schema()

    @classmethod
    def from_config(cls, cfg: Config) -> "FocusStore":
        return cls(
            db_path=cfg.store_path("db_path"),
            key_path=cfg.store_path("key_path"),
            encrypt=bool(cfg.get("storage.encrypt", True)),
        )

    # -- connection --------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    # -- writing -----------------------------------------------------------
    def record_activity(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert minute records, replacing any already stored for that minute."""
        payloads = []
        for row in rows:
            timestamp = pd.Timestamp(row["timestamp"]).isoformat()
            body = {k: v for k, v in row.items() if k not in ("user_id", "timestamp")}
            payloads.append((str(row["user_id"]), timestamp, self.cipher.encrypt(body)))
        if not payloads:
            return 0
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO activity (user_id, timestamp, payload) "
                "VALUES (?, ?, ?)",
                payloads,
            )
        return len(payloads)

    def record_self_report(
        self, user_id: str, date: Any, productivity: float | None, energy: float | None
    ) -> None:
        payload = {"productivity_rating": productivity, "energy_rating": energy}
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO self_report (user_id, date, payload) VALUES (?, ?, ?)",
                (str(user_id), pd.Timestamp(date).date().isoformat(),
                 self.cipher.encrypt(payload)),
            )

    def record_prediction(
        self, user_id: str, window_start: Any, payload: dict[str, Any]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO predictions (user_id, window_start, payload) "
                "VALUES (?, ?, ?)",
                (str(user_id), pd.Timestamp(window_start).isoformat(),
                 self.cipher.encrypt(payload)),
            )

    def record_notification(
        self, user_id: str, timestamp: Any, kind: str, payload: dict[str, Any]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO notifications (user_id, timestamp, kind, payload) "
                "VALUES (?, ?, ?, ?)",
                (str(user_id), pd.Timestamp(timestamp).isoformat(), kind,
                 self.cipher.encrypt(payload)),
            )

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    # -- reading -----------------------------------------------------------
    def load_activity(
        self,
        user_id: str | None = None,
        since: Any | None = None,
        until: Any | None = None,
    ) -> pd.DataFrame:
        """Return stored minutes in the raw-log schema, decrypted."""
        query = "SELECT user_id, timestamp, payload FROM activity WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(str(user_id))
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(pd.Timestamp(since).isoformat())
        if until is not None:
            query += " AND timestamp < ?"
            params.append(pd.Timestamp(until).isoformat())
        query += " ORDER BY user_id, timestamp"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        if not rows:
            return pd.DataFrame(columns=list(RAW_COLUMNS))

        records = []
        for user, timestamp, payload in rows:
            body = self.cipher.decrypt(payload)
            records.append({"user_id": user, "timestamp": timestamp, **body})

        frame = pd.DataFrame(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        for column in RAW_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return frame.loc[:, list(RAW_COLUMNS)]

    def load_notifications(self, user_id: str | None = None) -> pd.DataFrame:
        query = "SELECT user_id, timestamp, kind, payload FROM notifications"
        params: list[Any] = []
        if user_id is not None:
            query += " WHERE user_id = ?"
            params.append(str(user_id))
        query += " ORDER BY timestamp"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["user_id", "timestamp", "kind", "reason", "message"])

        records = [
            {"user_id": u, "timestamp": t, "kind": k, **self.cipher.decrypt(p)}
            for u, t, k, p in rows
        ]
        frame = pd.DataFrame(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame

    def load_predictions(self, user_id: str | None = None) -> pd.DataFrame:
        query = "SELECT user_id, window_start, payload FROM predictions"
        params: list[Any] = []
        if user_id is not None:
            query += " WHERE user_id = ?"
            params.append(str(user_id))
        query += " ORDER BY window_start"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["user_id", "window_start"])

        records = [
            {"user_id": u, "window_start": w, **self.cipher.decrypt(p)} for u, w, p in rows
        ]
        frame = pd.DataFrame(records)
        frame["window_start"] = pd.to_datetime(frame["window_start"])
        return frame

    def load_self_reports(self) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, date, payload FROM self_report ORDER BY date"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["user_id", "date", "productivity_rating",
                                         "energy_rating"])
        records = [
            {"user_id": u, "date": d, **self.cipher.decrypt(p)} for u, d, p in rows
        ]
        frame = pd.DataFrame(records)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    # -- housekeeping ------------------------------------------------------
    def stats(self) -> StoreStats:
        with self._connect() as connection:
            def count(table: str) -> int:
                return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

            span = connection.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM activity"
            ).fetchone()
            return StoreStats(
                activity_rows=count("activity"),
                self_reports=count("self_report"),
                predictions=count("predictions"),
                notifications=count("notifications"),
                first_timestamp=span[0],
                last_timestamp=span[1],
                encryption=self.cipher.describe(),
            )

    def purge_before(self, cutoff: Any) -> int:
        """Delete activity older than ``cutoff``. The user's data, the user's call."""
        stamp = pd.Timestamp(cutoff).isoformat()
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM activity WHERE timestamp < ?", (stamp,))
            return int(cursor.rowcount)
