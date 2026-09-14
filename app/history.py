"""Local, private meeting history (SQLite). Never leaves the device."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("snapassist.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title      TEXT,
    language   TEXT,
    transcript TEXT,
    summary    TEXT,
    action_items TEXT
);
"""


class History:
    def __init__(self, db_path: Path = DB_PATH):
        # check_same_thread=False + a lock: the server calls this from FastAPI's threadpool,
        # so the single connection is shared across threads with serialized access.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(_SCHEMA)

    def add(self, title: str, language: str, transcript: str, summary: str,
            action_items: list[str]) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO meetings (created_at, title, language, transcript, summary, action_items)"
                " VALUES (?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), title, language, transcript, summary,
                 "\n".join(action_items)),
            )
            self.conn.commit()
            return cur.lastrowid

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM meetings ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()

    def search(self, term: str) -> list[sqlite3.Row]:
        """Simple full-text-ish search — the basis for offline 'ask my past meetings'."""
        like = f"%{term}%"
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM meetings WHERE transcript LIKE ? OR summary LIKE ? "
                "ORDER BY created_at DESC", (like, like)
            ).fetchall()
