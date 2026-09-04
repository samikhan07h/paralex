"""
Feedback logging for the ParaLex Streamlit app.

WHY SQLITE INSTEAD OF A CSV FILE:
A CSV file written from multiple concurrent Streamlit sessions (the
normal case once deployed — many users, or even one user with multiple
browser tabs open) risks corrupted or interleaved writes, since CSV has
no built-in mechanism for safe concurrent append. SQLite handles
concurrent writes safely via file-level locking, while still being a
single, portable file with zero external database server to set up or
deploy — a good fit for a project that needs to "just work" on Streamlit
Community Cloud's free tier without provisioning separate infrastructure.

WHY THIS IS ITS OWN MODULE:
Consistent with app_logic.py and styles.py — logging logic is plain
Python with no Streamlit-runtime dependency, so it's directly unit
testable, and streamlit_app.py stays focused on UI orchestration.

WHAT GETS LOGGED AND WHY:
Beyond just the thumbs up/down, each feedback row captures the question,
answer, sources, retrieval mode, and top_k — enough context to actually
DO something with the feedback later (e.g. "which questions get thumbs-
down most often", "does a higher top_k correlate with better feedback",
or promoting a well-answered question into the Phase 2 eval set). A bare
thumbs-up/down count without this context would be much less useful.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "feedback.db"


@dataclass
class FeedbackEntry:
    question: str
    answer: str
    sources: List[str]
    rating: str  # "up" or "down"
    mode: str  # "Demo Documents" or "Upload Your Own"
    top_k: int
    timestamp: str = ""


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = db_path or DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,
            rating TEXT NOT NULL,
            mode TEXT NOT NULL,
            top_k INTEGER NOT NULL
        )
    """)
    return conn


def log_feedback(entry: FeedbackEntry, db_path: Optional[Path] = None) -> None:
    """Persist one feedback entry to the SQLite database, creating the table on first use."""
    timestamp = entry.timestamp or datetime.now(timezone.utc).isoformat()
    sources_joined = " | ".join(entry.sources)

    conn = _get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO feedback (timestamp, question, answer, sources, rating, mode, top_k) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, entry.question, entry.answer, sources_joined, entry.rating, entry.mode, entry.top_k),
        )
        conn.commit()
    finally:
        conn.close()


def get_feedback_summary(db_path: Optional[Path] = None) -> dict:
    """
    Return aggregate feedback counts — used to show a simple stat in the
    sidebar (e.g. "12 👍 / 2 👎") without needing to load every row.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
        counts = {"up": 0, "down": 0}
        for rating, count in cursor.fetchall():
            counts[rating] = count
        return counts
    finally:
        conn.close()


def get_all_feedback(db_path: Optional[Path] = None) -> List[dict]:
    """Return every logged feedback row as a list of dicts, most recent first."""
    conn = _get_connection(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM feedback ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
