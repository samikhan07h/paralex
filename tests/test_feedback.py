"""
Tests for app/feedback.py

Each test uses a fresh temporary database file, so tests never share
state or depend on execution order — a real requirement here since
log_feedback() has side effects (persisted rows) that would otherwise
leak between tests.
"""

import tempfile
from pathlib import Path

import pytest

from app.feedback import FeedbackEntry, log_feedback, get_feedback_summary, get_all_feedback


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_feedback.db"


def _make_entry(rating="up", question="What is the rent?"):
    return FeedbackEntry(
        question=question,
        answer="The rent is $2,400/month.",
        sources=["Source 1: lease.pdf, Clause 3 (RENT), page 1"],
        rating=rating,
        mode="Demo Documents",
        top_k=2,
    )


def test_log_feedback_creates_db_and_persists_entry(temp_db):
    log_feedback(_make_entry(), db_path=temp_db)

    assert temp_db.exists()
    rows = get_all_feedback(db_path=temp_db)
    assert len(rows) == 1
    assert rows[0]["question"] == "What is the rent?"
    assert rows[0]["rating"] == "up"


def test_log_feedback_joins_multiple_sources_into_one_field(temp_db):
    entry = _make_entry()
    entry.sources = ["Source 1: a.pdf, page 1", "Source 2: b.pdf, page 2"]
    log_feedback(entry, db_path=temp_db)

    rows = get_all_feedback(db_path=temp_db)
    assert "Source 1: a.pdf, page 1" in rows[0]["sources"]
    assert "Source 2: b.pdf, page 2" in rows[0]["sources"]


def test_log_feedback_handles_empty_sources_list(temp_db):
    entry = _make_entry()
    entry.sources = []
    log_feedback(entry, db_path=temp_db)  # should not raise

    rows = get_all_feedback(db_path=temp_db)
    assert rows[0]["sources"] == ""


def test_get_feedback_summary_counts_up_and_down_separately(temp_db):
    log_feedback(_make_entry(rating="up"), db_path=temp_db)
    log_feedback(_make_entry(rating="up"), db_path=temp_db)
    log_feedback(_make_entry(rating="down"), db_path=temp_db)

    summary = get_feedback_summary(db_path=temp_db)
    assert summary["up"] == 2
    assert summary["down"] == 1


def test_get_feedback_summary_returns_zero_counts_for_empty_db(temp_db):
    summary = get_feedback_summary(db_path=temp_db)
    assert summary == {"up": 0, "down": 0}


def test_get_all_feedback_returns_most_recent_first(temp_db):
    log_feedback(_make_entry(question="First question"), db_path=temp_db)
    log_feedback(_make_entry(question="Second question"), db_path=temp_db)

    rows = get_all_feedback(db_path=temp_db)
    assert rows[0]["question"] == "Second question"
    assert rows[1]["question"] == "First question"


def test_log_feedback_auto_generates_timestamp_when_not_provided(temp_db):
    entry = _make_entry()
    assert entry.timestamp == ""

    log_feedback(entry, db_path=temp_db)

    rows = get_all_feedback(db_path=temp_db)
    assert rows[0]["timestamp"] != ""
    assert "T" in rows[0]["timestamp"]  # ISO format marker


def test_multiple_log_calls_reuse_the_same_table_without_error(temp_db):
    """Calling log_feedback repeatedly should not fail on 'table already exists'."""
    for i in range(3):
        log_feedback(_make_entry(question=f"Question {i}"), db_path=temp_db)

    rows = get_all_feedback(db_path=temp_db)
    assert len(rows) == 3
