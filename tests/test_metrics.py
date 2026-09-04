"""
Tests for evaluation/metrics.py

We use a mocked Retriever (via a fake .retrieve() returning controlled
RetrievalResult lists) so we can test the metric MATH precisely — e.g.
"if the correct chunk is at rank 2, MRR contribution should be exactly
0.5" — without depending on real embeddings or the network. Real
end-to-end retrieval metrics against the actual pipeline are covered
separately in test_eval_integration.py.
"""

import pytest
from unittest.mock import MagicMock

from evaluation.metrics import (
    is_relevant_chunk,
    find_rank_of_relevant_chunk,
    evaluate_retrieval,
)
from evaluation.test_sets_loader import EvalItem
from src.chunking.chunker import Chunk
from src.retrieval.retriever import RetrievalResult


def _chunk(source, clause_number=None, text="some text"):
    metadata = {"clause_number": clause_number} if clause_number else {}
    return Chunk(text=text, source=source, page_number=1, chunk_id="x",
                 chunk_strategy="test", metadata=metadata)


def _item(source, clause_number=None, keywords=None):
    return EvalItem(
        id="test_item", question="a question?", expected_answer="an answer",
        expected_source_doc=source, expected_clause_number=clause_number,
        expected_keywords=keywords or [],
    )


# --- is_relevant_chunk ---

def test_is_relevant_chunk_matches_source_and_clause_number():
    chunk = _chunk("lease.pdf", clause_number="3")
    item = _item("lease.pdf", clause_number="3")
    assert is_relevant_chunk(chunk, item) is True


def test_is_relevant_chunk_rejects_wrong_clause_number():
    chunk = _chunk("lease.pdf", clause_number="7")
    item = _item("lease.pdf", clause_number="3")
    assert is_relevant_chunk(chunk, item) is False


def test_is_relevant_chunk_rejects_wrong_source_even_with_matching_clause():
    chunk = _chunk("loan.pdf", clause_number="3")
    item = _item("lease.pdf", clause_number="3")
    assert is_relevant_chunk(chunk, item) is False


def test_is_relevant_chunk_uses_keyword_matching_when_no_clause_number():
    chunk = _chunk("10k.pdf", text="Total net revenue was $4.82 billion.")
    item = _item("10k.pdf", clause_number=None, keywords=["4.82 billion"])
    assert is_relevant_chunk(chunk, item) is True


def test_is_relevant_chunk_keyword_matching_rejects_missing_keyword():
    chunk = _chunk("10k.pdf", text="Operating expenses increased.")
    item = _item("10k.pdf", clause_number=None, keywords=["4.82 billion"])
    assert is_relevant_chunk(chunk, item) is False


# --- find_rank_of_relevant_chunk ---

def test_find_rank_returns_correct_1_indexed_position():
    item = _item("lease.pdf", clause_number="3")
    results = [
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="7"), score=0.9),
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="3"), score=0.8),  # relevant, rank 2
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="1"), score=0.7),
    ]
    assert find_rank_of_relevant_chunk(results, item) == 2


def test_find_rank_returns_none_when_not_found():
    item = _item("lease.pdf", clause_number="3")
    results = [
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="7"), score=0.9),
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="1"), score=0.8),
    ]
    assert find_rank_of_relevant_chunk(results, item) is None


# --- evaluate_retrieval (aggregate metrics) ---

def test_evaluate_retrieval_perfect_retrieval_gives_mrr_of_one():
    """Every question's correct chunk retrieved at rank 1 -> MRR should be exactly 1.0."""
    items = [_item("lease.pdf", clause_number="3"), _item("loan.pdf", clause_number="2")]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = [
        [RetrievalResult(chunk=_chunk("lease.pdf", clause_number="3"), score=0.95)],
        [RetrievalResult(chunk=_chunk("loan.pdf", clause_number="2"), score=0.93)],
    ]

    report = evaluate_retrieval(mock_retriever, items, top_k=1)
    assert report.mrr == 1.0
    assert report.recall_at_k == 1.0


def test_evaluate_retrieval_complete_miss_gives_mrr_of_zero():
    items = [_item("lease.pdf", clause_number="3")]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="9"), score=0.5),
    ]

    report = evaluate_retrieval(mock_retriever, items, top_k=1)
    assert report.mrr == 0.0
    assert report.recall_at_k == 0.0


def test_evaluate_retrieval_mrr_reflects_rank_position():
    """Correct chunk found at rank 2 (out of top_k=4) -> reciprocal rank contribution should be 0.5."""
    items = [_item("lease.pdf", clause_number="3")]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="9"), score=0.9),
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="3"), score=0.85),  # rank 2
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="1"), score=0.7),
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="5"), score=0.6),
    ]

    report = evaluate_retrieval(mock_retriever, items, top_k=4)
    assert report.mrr == pytest.approx(0.5)
    assert report.per_item_results[0].rank == 2


def test_evaluate_retrieval_report_summary_is_readable_string():
    items = [_item("lease.pdf", clause_number="3")]
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(chunk=_chunk("lease.pdf", clause_number="3"), score=0.9),
    ]

    report = evaluate_retrieval(mock_retriever, items, top_k=1)
    summary = report.summary()
    assert "MRR" in summary
    assert "Recall@1" in summary
