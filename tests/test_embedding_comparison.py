"""
Tests for evaluation/embedding_comparison.py

Mocks Embedder and time.time() to verify the comparison ASSEMBLY logic
(embedding dimension, speed calculation, retrieval metrics per model)
deterministically, without downloading real models or depending on
timing variance. The real comparison (via `python -m evaluation.embedding_comparison`)
is what produces the actual numbers for the README.
"""

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from evaluation.embedding_comparison import (
    compare_embedding_models,
    print_comparison,
    save_comparison,
    ModelComparisonResult,
)
from evaluation.test_sets_loader import EvalItem
from src.chunking.chunker import Chunk


def _make_item(item_id, source, clause_number):
    return EvalItem(
        id=item_id, question="a question?", expected_answer="an answer",
        expected_source_doc=source, expected_clause_number=clause_number,
        expected_keywords=["keyword"],
    )


def _make_chunk(source, clause_number):
    return Chunk(text="some chunk text", source=source, page_number=1,
                 chunk_id="x", chunk_strategy="test", metadata={"clause_number": clause_number})


@patch("evaluation.embedding_comparison.load_all_test_sets")
@patch("evaluation.embedding_comparison.load_and_chunk_documents")
@patch("evaluation.embedding_comparison.Embedder")
def test_compare_embedding_models_reports_correct_dimension_and_chunk_count(
    mock_embedder_cls, mock_load_chunks, mock_load_items
):
    chunks = [_make_chunk("lease.pdf", "3")]
    mock_load_chunks.return_value = chunks
    mock_load_items.return_value = [_make_item("q1", "lease.pdf", "3")]

    mock_embedder = MagicMock()
    mock_embedder.embedding_dim = 384
    mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
    mock_embedder.embed_query.return_value = np.array([0.1] * 384)
    mock_embedder_cls.return_value = mock_embedder

    results = compare_embedding_models(model_names=["fake-model"], top_k=1)

    assert len(results) == 1
    assert results[0].embedding_dim == 384
    assert results[0].num_chunks == 1
    assert results[0].model_name == "fake-model"


@patch("evaluation.embedding_comparison.load_all_test_sets")
@patch("evaluation.embedding_comparison.load_and_chunk_documents")
@patch("evaluation.embedding_comparison.Embedder")
def test_compare_embedding_models_computes_chunks_per_second(
    mock_embedder_cls, mock_load_chunks, mock_load_items
):
    chunks = [_make_chunk("lease.pdf", "3")] * 10
    mock_load_chunks.return_value = chunks
    mock_load_items.return_value = [_make_item("q1", "lease.pdf", "3")]

    mock_embedder = MagicMock()
    mock_embedder.embedding_dim = 384
    mock_embedder.embed_texts.return_value = np.array([[0.1] * 384] * 10)
    mock_embedder.embed_query.return_value = np.array([0.1] * 384)
    mock_embedder_cls.return_value = mock_embedder

    # Simulate embedding taking exactly 2 seconds for 10 chunks -> 5 chunks/sec.
    call_times = iter([100.0, 102.0])
    with patch("evaluation.embedding_comparison.time.time", side_effect=lambda: next(call_times)):
        results = compare_embedding_models(model_names=["fake-model"], top_k=1)

    assert results[0].embed_time_seconds == 2.0
    assert results[0].chunks_per_second == 5.0


@patch("evaluation.embedding_comparison.load_all_test_sets")
@patch("evaluation.embedding_comparison.load_and_chunk_documents")
@patch("evaluation.embedding_comparison.Embedder")
def test_compare_embedding_models_runs_each_candidate_independently(
    mock_embedder_cls, mock_load_chunks, mock_load_items
):
    """Two candidate models should each get their own embedder instance and independent metrics."""
    chunks = [_make_chunk("lease.pdf", "3")]
    mock_load_chunks.return_value = chunks
    mock_load_items.return_value = [_make_item("q1", "lease.pdf", "3")]

    mock_embedder = MagicMock()
    mock_embedder.embedding_dim = 384
    mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
    mock_embedder.embed_query.return_value = np.array([0.1] * 384)
    mock_embedder_cls.return_value = mock_embedder

    results = compare_embedding_models(model_names=["model-a", "model-b"], top_k=1)

    assert len(results) == 2
    assert results[0].model_name == "model-a"
    assert results[1].model_name == "model-b"
    assert mock_embedder_cls.call_count == 2


def test_print_comparison_runs_without_error_and_shows_key_columns(capsys):
    results = [
        ModelComparisonResult(
            model_name="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384,
            num_chunks=24, embed_time_seconds=1.2, chunks_per_second=20.0,
            mrr=0.917, recall_at_k=1.0, precision_at_k=0.5, top_k=2,
        ),
        ModelComparisonResult(
            model_name="sentence-transformers/all-mpnet-base-v2", embedding_dim=768,
            num_chunks=24, embed_time_seconds=4.8, chunks_per_second=5.0,
            mrr=0.95, recall_at_k=1.0, precision_at_k=0.5, top_k=2,
        ),
    ]

    print_comparison(results)
    captured = capsys.readouterr()

    assert "MRR" in captured.out
    assert "all-MiniLM-L6-v2" in captured.out
    assert "all-mpnet-base-v2" in captured.out
    assert "vs" in captured.out  # the comparison summary line


def test_save_comparison_writes_valid_json():
    results = [
        ModelComparisonResult(
            model_name="fake-model", embedding_dim=384, num_chunks=10,
            embed_time_seconds=1.0, chunks_per_second=10.0,
            mrr=1.0, recall_at_k=1.0, precision_at_k=0.5, top_k=1,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "comparison.json"
        saved_path = save_comparison(results, path)

        with open(saved_path) as f:
            loaded = json.load(f)
        assert loaded[0]["model_name"] == "fake-model"
        assert loaded[0]["mrr"] == 1.0
