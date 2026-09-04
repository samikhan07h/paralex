"""
Tests for src/vectorstore/store.py

These tests use synthetic (random, normalized) embeddings rather than
real ones from the Embedder. This is intentional: the vector store's job
is to store and search vectors correctly regardless of where they came
from, so testing it in isolation from the embedding model keeps these
tests fast and independent of any network/model-download dependency.
End-to-end behavior with real embeddings is covered separately in
test_pipeline_integration.py.
"""

import numpy as np
import pytest
import tempfile
import shutil

from src.chunking.chunker import Chunk
from src.vectorstore.store import VectorStore


def _make_fake_chunks_and_embeddings(n=10, dim=384, seed=42):
    rng = np.random.default_rng(seed)
    chunks = [
        Chunk(
            text=f"This is fake chunk number {i}",
            source="fake.pdf",
            page_number=1,
            chunk_id=f"fake_{i}",
            chunk_strategy="test",
            metadata={},
        )
        for i in range(n)
    ]
    embeddings = rng.standard_normal((n, dim)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return chunks, embeddings


def test_add_increases_index_size():
    chunks, embeddings = _make_fake_chunks_and_embeddings(n=10)
    store = VectorStore(embedding_dim=384)
    store.add(chunks, embeddings)

    assert store.index.ntotal == 10
    assert len(store.chunks) == 10


def test_add_rejects_mismatched_chunk_and_embedding_counts():
    chunks, embeddings = _make_fake_chunks_and_embeddings(n=10)
    store = VectorStore(embedding_dim=384)

    with pytest.raises(ValueError, match="Mismatch"):
        store.add(chunks[:5], embeddings)  # 5 chunks but 10 embeddings


def test_add_rejects_wrong_embedding_dimension():
    chunks, embeddings = _make_fake_chunks_and_embeddings(n=5, dim=128)
    store = VectorStore(embedding_dim=384)  # store expects 384, not 128

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add(chunks, embeddings)


def test_search_retrieves_exact_match_with_similarity_near_one():
    chunks, embeddings = _make_fake_chunks_and_embeddings(n=20)
    store = VectorStore(embedding_dim=384)
    store.add(chunks, embeddings)

    query = embeddings[7]  # search using an already-indexed vector
    results = store.search(query, top_k=3)

    assert results[0][0].chunk_id == "fake_7"
    assert abs(results[0][1] - 1.0) < 1e-4


def test_search_returns_results_ordered_by_descending_similarity():
    chunks, embeddings = _make_fake_chunks_and_embeddings(n=20)
    store = VectorStore(embedding_dim=384)
    store.add(chunks, embeddings)

    results = store.search(embeddings[3], top_k=5)
    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)


def test_search_on_empty_index_returns_empty_list():
    store = VectorStore(embedding_dim=384)
    query = np.random.default_rng(0).standard_normal(384).astype("float32")

    assert store.search(query, top_k=5) == []


def test_search_caps_top_k_at_available_results():
    chunks, embeddings = _make_fake_chunks_and_embeddings(n=3)
    store = VectorStore(embedding_dim=384)
    store.add(chunks, embeddings)

    # Requesting more neighbors than exist shouldn't error, just cap out.
    results = store.search(embeddings[0], top_k=100)
    assert len(results) == 3


def test_save_and_load_round_trip_preserves_search_results():
    chunks, embeddings = _make_fake_chunks_and_embeddings(n=15)
    store = VectorStore(embedding_dim=384)
    store.add(chunks, embeddings)

    tmpdir = tempfile.mkdtemp()
    try:
        store.save(tmpdir)
        loaded = VectorStore.load(tmpdir)

        assert loaded.index.ntotal == 15
        assert len(loaded.chunks) == 15

        original_results = store.search(embeddings[4], top_k=3)
        loaded_results = loaded.search(embeddings[4], top_k=3)

        assert [c.chunk_id for c, _ in original_results] == [c.chunk_id for c, _ in loaded_results]
    finally:
        shutil.rmtree(tmpdir)


def test_load_raises_clear_error_if_no_saved_store_exists():
    tmpdir = tempfile.mkdtemp()
    try:
        with pytest.raises(FileNotFoundError, match="No saved vector store"):
            VectorStore.load(tmpdir)
    finally:
        shutil.rmtree(tmpdir)
