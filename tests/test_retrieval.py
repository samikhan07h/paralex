"""
Tests for src/retrieval/retriever.py

We mock the Embedder here (rather than loading the real model) to keep
these tests fast and independent of network access — the Retriever's job
is to correctly wire embedder output into vectorstore search, which we can
verify precisely with a controlled, fake embedding. Real end-to-end
retrieval quality (with the actual embedding model) is covered in
test_pipeline_integration.py.
"""

import numpy as np
from unittest.mock import MagicMock

from src.chunking.chunker import Chunk
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever, RetrievalResult


def _make_fake_store(n=5, dim=384, seed=0):
    rng = np.random.default_rng(seed)
    chunks = [
        Chunk(text=f"chunk {i}", source="fake.pdf", page_number=1,
              chunk_id=f"fake_{i}", chunk_strategy="test", metadata={})
        for i in range(n)
    ]
    embeddings = rng.standard_normal((n, dim)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    store = VectorStore(embedding_dim=dim)
    store.add(chunks, embeddings)
    return store, embeddings


def test_retrieve_returns_retrieval_result_objects():
    store, embeddings = _make_fake_store()
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = embeddings[0]

    retriever = Retriever(embedder=mock_embedder, vectorstore=store)
    results = retriever.retrieve("any query", top_k=2)

    assert all(isinstance(r, RetrievalResult) for r in results)
    assert len(results) == 2


def test_retrieve_routes_query_through_embedder_then_vectorstore():
    store, embeddings = _make_fake_store()
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = embeddings[3]  # pretend query embeds to chunk 3's vector

    retriever = Retriever(embedder=mock_embedder, vectorstore=store)
    results = retriever.retrieve("what does clause 3 say?", top_k=1)

    mock_embedder.embed_query.assert_called_once_with("what does clause 3 say?")
    assert results[0].chunk.chunk_id == "fake_3"
    assert abs(results[0].score - 1.0) < 1e-4


def test_retrieve_respects_default_top_k_from_config():
    from src import config
    store, embeddings = _make_fake_store(n=10)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = embeddings[0]

    retriever = Retriever(embedder=mock_embedder, vectorstore=store)
    results = retriever.retrieve("a query")  # no explicit top_k

    assert len(results) == min(config.TOP_K, 10)


def test_retrieve_with_no_min_score_returns_all_top_k_results():
    """Default behavior (min_score=None) must be unchanged — this is what Phase 2's measured numbers depend on."""
    store, embeddings = _make_fake_store(n=5)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = embeddings[0]

    retriever = Retriever(embedder=mock_embedder, vectorstore=store)
    results = retriever.retrieve("a query", top_k=5)

    assert len(results) == 5


def test_retrieve_with_min_score_drops_low_scoring_results():
    store, embeddings = _make_fake_store(n=5, seed=1)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = embeddings[0]  # exact match on itself -> score 1.0 for chunk 0

    retriever = Retriever(embedder=mock_embedder, vectorstore=store)
    all_results = retriever.retrieve("a query", top_k=5, min_score=None)
    filtered_results = retriever.retrieve("a query", top_k=5, min_score=0.99)

    # The exact self-match (score ~1.0) should survive a 0.99 threshold;
    # the other 4 near-random-vector matches (much lower similarity)
    # should not.
    assert len(filtered_results) < len(all_results)
    assert all(r.score >= 0.99 for r in filtered_results)


def test_retrieve_min_score_can_filter_out_everything():
    store, embeddings = _make_fake_store(n=3)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = embeddings[0]

    retriever = Retriever(embedder=mock_embedder, vectorstore=store)
    results = retriever.retrieve("a query", top_k=3, min_score=1.5)  # impossible threshold for cosine similarity

    assert results == []
