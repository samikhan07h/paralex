"""
Tests for src/embeddings/embedder.py

NOTE: These tests download the sentence-transformers model on first run
(~80MB, cached afterward under ~/.cache/huggingface). They require an
internet connection the first time they run.
"""

import numpy as np
import pytest

from src.ingestion.loaders import load_documents_from_dir
from src.chunking.chunker import clause_aware_chunk
from src.embeddings.embedder import Embedder
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(scope="module")
def embedder():
    # scope="module" so the (somewhat expensive) model load happens once
    # for all tests in this file, not once per test.
    return Embedder()


def test_embedder_loads_and_reports_correct_dimension(embedder):
    assert embedder.embedding_dim == 384  # all-MiniLM-L6-v2's known output size


def test_embed_texts_returns_correct_shape(embedder):
    texts = ["This is a test sentence.", "Another unrelated sentence about finance."]
    embeddings = embedder.embed_texts(texts)

    assert embeddings.shape == (2, embedder.embedding_dim)


def test_embeddings_are_normalized(embedder):
    embeddings = embedder.embed_texts(["Any text will do for this check."])
    norm = np.linalg.norm(embeddings[0])

    assert abs(norm - 1.0) < 1e-4  # normalize_embeddings=True should give unit vectors


def test_embed_query_matches_embed_texts_single_item(embedder):
    text = "What is the monthly rent?"
    via_query = embedder.embed_query(text)
    via_texts = embedder.embed_texts([text])[0]

    assert np.allclose(via_query, via_texts)


def test_semantically_similar_clause_scores_higher_than_unrelated(embedder):
    """
    End-to-end sanity check: embed real chunked clauses and confirm a
    pets-related query is closer (higher dot product / cosine similarity,
    since vectors are normalized) to the PETS clause than to an unrelated
    loan DEFAULT clause. This is the kind of check that catches a broken
    or misconfigured embedding pipeline even when the code "runs".
    """
    pages = load_documents_from_dir(SAMPLE_DIR)
    chunks = clause_aware_chunk(pages)

    pets_chunk = next(
        c for c in chunks
        if c.source == "sample_lease_agreement.pdf" and c.metadata.get("clause_number") == "7"
    )
    loan_default_chunk = next(
        c for c in chunks
        if c.source == "sample_loan_agreement.pdf" and c.metadata.get("clause_number") == "6"
    )

    query_emb = embedder.embed_query("Are pets allowed in the apartment?")
    pets_emb = embedder.embed_query(pets_chunk.text)
    loan_emb = embedder.embed_query(loan_default_chunk.text)

    sim_to_pets = float(np.dot(query_emb, pets_emb))
    sim_to_loan = float(np.dot(query_emb, loan_emb))

    assert sim_to_pets > sim_to_loan
