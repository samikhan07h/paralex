"""
Tests for src/chunking/chunker.py
"""

from src.ingestion.loaders import load_documents_from_dir
from src.chunking.chunker import recursive_chunk, clause_aware_chunk
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def _load_pages():
    return load_documents_from_dir(SAMPLE_DIR)


def test_recursive_chunk_produces_nonempty_chunks():
    pages = _load_pages()
    chunks = recursive_chunk(pages)

    assert len(chunks) > 0
    assert all(c.text.strip() for c in chunks)
    assert all(c.chunk_strategy == "recursive" for c in chunks)


def test_recursive_chunk_respects_size_roughly():
    pages = _load_pages()
    chunks = recursive_chunk(pages, chunk_size=200, chunk_overlap=20)

    # No chunk should wildly exceed the requested size (small tolerance for
    # separator-boundary splitting behavior in RecursiveCharacterTextSplitter).
    assert all(len(c.text) <= 260 for c in chunks)


def test_clause_aware_chunk_detects_numbered_clauses():
    pages = _load_pages()
    chunks = clause_aware_chunk(pages)

    lease_chunks = [c for c in chunks if c.source == "sample_lease_agreement.pdf"]
    clause_numbers = {c.metadata.get("clause_number") for c in lease_chunks}

    # The sample lease has 10 numbered clauses — every one should be detected.
    assert clause_numbers == {str(i) for i in range(1, 11)}
    assert all(c.chunk_strategy == "clause_aware" for c in lease_chunks)


def test_clause_aware_chunk_keeps_each_clause_intact():
    pages = _load_pages()
    chunks = clause_aware_chunk(pages)

    pets_clause = next(
        c for c in chunks
        if c.source == "sample_lease_agreement.pdf" and c.metadata.get("clause_number") == "7"
    )
    assert "PETS" in pets_clause.text
    assert "pet deposit" in pets_clause.text  # full clause body present, not cut off


def test_clause_aware_chunk_falls_back_for_unstructured_text():
    pages = _load_pages()
    chunks = clause_aware_chunk(pages)

    tenk_chunks = [c for c in chunks if c.source == "sample_10k_excerpt.pdf"]

    # The 10-K excerpt has no numbered clause headers, so every chunk
    # should have used the fallback strategy, not "clause_aware".
    assert len(tenk_chunks) > 0
    assert all(c.chunk_strategy == "clause_aware_fallback" for c in tenk_chunks)


def test_clause_aware_produces_more_chunks_than_baseline_for_clause_heavy_docs():
    pages = _load_pages()
    r_chunks = recursive_chunk(pages)
    ca_chunks = clause_aware_chunk(pages)

    # Clause-aware should split the clause-heavy lease/loan docs more finely
    # (one chunk per clause) than the baseline, which merges multiple
    # clauses per chunk until it hits the character limit.
    assert len(ca_chunks) > len(r_chunks)
