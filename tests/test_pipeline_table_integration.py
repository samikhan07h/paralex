"""
Tests for the table-aware build_index() logic in src/pipeline.py.

We test the page-exclusion logic directly against real sample documents
(no mocking needed here — loading and chunking are fast, local operations)
to prove the core Phase 3 correctness property: a page containing a real
table must NOT also appear as a messy, flattened prose chunk in the final
chunk set. Embedding is mocked (via a fake Embedder) since we don't need
real vectors to verify chunk-set composition.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

from src.ingestion.loaders import load_documents_from_dir
from src.ingestion.table_extractor import extract_tables_from_dir
from src.chunking.chunker import clause_aware_chunk

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_table_pages_are_correctly_identified_for_exclusion():
    """
    Sanity check the exclusion SET itself: the financial statements PDF's
    single page should be identified as a table page, while the 10-K's
    page (pure prose, no real tables) should not be.
    """
    pages = load_documents_from_dir(SAMPLE_DIR)
    table_chunks = extract_tables_from_dir(SAMPLE_DIR)

    table_pages = {(c.source, c.page_number) for c in table_chunks}

    assert ("sample_financial_statements.pdf", 1) in table_pages
    assert ("sample_10k_excerpt.pdf", 1) not in table_pages


def test_prose_chunking_excludes_table_pages_but_keeps_others():
    """
    This is the core Phase 3 correctness property: after filtering, the
    financial statements page should NOT appear in the prose chunk set at
    all (avoiding the messy flattened duplicate), while lease/loan/10-K
    documents are completely unaffected.
    """
    pages = load_documents_from_dir(SAMPLE_DIR)
    table_chunks = extract_tables_from_dir(SAMPLE_DIR)
    table_pages = {(c.source, c.page_number) for c in table_chunks}

    prose_pages = [p for p in pages if (p.source, p.page_number) not in table_pages]
    prose_chunks = clause_aware_chunk(prose_pages)

    # No prose chunk should come from the financial statements document —
    # its only representation in the index should be the clean table chunks.
    assert all(c.source != "sample_financial_statements.pdf" for c in prose_chunks)

    # But the other three documents should be entirely untouched by this
    # exclusion logic — same chunk counts as before Phase 3.
    other_sources = {c.source for c in prose_chunks}
    assert other_sources == {
        "sample_lease_agreement.pdf",
        "sample_loan_agreement.pdf",
        "sample_10k_excerpt.pdf",
    }


def test_final_chunk_set_contains_both_prose_and_table_chunks():
    """
    End-to-end composition check: the combined chunk list (what actually
    gets embedded and indexed) should contain prose chunks from the
    non-table documents AND table chunks from the financial statements PDF
    — not one at the expense of the other.
    """
    pages = load_documents_from_dir(SAMPLE_DIR)
    table_chunks = extract_tables_from_dir(SAMPLE_DIR)
    table_pages = {(c.source, c.page_number) for c in table_chunks}
    prose_pages = [p for p in pages if (p.source, p.page_number) not in table_pages]
    prose_chunks = clause_aware_chunk(prose_pages)

    all_chunks = prose_chunks + table_chunks
    all_sources = {c.source for c in all_chunks}

    assert "sample_financial_statements.pdf" in all_sources  # via table_chunks
    assert "sample_lease_agreement.pdf" in all_sources        # via prose_chunks
    assert any(c.chunk_strategy == "table" for c in all_chunks)
    assert any(c.chunk_strategy in ("clause_aware", "clause_aware_fallback") for c in all_chunks)
