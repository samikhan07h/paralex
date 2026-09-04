"""
Tests for src/ingestion/loaders.py

These are intentionally lightweight (no mocking framework) since the goal
of Phase 1 tests is to catch regressions in extraction behavior, not to
achieve exhaustive coverage.
"""

import pytest
from pathlib import Path

from src.ingestion.loaders import (
    load_document,
    load_pdf,
    load_documents_from_dir,
    PageContent,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_load_pdf_returns_page_content_objects():
    pdf_path = SAMPLE_DIR / "sample_lease_agreement.pdf"
    pages = load_pdf(pdf_path)

    assert len(pages) > 0
    assert all(isinstance(p, PageContent) for p in pages)
    assert pages[0].source == "sample_lease_agreement.pdf"
    assert pages[0].page_number == 1
    assert "Lease Agreement" in pages[0].text


def test_load_document_dispatches_by_extension():
    pdf_path = SAMPLE_DIR / "sample_loan_agreement.pdf"
    pages = load_document(pdf_path)
    assert len(pages) > 0
    assert pages[0].metadata["file_type"] == "pdf"


def test_load_document_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("just some text")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(bad_file)


def test_load_documents_from_dir_loads_all_sample_docs():
    pages = load_documents_from_dir(SAMPLE_DIR)
    sources = {p.source for p in pages}

    assert "sample_lease_agreement.pdf" in sources
    assert "sample_loan_agreement.pdf" in sources
    assert "sample_10k_excerpt.pdf" in sources
