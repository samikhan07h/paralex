"""
Tests for evaluation/test_sets_loader.py

The most important test here isn't schema validation — it's
test_all_expected_clauses_actually_exist_in_chunked_documents. Ground truth
that references a clause number that doesn't actually exist in the real
chunked output would silently break every retrieval metric downstream
(Day 2) without ever raising an error, since "clause not found" and
"clause found but not retrieved" look identical to a naive metric. This
test catches that class of bug at the source, in the eval data itself.
"""

import pytest
from pathlib import Path

from evaluation.test_sets_loader import load_all_test_sets, EvalItem, _load_and_validate
from src.ingestion.loaders import load_documents_from_dir
from src.chunking.chunker import clause_aware_chunk

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_load_all_test_sets_returns_eval_items():
    items = load_all_test_sets()
    assert len(items) > 0
    assert all(isinstance(item, EvalItem) for item in items)


def test_all_three_document_test_sets_are_represented():
    items = load_all_test_sets()
    sources = {item.expected_source_doc for item in items}

    assert "sample_lease_agreement.pdf" in sources
    assert "sample_loan_agreement.pdf" in sources
    assert "sample_10k_excerpt.pdf" in sources


def test_all_eval_item_ids_are_unique():
    items = load_all_test_sets()
    ids = [item.id for item in items]
    assert len(ids) == len(set(ids))


def test_every_item_has_at_least_one_expected_keyword():
    items = load_all_test_sets()
    assert all(len(item.expected_keywords) >= 1 for item in items)


def test_missing_required_field_raises_clear_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('[{"id": "x1", "question": "some question?"}]')  # missing most fields

    with pytest.raises(ValueError, match="missing required field"):
        _load_and_validate(bad_file)


def test_all_expected_clauses_actually_exist_in_chunked_documents():
    """
    Cross-check every EvalItem's expected_clause_number against the REAL
    output of clause_aware_chunk() on the actual sample documents. This
    guards against a typo'd ground-truth clause number silently making
    every retrieval metric look worse (or, worse, artificially better)
    than it really is.
    """
    items = load_all_test_sets()

    pages = load_documents_from_dir(SAMPLE_DIR)
    chunks = clause_aware_chunk(pages)

    # Build a lookup of (source_doc, clause_number) -> exists
    real_clause_pairs = {
        (c.source, c.metadata.get("clause_number"))
        for c in chunks
        if c.metadata.get("clause_number") is not None
    }

    for item in items:
        if item.expected_clause_number is None:
            # Documents without numbered clauses (e.g. the 10-K) are expected
            # to have expected_clause_number=None — nothing to cross-check.
            continue

        pair = (item.expected_source_doc, item.expected_clause_number)
        assert pair in real_clause_pairs, (
            f"Eval item '{item.id}' references clause {item.expected_clause_number} "
            f"in {item.expected_source_doc}, but no such clause was found in the "
            f"actual chunked output. Check for a typo in the test set JSON."
        )


def test_every_expected_keyword_appears_somewhere_in_the_correct_source_document():
    """
    Sanity check the ground truth itself: every expected_keyword should
    actually be findable in the source document's text. If a keyword typo
    slips into a test set (e.g. "$2,4000" instead of "$2,400"), this test
    catches it before it silently causes faithfulness checks to fail for
    the wrong reason in Day 3.
    """
    items = load_all_test_sets()
    pages = load_documents_from_dir(SAMPLE_DIR)

    text_by_source = {}
    for page in pages:
        text_by_source.setdefault(page.source, "")
        text_by_source[page.source] += page.text

    for item in items:
        source_text = text_by_source.get(item.expected_source_doc, "")
        for keyword in item.expected_keywords:
            assert keyword in source_text, (
                f"Eval item '{item.id}' expects keyword '{keyword}' to appear in "
                f"{item.expected_source_doc}, but it wasn't found in the document text."
            )
