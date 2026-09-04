"""
Tests for src/generation/generator.py

The actual Groq API call is mocked in all but the (optional, skipped-by-
default) live test at the bottom — we want to verify our prompt
construction, citation labeling, and error handling deterministically and
without spending API quota or requiring network access on every test run.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from src.generation.generator import Generator, GeneratedAnswer, _format_source_label, _build_context_block
from src.retrieval.retriever import RetrievalResult
from src.chunking.chunker import Chunk


def _make_result(source, page, text, clause_number=None, clause_title=None, score=0.9):
    metadata = {}
    if clause_number:
        metadata["clause_number"] = clause_number
    if clause_title:
        metadata["clause_title"] = clause_title
    chunk = Chunk(
        text=text, source=source, page_number=page,
        chunk_id="x", chunk_strategy="test", metadata=metadata,
    )
    return RetrievalResult(chunk=chunk, score=score)


def test_format_source_label_with_clause_metadata():
    result = _make_result("lease.pdf", 1, "rent text", clause_number="3", clause_title="RENT")
    label = _format_source_label(result, 1)
    assert label == "Source 1: lease.pdf, Clause 3 (RENT), page 1"


def test_format_source_label_without_clause_metadata():
    result = _make_result("10k.pdf", 2, "revenue text")
    label = _format_source_label(result, 2)
    assert label == "Source 2: 10k.pdf, page 2"


def test_format_source_label_with_table_metadata():
    chunk = Chunk(text="| Revenue | $4,820 |", source="financials.pdf", page_number=1,
                  chunk_id="x", chunk_strategy="table", metadata={"table_index": 1})
    result = RetrievalResult(chunk=chunk, score=0.9)
    label = _format_source_label(result, 1)
    assert label == "Source 1: financials.pdf, Table 1, page 1"


def test_format_source_label_with_table_heading_metadata():
    """
    Real-world finding (Phase 4): a document can have two structurally
    identical tables on the same page distinguished only by a heading
    like "GAAP" vs "Non-GAAP" — the citation must surface this, or two
    genuinely different (and both correct) figures look like an
    unexplained data conflict.
    """
    chunk = Chunk(text="| Net Income | $102,658 |", source="annualreport_2024.pdf", page_number=24,
                  chunk_id="x", chunk_strategy="table",
                  metadata={"table_index": 1, "table_heading": "GAAP"})
    result = RetrievalResult(chunk=chunk, score=0.9)
    label = _format_source_label(result, 1)
    assert label == "Source 1: annualreport_2024.pdf, Table 1 (GAAP), page 24"


def test_build_context_block_includes_all_sources_and_text():
    results = [
        _make_result("lease.pdf", 1, "Rent is $2,400.", clause_number="3", clause_title="RENT"),
        _make_result("loan.pdf", 1, "Interest rate is 6.75%.", clause_number="2", clause_title="INTEREST RATE"),
    ]
    block = _build_context_block(results)

    assert "Source 1: lease.pdf" in block
    assert "Rent is $2,400." in block
    assert "Source 2: loan.pdf" in block
    assert "Interest rate is 6.75%." in block


def test_generator_raises_clear_error_with_no_api_key(monkeypatch):
    # Explicitly patch config.GROQ_API_KEY to empty, rather than relying on
    # the ambient environment having no key set. Without this, this test's
    # correctness would depend on whether GROQ_API_KEY happens to be unset
    # wherever it runs — it should pass deterministically everywhere,
    # including on a machine with a real .env configured (which is exactly
    # the setup we want in production).
    monkeypatch.setattr("src.generation.generator.config.GROQ_API_KEY", "")

    with pytest.raises(ValueError, match="No Groq API key found"):
        Generator(api_key="")


def test_generate_short_circuits_on_empty_results_without_calling_llm():
    generator = Generator(api_key="fake_key_not_used")
    with patch.object(generator.client.chat.completions, "create") as mock_create:
        result = generator.generate("any question", [])

        mock_create.assert_not_called()
        assert isinstance(result, GeneratedAnswer)
        assert "don't contain enough information" in result.answer
        assert result.sources_used == []


def test_generate_calls_llm_with_grounded_prompt_and_returns_sources():
    generator = Generator(api_key="fake_key_not_used")
    results = [_make_result("lease.pdf", 1, "Rent is $2,400/month.", clause_number="3", clause_title="RENT")]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="The rent is $2,400/month (Source 1)."))]

    with patch.object(generator.client.chat.completions, "create", return_value=mock_response) as mock_create:
        result = generator.generate("What is the rent?", results)

        # Verify the LLM was called with our grounding system prompt and the context
        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "ONLY the context" in messages[0]["content"]
        assert "Rent is $2,400/month." in messages[1]["content"]

        assert result.answer == "The rent is $2,400/month (Source 1)."
        assert result.sources_used == ["Source 1: lease.pdf, Clause 3 (RENT), page 1"]


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Requires a real GROQ_API_KEY to hit the live API")
def test_live_generation_smoke_test():
    """
    Optional live test — only runs if GROQ_API_KEY is set in the
    environment. Confirms an actual API call succeeds and returns a
    sensible, grounded answer. Skipped in CI/sandbox environments without
    a key, which is why this isn't part of the required test count.
    """
    generator = Generator()
    results = [_make_result("lease.pdf", 1, "Tenant shall pay Landlord monthly rent of $2,400.00.",
                             clause_number="3", clause_title="RENT")]

    result = generator.generate("What is the monthly rent?", results)
    assert "2,400" in result.answer or "2400" in result.answer
