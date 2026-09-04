"""
Tests for app/styles.py

render_source_chips() is a pure function (no Streamlit calls), so it's
tested directly on its HTML output string — no browser or Streamlit
runtime needed.
"""

from app.styles import render_source_chips, _escape


def test_render_source_chips_wraps_each_source_in_a_chip_span():
    html = render_source_chips(["Source 1: lease.pdf, Clause 3 (RENT), page 1"])
    assert html.count('<span class="source-chip">') == 1
    assert "Source 1: lease.pdf, Clause 3 (RENT), page 1" in html


def test_render_source_chips_handles_multiple_sources():
    sources = ["Source 1: a.pdf, page 1", "Source 2: b.pdf, page 2"]
    html = render_source_chips(sources)
    assert html.count('<span class="source-chip">') == 2
    assert "a.pdf" in html
    assert "b.pdf" in html


def test_render_source_chips_handles_empty_list_gracefully():
    html = render_source_chips([])
    assert "No sources available" in html
    assert "<span" not in html  # no empty/broken chip markup


def test_render_source_chips_escapes_html_special_characters():
    """
    Source strings ultimately derive from document filenames/content —
    escaping prevents a filename containing < or & from breaking the
    rendered HTML structure.
    """
    html = render_source_chips(["Source 1: weird<file>&name.pdf, page 1"])
    assert "<file>" not in html
    assert "&lt;file&gt;" in html
    assert "&amp;name" in html


def test_escape_handles_plain_text_unchanged():
    assert _escape("Clause 3 (RENT), page 1") == "Clause 3 (RENT), page 1"


# --- safe_markdown_text ---

def test_safe_markdown_text_escapes_single_dollar_sign():
    from app.styles import safe_markdown_text
    assert safe_markdown_text("The rent is $2,400.") == "The rent is \\$2,400."


def test_safe_markdown_text_escapes_multiple_dollar_signs():
    """
    This is the actual bug scenario: two or more literal dollar signs in
    one string get interpreted by st.markdown() as LaTeX math delimiters,
    garbling the text between them. Both must be escaped.
    """
    from app.styles import safe_markdown_text
    text = "Revenue declined from $2.29 billion in 2023 to $1.33 billion in 2024."
    result = safe_markdown_text(text)
    assert result.count("\\$") == 2
    assert result == "Revenue declined from \\$2.29 billion in 2023 to \\$1.33 billion in 2024."


def test_safe_markdown_text_leaves_text_without_dollar_signs_unchanged():
    from app.styles import safe_markdown_text
    text = "No pets are permitted without written consent."
    assert safe_markdown_text(text) == text


# --- render_evidence_panel ---

def test_render_evidence_panel_includes_label_and_text():
    from app.styles import render_evidence_panel
    evidence = [{"label": "Source 1: lease.pdf, Clause 3 (RENT), page 1", "text": "Tenant shall pay $2,400/month."}]
    html = render_evidence_panel(evidence)

    assert 'class="evidence-block"' in html
    assert "Source 1: lease.pdf, Clause 3 (RENT), page 1" in html
    assert "Tenant shall pay" in html


def test_render_evidence_panel_handles_multiple_sources():
    from app.styles import render_evidence_panel
    evidence = [
        {"label": "Source 1: a.pdf, page 1", "text": "First excerpt."},
        {"label": "Source 2: b.pdf, page 2", "text": "Second excerpt."},
    ]
    html = render_evidence_panel(evidence)

    assert html.count('class="evidence-block"') == 2
    assert "First excerpt." in html
    assert "Second excerpt." in html


def test_render_evidence_panel_handles_empty_list():
    from app.styles import render_evidence_panel
    html = render_evidence_panel([])
    assert "No evidence available" in html


def test_render_evidence_panel_truncates_long_excerpts_at_word_boundary():
    from app.styles import render_evidence_panel
    long_text = "word " * 200  # far longer than the default 400-char limit
    evidence = [{"label": "Source 1: doc.pdf, page 1", "text": long_text}]

    html = render_evidence_panel(evidence, max_excerpt_length=50)

    assert "…" in html
    # Should not cut off mid-word — the truncated text should end cleanly
    # before the ellipsis, not with a fragment of "word".
    assert "wor…" not in html


def test_render_evidence_panel_escapes_html_in_excerpt_text():
    from app.styles import render_evidence_panel
    evidence = [{"label": "Source 1: doc.pdf, page 1", "text": "Contains <script>alert('x')</script> text."}]
    html = render_evidence_panel(evidence)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# --- extract_cited_source_numbers ---

def test_extract_cited_source_numbers_finds_single_citation():
    from app.styles import extract_cited_source_numbers
    answer = "The rent is $2,400/month (Source 1)."
    assert extract_cited_source_numbers(answer) == {1}


def test_extract_cited_source_numbers_finds_multiple_citations():
    from app.styles import extract_cited_source_numbers
    answer = "Revenue was $1.33 billion (Source 4), down from $2.29 billion (Source 2)."
    assert extract_cited_source_numbers(answer) == {2, 4}


def test_extract_cited_source_numbers_returns_empty_set_when_no_citations():
    from app.styles import extract_cited_source_numbers
    answer = "The retrieved documents don't contain enough information to answer this question."
    assert extract_cited_source_numbers(answer) == set()


def test_extract_cited_source_numbers_deduplicates_repeated_citations():
    from app.styles import extract_cited_source_numbers
    answer = "The deposit is $2,400 (Source 1), returned within 30 days (Source 1)."
    assert extract_cited_source_numbers(answer) == {1}


# --- render_evidence_panel: used vs. other retrieved evidence ---

def test_render_evidence_panel_marks_cited_source_as_used():
    from app.styles import render_evidence_panel
    evidence = [
        {"label": "Source 1: a.pdf, page 1", "text": "First."},
        {"label": "Source 2: b.pdf, page 2", "text": "Second."},
    ]
    html = render_evidence_panel(evidence, cited_indices={2})

    # Source 2 (cited) should be marked used; Source 1 should be grouped
    # under "other retrieved evidence" and marked unused.
    assert "★ used in answer" in html
    assert "Other retrieved evidence" in html
    assert 'class="evidence-block unused"' in html


def test_render_evidence_panel_puts_used_sources_before_others():
    from app.styles import render_evidence_panel
    evidence = [
        {"label": "Source 1: a.pdf, page 1", "text": "First excerpt marker AAA."},
        {"label": "Source 2: b.pdf, page 2", "text": "Second excerpt marker BBB."},
    ]
    html = render_evidence_panel(evidence, cited_indices={2})

    assert html.index("BBB") < html.index("AAA")  # Source 2 (used) appears before Source 1 (other)


def test_render_evidence_panel_with_no_cited_indices_shows_flat_list():
    """Backward-compatible behavior: omitting cited_indices renders everything as before, no grouping."""
    from app.styles import render_evidence_panel
    evidence = [{"label": "Source 1: a.pdf, page 1", "text": "Some text."}]
    html = render_evidence_panel(evidence)

    assert "Other retrieved evidence" not in html
    assert "★ used in answer" not in html


def test_render_evidence_panel_all_sources_cited_has_no_other_section():
    from app.styles import render_evidence_panel
    evidence = [{"label": "Source 1: a.pdf, page 1", "text": "Some text."}]
    html = render_evidence_panel(evidence, cited_indices={1})

    assert "Other retrieved evidence" not in html
    assert "★ used in answer" in html
