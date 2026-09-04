"""
Tests for src/ingestion/table_extractor.py
"""

from pathlib import Path

from src.ingestion.table_extractor import (
    extract_tables_from_pdf,
    extract_tables_from_dir,
    _table_to_markdown,
    _clean_cell,
)
from src.ingestion.loaders import load_pdf
from src.chunking.chunker import Chunk

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
FINANCIAL_PDF = SAMPLE_DIR / "sample_financial_statements.pdf"


# --- _clean_cell ---

def test_clean_cell_converts_none_to_empty_string():
    assert _clean_cell(None) == ""


def test_clean_cell_strips_whitespace():
    assert _clean_cell("  $4,820  ") == "$4,820"


# --- _table_to_markdown ---

def test_table_to_markdown_produces_valid_markdown_table():
    table = [["Item", "2024", "2023"], ["Revenue", "$100", "$90"]]
    markdown = _table_to_markdown(table)

    lines = markdown.split("\n")
    assert lines[0] == "| Item | 2024 | 2023 |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| Revenue | $100 | $90 |"


def test_table_to_markdown_handles_none_cells():
    table = [["Item", "2024"], ["Revenue", None]]
    markdown = _table_to_markdown(table)
    assert "| Revenue |  |" in markdown


def test_table_to_markdown_returns_empty_string_for_empty_table():
    assert _table_to_markdown([]) == ""
    assert _table_to_markdown([[]]) == ""


def test_table_to_markdown_pads_short_rows_defensively():
    table = [["A", "B", "C"], ["1", "2"]]  # second row missing a cell
    markdown = _table_to_markdown(table)
    # Should not crash, and should pad the short row rather than misalign columns.
    assert "| 1 | 2 |  |" in markdown


# --- extract_tables_from_pdf: real end-to-end extraction ---

def test_extract_tables_finds_both_tables_in_sample_pdf():
    chunks = extract_tables_from_pdf(FINANCIAL_PDF)
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.chunk_strategy == "table" for c in chunks)


def test_extract_tables_preserves_row_column_alignment():
    """
    The core correctness property of table extraction: a specific line
    item's value for a specific year must be traceable, not scrambled.
    This directly tests the failure mode pdfplumber is meant to avoid.
    """
    chunks = extract_tables_from_pdf(FINANCIAL_PDF)
    income_statement = next(c for c in chunks if "Net revenue" in c.text)

    assert "| Net revenue | $3,150 | $2,780 |" in income_statement.text


def test_extract_tables_second_table_is_balance_sheet():
    chunks = extract_tables_from_pdf(FINANCIAL_PDF)
    balance_sheet = next(c for c in chunks if "Total assets" in c.text)

    assert "| Total assets | $3,050 | $2,650 |" in balance_sheet.text
    assert balance_sheet.metadata["table_index"] == 2


def test_extract_tables_chunk_metadata_has_no_clause_number():
    """
    Table chunks have no clause_number (they're not numbered-clause legal
    text) — this matters for citation formatting downstream, which should
    fall back to "Table N, page P" rather than a clause reference.
    """
    chunks = extract_tables_from_pdf(FINANCIAL_PDF)
    assert all("clause_number" not in c.metadata for c in chunks)
    assert all(c.metadata["file_type"] == "pdf_table" for c in chunks)


def test_extract_tables_caption_includes_line_item_names():
    """
    The caption enrichment (Phase 3 fix): a bare "Table 1, page 1" caption
    gives an embedding model almost no natural-language signal to match
    against a question. We discovered concretely that this caused a
    prose sentence elsewhere in the corpus to outrank the correct table
    for a net-income question, purely on lexical overlap. Captions must
    include the table's actual line-item names so retrieval has real
    semantic signal to work with.
    """
    chunks = extract_tables_from_pdf(FINANCIAL_PDF)
    income_statement = next(c for c in chunks if "Net revenue" in c.text)
    caption_line = income_statement.text.split("\n")[0]

    assert "includes:" in caption_line
    assert "Net revenue" in caption_line  # an early line item, within the capped label list


def test_extract_tables_row_label_summary_is_capped_for_large_tables():
    """
    Phase 4 finding (from testing against a real annual report): an
    uncapped label list can grow long enough to consume the UI's entire
    excerpt-length budget, cutting off before the actual table numbers
    become visible. Our sample income statement has 9 line items —
    the caption should cap at 6 and note how many more exist, while the
    full markdown table (checked separately) still contains every row.
    """
    chunks = extract_tables_from_pdf(FINANCIAL_PDF)
    income_statement = next(c for c in chunks if "Net revenue" in c.text)
    caption_line = income_statement.text.split("\n")[0]

    assert "and 3 more" in caption_line  # 9 line items, capped at 6 -> 3 remaining
    # Net income (line item #9) is capped out of the SUMMARY caption, but
    # must still be fully present in the actual markdown table body below it.
    assert "Net income" not in caption_line
    assert "| Net income |" in income_statement.text


def test_extract_tables_from_dir_finds_tables_across_pdfs():
    chunks = extract_tables_from_dir(SAMPLE_DIR)
    sources = {c.source for c in chunks}
    # Only the financial statements PDF actually contains real tables;
    # the lease/loan/10-K sample docs are prose-only.
    assert sources == {"sample_financial_statements.pdf"}


# --- _find_heading_above_table: real-world GAAP/Non-GAAP finding (Phase 4) ---

def test_find_heading_above_table_detects_short_label():
    from src.ingestion.table_extractor import _find_heading_above_table
    from unittest.mock import MagicMock

    mock_page = MagicMock()
    mock_page.width = 600
    mock_cropped = MagicMock()
    mock_cropped.extract_text.return_value = "Some prior paragraph text.\nGAAP"
    mock_page.crop.return_value = mock_cropped

    heading = _find_heading_above_table(mock_page, (0, 300, 500, 450), region_top=0)
    assert heading == "GAAP"


def test_find_heading_above_table_strips_footnote_digit():
    """
    pdfplumber sometimes glues a footnote-marker digit onto a heading word
    (observed directly in a real annual report: "Non-GAAP1" for a
    footnoted "Non-GAAP" heading) — this must be cleaned up.
    """
    from src.ingestion.table_extractor import _find_heading_above_table
    from unittest.mock import MagicMock

    mock_page = MagicMock()
    mock_page.width = 600
    mock_cropped = MagicMock()
    mock_cropped.extract_text.return_value = "Net cash provided by operations\nNon-GAAP1"
    mock_page.crop.return_value = mock_cropped

    heading = _find_heading_above_table(mock_page, (0, 500, 500, 650), region_top=300)
    assert heading == "Non-GAAP"


def test_find_heading_above_table_rejects_full_sentences():
    """
    A long line ending like a sentence just above a table is ordinary
    prose, not a caption — must NOT be treated as a heading, or every
    table following a paragraph would get a meaningless "heading".
    """
    from src.ingestion.table_extractor import _find_heading_above_table
    from unittest.mock import MagicMock

    mock_page = MagicMock()
    mock_page.width = 600
    mock_cropped = MagicMock()
    mock_cropped.extract_text.return_value = "Here is a snapshot of our full-year performance for the period."
    mock_page.crop.return_value = mock_cropped

    heading = _find_heading_above_table(mock_page, (0, 300, 500, 450), region_top=0)
    assert heading == ""


def test_find_heading_above_table_returns_empty_when_table_is_at_region_top():
    """If the table starts right at the search region's top (no gap above it), there's nothing to search."""
    from src.ingestion.table_extractor import _find_heading_above_table
    from unittest.mock import MagicMock

    mock_page = MagicMock()
    heading = _find_heading_above_table(mock_page, (0, 100, 500, 250), region_top=100)
    assert heading == ""
    mock_page.crop.assert_not_called()


# --- _merge_currency_symbol_columns: real-world finding (Phase 4) ---

def test_merge_currency_symbol_columns_merges_dollar_sign_into_value():
    """
    This is the exact structure pdfplumber produced from a real annual
    report table: the currency symbol sits in its own column, separate
    from the numeric value, for every data row.
    """
    from src.ingestion.table_extractor import _merge_currency_symbol_columns
    raw_table = [
        ["", "", "2024", "", "2023"],
        ["Revenue", "$", "1,330,383", "$", "2,290,786"],
    ]
    merged = _merge_currency_symbol_columns(raw_table)

    assert merged[0] == ["", "2024", "2023"]
    assert merged[1] == ["Revenue", "$1,330,383", "$2,290,786"]


def test_merge_currency_symbol_columns_handles_rows_without_dollar_signs():
    """
    A row like "Gross Margin" has empty placeholders (not "$") in the
    same column position as dollar-figure rows — the merge must still
    treat that column as a symbol column (since '' counts as "no symbol
    present", not "a real value"), producing a clean percentage cell.
    """
    from src.ingestion.table_extractor import _merge_currency_symbol_columns
    raw_table = [
        ["", "", "2024", "", "2023"],
        ["Revenue", "$", "1,330,383", "$", "2,290,786"],
        ["Gross Margin", "", "47.3 %", "", "46.2 %"],
    ]
    merged = _merge_currency_symbol_columns(raw_table)

    assert merged[2] == ["Gross Margin", "47.3 %", "46.2 %"]


def test_merge_currency_symbol_columns_leaves_normal_tables_unchanged():
    """
    Our own synthetic test PDFs (Phase 3) already produce single-cell
    "$3,150" style values via reportlab — the merge function must be a
    complete no-op for tables that were never split this way.
    """
    from src.ingestion.table_extractor import _merge_currency_symbol_columns
    raw_table = [
        ["Line Item", "FY2024", "FY2023"],
        ["Net revenue", "$3,150", "$2,780"],
    ]
    merged = _merge_currency_symbol_columns(raw_table)
    assert merged == raw_table


def test_merge_currency_symbol_columns_does_not_merge_a_column_with_real_mixed_values():
    """
    A column should only be merged if EVERY non-empty value in it is a
    bare currency symbol — a column containing a real value anywhere
    must never be merged away, even if some rows in it happen to be empty.
    """
    from src.ingestion.table_extractor import _merge_currency_symbol_columns
    raw_table = [
        ["Label", "Note", "Value"],
        ["Row A", "see note 1", "$100"],
        ["Row B", "", "$200"],
    ]
    merged = _merge_currency_symbol_columns(raw_table)
    assert merged == raw_table  # "Note" column has a real value ("see note 1"), must not be merged


def test_merge_currency_symbol_columns_handles_empty_table():
    from src.ingestion.table_extractor import _merge_currency_symbol_columns
    assert _merge_currency_symbol_columns([]) == []
    assert _merge_currency_symbol_columns([[]]) == [[]]


# --- Demonstrates WHY table extraction matters (pypdf vs pdfplumber) ---

def test_pypdf_flat_extraction_loses_row_column_structure():
    """
    This test doesn't assert anything about OUR code — it documents and
    verifies the actual problem table extraction solves. pypdf's flat text
    extraction turns each table cell into its own line with no structural
    marker connecting a line item to its value, unlike pdfplumber's
    row-aware extraction (verified in test_extract_tables_preserves_row_column_alignment).
    """
    pages = load_pdf(FINANCIAL_PDF)
    flat_text = pages[0].text

    # The values are present in the flat text...
    assert "Net revenue" in flat_text
    assert "$3,150" in flat_text
    # ...but NOT in a structurally connected form — pypdf never produces
    # "Net revenue | $3,150", since it has no concept of table cells.
    assert "Net revenue | $3,150" not in flat_text
    assert "Net revenue\n$3,150" in flat_text or "Net revenue $3,150" in flat_text.replace("\n", " ")
