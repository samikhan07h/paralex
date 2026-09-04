"""
Table extraction for ParaLex.

WHY A SEPARATE MODULE FROM loaders.py:
loaders.py's job is to extract PROSE text page-by-page — it deliberately
stays format-agnostic and simple. Table extraction is a genuinely
different problem (detecting a grid structure, not just reading a text
stream) requiring a different library and producing a different kind of
output (structured rows/columns, not a flat string). Keeping these
concerns in separate modules means each one stays focused and testable in
isolation, and either can evolve independently — e.g. swapping the table
detection library later wouldn't touch prose extraction at all.

WHY pdfplumber INSTEAD OF pypdf FOR TABLES:
pypdf's extract_text() flattens every piece of text on a page into a
single stream in roughly reading order, with no awareness that some of
that text was visually arranged into a table's rows and columns. Run on a
financial statement, this turns "Net revenue | $4,820 | $4,330" into three
disconnected lines — "Net revenue", then "$4,820", then "$4,330" — with
nothing in the extracted text indicating which dollar figure belongs to
which fiscal year. If chunked naively, a chunk boundary could easily land
between a line item and its own values, silently producing wrong answers
downstream. pdfplumber's extract_tables() instead detects a table's actual
grid lines in the PDF's structure and returns properly aligned rows and
columns, preserving exactly the row/column relationship a table's meaning
depends on.

WHY EACH TABLE BECOMES ONE MARKDOWN-FORMATTED CHUNK (chunk_strategy="table"):
Markdown table syntax (`| col | col |`) is a format LLMs are extensively
trained on and reliably parse correctly when reasoning over — it's a
better fit here than either raw nested lists (which need custom formatting
instructions in the prompt) or flattening cells into prose (which would
reintroduce the exact row/column ambiguity we're trying to avoid). Keeping
each table as a single chunk (rather than splitting rows into separate
chunks) preserves the full header-to-value context a question about the
table would need.

WHY WE DETECT A HEADING ABOVE EACH TABLE (real-world finding, Phase 4):
Testing against a real annual report surfaced a genuine problem our
synthetic test PDF never exercised: many real financial documents place
TWO side-by-side tables labeled only by a short heading sitting above the
table's grid — e.g. "GAAP" above one results table and "Non-GAAP" above a
second, otherwise identically-shaped table just below it on the same page.
pdfplumber's table detection only sees the grid itself, not the heading
text floating above it, so without special handling, both tables end up
labeled as indistinguishable "Table 1" / "Table 2" — meaning a genuinely
different, correctly-extracted GAAP net income figure and Non-GAAP net
income figure look like an unexplained data conflict to anyone (or any
LLM) reading the citations, when they're actually two different, both-
correct numbers that just need the right label attached. We fix this by
looking at the text in the page region immediately above each table's
bounding box and, if it's short and label-like (not a full sentence),
treating it as the table's heading.
"""

import re
from pathlib import Path
from typing import List, Optional

import pdfplumber

from src.chunking.chunker import Chunk

# A candidate heading line must be short and not end like a sentence to be
# treated as a table label rather than the tail end of ordinary prose that
# happens to sit just above the table on the page.
_MAX_HEADING_LENGTH = 40

# Bare currency symbols that real-world PDF tables frequently typeset in
# their own separate grid column (right-aligned apart from the numeric
# value) — see _merge_currency_symbol_columns's docstring.
_CURRENCY_SYMBOLS = {"$", "€", "£", "¥"}


def _clean_cell(value) -> str:
    """pdfplumber returns None for empty cells; normalize to an empty string."""
    if value is None:
        return ""
    return str(value).strip()


def _merge_currency_symbol_columns(table: List[List]) -> List[List]:
    """
    Merge PDF table columns that consist ONLY of a bare currency symbol
    (e.g. "$") into the next column, producing "$1,330,383" instead of two
    separate cells "$" | "1,330,383".

    WHY THIS EXISTS (a real finding from testing against an actual annual
    report, not a hypothetical): many real financial tables are typeset
    with the currency symbol in its own column, right-aligned separately
    from the numeric value — a common accounting/typesetting convention.
    pdfplumber faithfully preserves this as a genuinely separate grid
    column. Without merging, this produces a markdown table riddled with
    confusing near-empty "ghost" columns (e.g. a header row rendering as
    "| | | 2024 | | 2023 |") and splits every dollar figure into two
    separate cells — actively hurting both the LLM's ability to read the
    correct figure and the evidence panel's readability for a human.

    Detection is column-level, not row-level: a column only qualifies for
    merging if EVERY non-empty value in that column across the whole table
    is a bare currency symbol — this avoids accidentally merging a column
    that happens to contain a real "$" value mixed with other content.
    """
    if not table or not table[0]:
        return table

    num_cols = max(len(row) for row in table)
    cols_to_merge = set()

    for col_idx in range(num_cols - 1):  # never merge the very last column forward
        col_values = [_clean_cell(row[col_idx]) if col_idx < len(row) else "" for row in table]
        non_empty = [v for v in col_values if v]
        if non_empty and all(v in _CURRENCY_SYMBOLS for v in non_empty):
            cols_to_merge.add(col_idx)

    if not cols_to_merge:
        return table

    merged_table = []
    for row in table:
        new_row = []
        skip_next = False
        for col_idx, cell in enumerate(row):
            if skip_next:
                skip_next = False
                continue
            cell_clean = _clean_cell(cell)
            if col_idx in cols_to_merge:
                next_cell = _clean_cell(row[col_idx + 1]) if col_idx + 1 < len(row) else ""
                new_row.append(f"{cell_clean}{next_cell}" if cell_clean and next_cell else (next_cell or cell_clean))
                skip_next = True
            else:
                new_row.append(cell)
        merged_table.append(new_row)

    return merged_table


def _table_to_markdown(table: List[List]) -> str:
    """
    Convert a pdfplumber-extracted table (list of rows, each a list of
    cell values) into a markdown table string. Assumes the first row is
    the header row, which is the standard convention for financial
    statement tables (and pdfplumber's typical table shape).
    """
    if not table or not table[0]:
        return ""

    header = [_clean_cell(c) for c in table[0]]
    data_rows = [[_clean_cell(c) for c in row] for row in table[1:]]

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in data_rows:
        # Defensively pad/truncate rows that don't match the header length
        # (can happen with slightly irregular table detection) rather than
        # crashing on a malformed row.
        padded = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def _summarize_row_labels(table: List[List], max_labels: int = 6) -> str:
    """
    Build a natural-language summary of a table's row labels (its first
    column), e.g. "Net revenue, Cost of revenue, Gross profit, ... Net income".

    WHY THIS EXISTS (a real Phase 3 finding):
    A bare markdown table like "| Net income | $645 | $560 |" has very
    little natural-language overlap with a question like "What was the
    net income for fiscal 2024?" — embedding models are trained
    predominantly on prose, and a terse pipe-delimited grid doesn't give
    them much lexical signal to match against. Prepending the table's
    actual line-item names as a natural-language caption gives the
    embedding real semantic signal to match against, without altering the
    underlying markdown grid the LLM reads for exact figures.

    WHY THE LABEL LIST IS CAPPED (a real Phase 4 finding, from testing
    against an actual annual report rather than only our small synthetic
    test table): a real-world financial table can have far more line
    items than our test fixtures did. An uncapped label list can grow long
    enough that, combined with the UI's excerpt-length truncation
    (app/styles.py's render_evidence_panel), the caption alone consumes
    the entire display budget — cutting off before the actual markdown
    table (with the real numbers) ever becomes visible to the user. Capping
    the label list keeps the caption bounded regardless of table size.
    """
    if len(table) < 2:
        return ""
    row_labels = [_clean_cell(row[0]) for row in table[1:] if row and row[0]]
    row_labels = [label for label in row_labels if label]

    if len(row_labels) > max_labels:
        remaining = len(row_labels) - max_labels
        shown = row_labels[:max_labels]
        return ", ".join(shown) + f", and {remaining} more"
    return ", ".join(row_labels)


def _find_heading_above_table(page, table_bbox: tuple, region_top: float) -> str:
    """
    Look for a short heading/label line immediately above a table's
    bounding box on the page (e.g. "GAAP" or "Non-GAAP" preceding a
    financial results table).

    `region_top` bounds the search from above — it's the bottom edge of
    the previous table on this page (or 0 for the first table), so we
    never accidentally attribute text belonging to an earlier table or
    an unrelated paragraph further up the page as this table's heading.

    Returns an empty string if no short, label-like text is found — this
    is a deliberately conservative heuristic that only activates for
    genuinely caption-shaped text, not any word that happens to appear
    above a table.
    """
    x0, top, x1, bottom = table_bbox
    if top <= region_top:
        return ""

    try:
        cropped = page.crop((0, region_top, page.width, top))
        text = cropped.extract_text() or ""
    except Exception:
        return ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    candidate = lines[-1]  # the line closest to the table's top edge
    if len(candidate) > _MAX_HEADING_LENGTH or candidate.endswith((".", ",", ";", ":")):
        return ""

    # pdfplumber sometimes glues a footnote-marker digit onto the end of a
    # heading word (e.g. "Non-GAAP1" for a footnoted "Non-GAAP" heading) —
    # strip a trailing digit run immediately after a letter.
    candidate = re.sub(r"(?<=[A-Za-z])\d+$", "", candidate)
    return candidate


def extract_tables_from_pdf(file_path: str | Path) -> List[Chunk]:
    """
    Extract every table found in a PDF, returning one Chunk per table with
    chunk_strategy="table" and its markdown representation as the chunk text.

    Tables with fewer than 2 rows (header only, or a detection false
    positive) are skipped, since a table with no data rows carries no
    retrievable information.
    """
    file_path = Path(file_path)
    chunks: List[Chunk] = []

    with pdfplumber.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            found_tables = page.find_tables()
            region_top = 0.0  # tracks how far down the page we've already searched for headings

            for table_idx, table_obj in enumerate(found_tables, start=1):
                table = table_obj.extract()

                if not table or len(table) < 2:
                    region_top = table_obj.bbox[3]
                    continue

                # Merge separated currency-symbol columns before anything
                # else touches the table — this must happen first so both
                # the markdown body and the row-label summary see clean,
                # already-merged cells (e.g. "$1,330,383" instead of two
                # cells "$" and "1,330,383").
                table = _merge_currency_symbol_columns(table)

                markdown = _table_to_markdown(table)
                if not markdown.strip():
                    region_top = table_obj.bbox[3]
                    continue

                heading = _find_heading_above_table(page, table_obj.bbox, region_top)
                region_top = table_obj.bbox[3]  # advance past this table for the next one

                # A caption naming the table's actual line items (not just
                # "Table N, page P") gives the embedding model real
                # natural-language signal to match against a question —
                # see _summarize_row_labels()'s docstring. The detected
                # heading (e.g. "GAAP"/"Non-GAAP"), when present, is what
                # lets two structurally-identical tables on the same page
                # be told apart instead of looking like an unexplained
                # data conflict.
                row_summary = _summarize_row_labels(table)
                caption = f"Table {table_idx}"
                if heading:
                    caption += f" ({heading})"
                caption += f" (page {page_num} of {file_path.name})"
                if row_summary:
                    caption += f" — includes: {row_summary}."
                else:
                    caption += ":"
                chunk_text = f"{caption}\n{markdown}"

                chunks.append(Chunk(
                    text=chunk_text,
                    source=file_path.name,
                    page_number=page_num,
                    chunk_id=f"{file_path.name}_p{page_num}_table{table_idx}",
                    chunk_strategy="table",
                    metadata={
                        "table_index": table_idx,
                        "file_type": "pdf_table",
                        "table_heading": heading or None,
                    },
                ))

    return chunks


def extract_tables_from_dir(dir_path: str | Path) -> List[Chunk]:
    """Extract tables from every PDF in a directory (non-recursive)."""
    dir_path = Path(dir_path)
    all_chunks: List[Chunk] = []

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            all_chunks.extend(extract_tables_from_pdf(file_path))

    return all_chunks
