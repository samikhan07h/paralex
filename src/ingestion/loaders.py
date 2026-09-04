"""
Document loaders for ParaLex.

WHY A SEPARATE LOADERS MODULE:
Legal/financial documents arrive as PDFs (contracts, 10-Ks) or occasionally
DOCX (draft agreements). Each format needs different extraction logic, but
the rest of the pipeline (chunking, embedding, retrieval) shouldn't care
which format the text came from. This module's job is to normalize any
input document into a consistent internal representation: a list of
per-page text blocks with metadata attached.

We attach metadata (source filename, page number) at extraction time —
NOT later — because retrofitting citations after chunking is painful.
Every downstream chunk inherits this metadata, which is what makes
Phase 3's "source citation" feature possible without rework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader
from docx import Document as DocxDocument


@dataclass
class PageContent:
    """One page (or page-equivalent block) of extracted text plus metadata."""

    text: str
    source: str          # original filename, e.g. "sample_lease.pdf"
    page_number: int      # 1-indexed page number (or block index for docx)
    metadata: dict = field(default_factory=dict)


def load_pdf(file_path: str | Path) -> List[PageContent]:
    """
    Extract text from a PDF, page by page.

    We keep page-level granularity (rather than joining the whole PDF into
    one string) because:
      - It lets us cite "page 4 of lease.pdf" later.
      - Legal documents are often referenced by page/section in practice,
        so this matches how a lawyer or analyst would expect answers to be
        grounded.
    """
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))
    pages: List[PageContent] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            # Skip genuinely empty pages (e.g. blank separator pages) but
            # don't silently skip pages with sparse-but-real content.
            continue
        pages.append(
            PageContent(
                text=text,
                source=file_path.name,
                page_number=i,
                metadata={"file_type": "pdf", "total_pages": len(reader.pages)},
            )
        )
    return pages


def load_docx(file_path: str | Path) -> List[PageContent]:
    """
    Extract text from a DOCX file.

    DOCX has no native "page" concept in the underlying XML (pagination is
    a rendering-time concern), so we treat each non-empty paragraph run as
    a block and assign it a sequential block number instead of a page
    number. This keeps the PageContent interface consistent across loaders
    even though the semantics of "page_number" differ slightly by format.
    """
    file_path = Path(file_path)
    doc = DocxDocument(str(file_path))
    blocks: List[PageContent] = []

    block_index = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        block_index += 1
        blocks.append(
            PageContent(
                text=text,
                source=file_path.name,
                page_number=block_index,
                metadata={"file_type": "docx"},
            )
        )
    return blocks


def load_document(file_path: str | Path) -> List[PageContent]:
    """
    Dispatch to the correct loader based on file extension.

    This is the single entry point the rest of the pipeline should call —
    callers shouldn't need to know or care whether a document is a PDF or
    DOCX.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for '{file_path.name}'. "
            f"Supported types: .pdf, .docx"
        )


def load_documents_from_dir(dir_path: str | Path) -> List[PageContent]:
    """
    Load every supported document in a directory (non-recursive).
    Used by the pipeline to bulk-ingest data/raw/ or data/sample_docs/.
    """
    dir_path = Path(dir_path)
    all_pages: List[PageContent] = []

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in (".pdf", ".docx"):
            all_pages.extend(load_document(file_path))

    return all_pages
