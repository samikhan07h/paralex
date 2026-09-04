"""
Chunking strategies for ParaLex.

WHY TWO STRATEGIES:
Legal and financial documents are structurally different from generic prose
(blog posts, articles) that most RAG tutorials are built around. They have
explicit numbered structure — "1. PARTIES.", "Section 4.2", "ITEM 7." — that
carries real semantic meaning: a clause is a self-contained unit of meaning,
and splitting it mid-sentence produces a chunk that is useless (or
misleading) on its own.

We implement:
  1. `recursive_chunk()` — a general-purpose baseline (character/paragraph
     aware, with overlap). This is the industry-standard default and a
     fair comparison point.
  2. `clause_aware_chunk()` — detects numbered clause headers via regex and
     keeps each clause as a single chunk wherever possible, only falling
     back to recursive splitting when an individual clause is unusually
     long (so a single giant chunk doesn't blow past a reasonable token
     budget for the LLM context window).

Being able to explain WHY the clause-aware approach exists — and show it
side by side with the naive baseline — is a strong, concrete talking point
for interviews: it shows domain-aware engineering, not just calling a
library function.
"""

import re
from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import PageContent
from src import config


@dataclass
class Chunk:
    """A single retrievable unit of text, with metadata inherited from its source page."""

    text: str
    source: str
    page_number: int
    chunk_id: str
    chunk_strategy: str          # "recursive" or "clause_aware"
    metadata: dict = field(default_factory=dict)


# Matches clause headers like "1. PARTIES.", "12. GOVERNING LAW.", "3.2 Interest Rate"
# at the start of a line. Deliberately conservative (requires a number + period/space)
# to avoid false-positives on things like "Section 3 discusses..." mid-sentence.
CLAUSE_HEADER_PATTERN = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})?)\.\s+([A-Z][A-Z\s&/'-]{2,60})\.?",
    re.MULTILINE,
)


def recursive_chunk(
    pages: List[PageContent],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Baseline chunking: LangChain's RecursiveCharacterTextSplitter.

    This splitter tries to break on paragraph boundaries first, then
    sentences, then words — falling back to a hard character cut only as a
    last resort. It's a strong general-purpose default and what most teams
    reach for first, which makes it a fair comparison baseline against the
    clause-aware strategy below.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page in pages:
        splits = splitter.split_text(page.text)
        for i, split_text in enumerate(splits):
            chunks.append(
                Chunk(
                    text=split_text,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_id=f"{page.source}_p{page.page_number}_r{i}",
                    chunk_strategy="recursive",
                    metadata={**page.metadata},
                )
            )
    return chunks


def clause_aware_chunk(
    pages: List[PageContent],
    max_chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Clause-aware chunking for legal/financial documents.

    Strategy:
      1. Scan the page text for numbered clause headers (e.g. "4. RENT.").
      2. Treat the text between one header and the next as a single
         candidate chunk — this keeps a clause's full meaning intact
         (e.g. "Tenant shall pay $2,400/month, due on the 1st..." stays
         together rather than being split across two chunks).
      3. If a candidate clause chunk exceeds `max_chunk_size`, fall back to
         recursive splitting FOR THAT CLAUSE ONLY, so we don't feed an
         oversized chunk into the embedding model or blow the LLM's
         context budget.
      4. If NO clause headers are found on a page (e.g. a financial
         narrative page like an MD&A section with prose, not numbered
         clauses), fall back to recursive chunking for the whole page.

    This graceful fallback is important: it means clause_aware_chunk()
    never produces worse results than the baseline — it only improves on
    documents that actually have the numbered structure it's designed for.
    """
    max_chunk_size = max_chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []

    for page in pages:
        matches = list(CLAUSE_HEADER_PATTERN.finditer(page.text))

        if not matches:
            # No numbered clause structure detected — fall back to recursive
            # splitting so this page still gets sensibly chunked.
            splits = fallback_splitter.split_text(page.text)
            for i, split_text in enumerate(splits):
                chunks.append(
                    Chunk(
                        text=split_text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_id=f"{page.source}_p{page.page_number}_ca_fb{i}",
                        chunk_strategy="clause_aware_fallback",
                        metadata={**page.metadata},
                    )
                )
            continue

        # Walk consecutive header matches, slicing the text between them.
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(page.text)
            clause_text = page.text[start:end].strip()
            clause_number = match.group(1)
            clause_title = match.group(2).strip()

            if len(clause_text) <= max_chunk_size:
                # Clause fits comfortably in one chunk — keep it intact.
                chunks.append(
                    Chunk(
                        text=clause_text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_id=f"{page.source}_p{page.page_number}_clause{clause_number}",
                        chunk_strategy="clause_aware",
                        metadata={
                            **page.metadata,
                            "clause_number": clause_number,
                            "clause_title": clause_title,
                        },
                    )
                )
            else:
                # Clause is unusually long — fall back to recursive
                # splitting within this clause only, but preserve the
                # clause metadata on every sub-chunk so citations still
                # say "Clause 7" even when split into parts.
                sub_splits = fallback_splitter.split_text(clause_text)
                for i, sub_text in enumerate(sub_splits):
                    chunks.append(
                        Chunk(
                            text=sub_text,
                            source=page.source,
                            page_number=page.page_number,
                            chunk_id=f"{page.source}_p{page.page_number}_clause{clause_number}_part{i}",
                            chunk_strategy="clause_aware_split",
                            metadata={
                                **page.metadata,
                                "clause_number": clause_number,
                                "clause_title": clause_title,
                            },
                        )
                    )

    return chunks
