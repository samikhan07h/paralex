"""
Pipeline orchestration for ParaLex.

WHY THIS FILE EXISTS SEPARATELY FROM run_pipeline.py:
This module contains the actual pipeline LOGIC (build an index, answer a
question) as plain functions with no CLI or UI concerns. run_pipeline.py
(a thin CLI) and the future Streamlit app (Phase 4) both import and call
these same functions — neither duplicates pipeline logic. This is the
single most important structural decision for avoiding "I have to fix
this bug in three places" later.
"""

from pathlib import Path
from typing import List, Optional

from src.ingestion.loaders import load_documents_from_dir
from src.ingestion.table_extractor import extract_tables_from_dir
from src.chunking.chunker import clause_aware_chunk, Chunk
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator, GeneratedAnswer
from src import config


def load_and_chunk_documents(source_dir: Optional[str] = None, verbose: bool = False) -> List[Chunk]:
    """
    Load every document in source_dir, extract tables, and produce the
    final combined chunk list (prose + table) ready for embedding.

    WHY THIS IS ITS OWN FUNCTION (extracted from build_index in Phase 3):
    Both build_index() (below) and evaluation/embedding_comparison.py
    (Phase 3, Day 3) need the EXACT same chunking result — the comparison
    script must embed identical chunks under different models to make the
    comparison fair, and build_index() needs it to actually build the
    index. Keeping this logic in one place means a future change to
    chunking or table-page exclusion only needs to happen once, and the
    two callers can never silently drift out of sync with each other.

    See build_index()'s docstring for why table-containing pages are
    excluded from prose chunking.
    """
    source_dir = source_dir or config.SAMPLE_DOCS_DIR

    if verbose:
        print(f"[1/3] Loading documents from {source_dir} ...")
    pages = load_documents_from_dir(source_dir)
    if verbose:
        print(f"      Loaded {len(pages)} page(s)/block(s).")

    if verbose:
        print("[2/3] Extracting tables ...")
    table_chunks = extract_tables_from_dir(source_dir)
    if verbose:
        print(f"      Found {len(table_chunks)} table(s).")

    table_pages = {(c.source, c.page_number) for c in table_chunks}
    prose_pages = [p for p in pages if (p.source, p.page_number) not in table_pages]
    skipped = len(pages) - len(prose_pages)
    if verbose and skipped:
        print(f"      Excluding {skipped} page(s) with tables from prose chunking (using clean table extraction instead).")

    if verbose:
        print("[3/3] Chunking prose (clause-aware) ...")
    prose_chunks = clause_aware_chunk(prose_pages)
    if verbose:
        print(f"      Produced {len(prose_chunks)} prose chunk(s).")

    all_chunks = prose_chunks + table_chunks
    if verbose:
        print(f"      Total chunks (prose + table): {len(all_chunks)}")

    return all_chunks


def build_index(
    source_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    embedder: Optional[Embedder] = None,
) -> VectorStore:
    """
    Run the full ingestion side of the pipeline: load documents, chunk them
    clause-aware, extract tables, embed everything, and build + persist a
    FAISS index.

    This is meant to be run once whenever the document set changes (not on
    every query) — the resulting index is then loaded quickly at query time
    via Retriever.from_saved_store().

    WHY TABLE-CONTAINING PAGES ARE EXCLUDED FROM PROSE CHUNKING (Phase 3):
    pypdf's flat text extraction on a page containing a table produces a
    jumbled, cell-by-cell dump with no row/column structure (see
    src/ingestion/table_extractor.py's docstring for why). If we chunked
    that raw page text AND indexed the clean, structurally-correct table
    extraction for the SAME page, we'd end up with two versions of the same
    financial data in the index — one clean, one garbled — competing for
    retrieval. The garbled version could get retrieved instead of the
    clean one, directly hurting both retrieval precision and answer
    faithfulness. So: for any page where extract_tables_from_pdf() found
    at least one real table, we skip that page in the prose chunking pass
    entirely and rely solely on the clean table chunk. Pages with no
    tables (e.g. the 10-K's pure-prose MD&A) are unaffected.
    """
    save_dir = save_dir or config.VECTORSTORE_DIR
    embedder = embedder or Embedder()

    all_chunks = load_and_chunk_documents(source_dir, verbose=True)

    print(f"[4/5] Embedding chunks with '{embedder.model_name}' ...")
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_texts(texts, show_progress=True)

    print(f"[5/5] Building and saving FAISS index to {save_dir} ...")
    store = VectorStore(embedding_dim=embedder.embedding_dim)
    store.add(all_chunks, embeddings)
    store.save(save_dir)

    print("Done. Index is ready for querying.")
    return store


def answer_question(
    question: str,
    retriever: Optional[Retriever] = None,
    generator: Optional[Generator] = None,
    top_k: Optional[int] = None,
) -> GeneratedAnswer:
    """
    Run the full query-time pipeline: retrieve relevant chunks for
    `question`, then generate a grounded, cited answer from them.

    Accepting pre-built retriever/generator instances (rather than always
    constructing new ones) matters for the Streamlit app and for tests:
    both the embedding model and the Groq client should be created ONCE
    and reused across many questions, not reloaded per call.
    """
    retriever = retriever or Retriever.from_saved_store()
    generator = generator or Generator()

    results = retriever.retrieve(question, top_k=top_k)
    return generator.generate(question, results)
