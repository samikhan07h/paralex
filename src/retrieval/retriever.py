"""
Retrieval for ParaLex.

WHY A SEPARATE RETRIEVER CLASS (RATHER THAN CALLING VectorStore.search DIRECTLY):
The retriever is the layer that turns a raw user question into embedded-
query search results. Keeping it separate from VectorStore matters because:
  1. VectorStore shouldn't need to know about the Embedder — it just stores
     and searches vectors. Mixing those concerns would make VectorStore
     harder to test and reuse.
  2. This is the natural place to add retrieval-quality improvements later
     (e.g. re-ranking, hybrid keyword+vector search, metadata filtering by
     document type) without touching the storage layer or the embedding
     layer — each Phase 3 experiment becomes a change in ONE place.
  3. It gives evaluation (Phase 2) a single, stable function to call
     (`retrieve`) when measuring precision/recall against a test set,
     regardless of what's happening underneath.
"""

from dataclasses import dataclass
from typing import List

from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.chunking.chunker import Chunk
from src import config


@dataclass
class RetrievalResult:
    """One retrieved chunk plus its similarity score, ready for the generation step."""

    chunk: Chunk
    score: float


class Retriever:
    """
    Ties an Embedder and a VectorStore together behind a single
    `retrieve(query)` method — the only thing the generation step (and
    Phase 2's evaluation harness) needs to know about.
    """

    def __init__(self, embedder: Embedder, vectorstore: VectorStore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = None, min_score: float = None) -> List[RetrievalResult]:
        """
        Embed the query and return the top_k most similar chunks.

        top_k defaults to config.TOP_K if not specified, keeping this
        consistent with the rest of the pipeline's configuration and easy
        to override for Phase 2 experiments (e.g. testing top_k=2 vs
        top_k=8 for precision/recall tradeoffs).

        min_score, if given, drops any result whose similarity score falls
        below the threshold — even if top_k would otherwise allow more
        slots. WHY THIS EXISTS: top_k alone guarantees a FIXED NUMBER of
        results, not that all of them are actually relevant. For a small,
        curated corpus (our Phase 2 demo documents), this doesn't matter —
        top_k=2 was empirically measured to achieve perfect recall there.
        But for an arbitrary user-uploaded document (Phase 4's upload
        mode), which can be far larger and topically diverse, blindly
        passing top_k chunks to the LLM risks padding the context with
        genuinely irrelevant material once a query's true matches are
        exhausted. min_score is an opt-in safety net for exactly that
        case — left as None (no filtering) by default so it never changes
        the already-measured demo-mode behavior from Phase 2.
        """
        top_k = top_k or config.TOP_K
        query_embedding = self.embedder.embed_query(query)
        raw_results = self.vectorstore.search(query_embedding, top_k=top_k)

        results = [RetrievalResult(chunk=chunk, score=score) for chunk, score in raw_results]
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

    @classmethod
    def from_saved_store(cls, embedder: Embedder = None, directory: str = None) -> "Retriever":
        """
        Convenience constructor: load a previously persisted VectorStore
        from disk and pair it with an Embedder. This is what the Streamlit
        app (Phase 4) and CLI pipeline will use at query time, since the
        index is built once (ingestion) and queried many times afterward.
        """
        embedder = embedder or Embedder()
        vectorstore = VectorStore.load(directory)
        return cls(embedder=embedder, vectorstore=vectorstore)
