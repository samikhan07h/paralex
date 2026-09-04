"""
End-to-end retrieval evaluation: runs the REAL retriever (real embeddings,
real FAISS index) against the full 18-question labeled eval set, and
prints a metrics report.

This is kept separate from test_metrics.py (which tests the metric math
in isolation with mocks) for the same reason test_pipeline_integration.py
is separate from the unit tests in test_vectorstore.py / test_retrieval.py:
this one is slower, has a real dependency (the embedding model), and
exists to prove real-world retrieval QUALITY, not just correct metric
computation.

Requires the embedding model to be available locally (downloaded once in
Phase 1, Day 3 — cached afterward, no network needed on subsequent runs).
"""

from pathlib import Path

from src.ingestion.loaders import load_documents_from_dir
from src.chunking.chunker import clause_aware_chunk
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever
from evaluation.test_sets_loader import load_all_test_sets
from evaluation.metrics import evaluate_retrieval

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def _build_test_retriever() -> Retriever:
    pages = load_documents_from_dir(SAMPLE_DIR)
    chunks = clause_aware_chunk(pages)
    embedder = Embedder()
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = VectorStore(embedding_dim=embedder.embedding_dim)
    store.add(chunks, embeddings)
    return Retriever(embedder=embedder, vectorstore=store)


def test_retrieval_quality_meets_minimum_bar_on_full_eval_set():
    """
    This is a REGRESSION GUARDRAIL, not a precise quality bar: it fails
    loudly if a future change (e.g. a chunking regression, a broken
    embedding call) tanks retrieval quality, without being so strict that
    reasonable variation in embedding behavior causes false failures.
    """
    retriever = _build_test_retriever()
    items = load_all_test_sets()

    report = evaluate_retrieval(retriever, items, top_k=4)
    print("\n" + report.summary())

    # A well-functioning system on this small, clean eval set should find
    # the correct chunk within the top 4 results for the large majority of
    # questions. Anything below this on our fixed sample set indicates a
    # real regression worth investigating, not normal variance.
    assert report.recall_at_k >= 0.8, (
        f"Recall@4 dropped to {report.recall_at_k:.2f} — expected >= 0.80. "
        f"Per-item results: {report.per_item_results}"
    )
    assert report.mrr >= 0.7, f"MRR dropped to {report.mrr:.2f} — expected >= 0.70."


def test_retrieval_quality_report_at_multiple_k_values():
    """
    Not a pass/fail test — prints a comparison of MRR/recall across
    top_k in {1, 2, 4, 6} so we can see the actual tradeoff described in
    Phase 2's plan: does a smaller top_k lose meaningful recall, or was
    top_k=4 more generous than necessary? This directly answers the
    observation from Phase 1 (the pets/rent questions retrieving 4 sources
    when only 1 was actually relevant).
    """
    retriever = _build_test_retriever()
    items = load_all_test_sets()

    print()
    for k in (1, 2, 4, 6):
        report = evaluate_retrieval(retriever, items, top_k=k)
        print(report.summary())
        print()
