"""
End-to-end integration test: ingestion -> chunking -> embedding -> vector
store -> retrieval, using the REAL embedding model (not mocked).

This test requires internet access on first run (to download the
sentence-transformers model, cached afterward). It's kept separate from
the unit tests in test_vectorstore.py / test_retrieval.py because it's
slower and has a real external dependency — those unit tests should stay
fast and network-independent, while this one proves the whole Phase 1
pipeline actually works together end-to-end, not just in isolated pieces.
"""

from pathlib import Path

from src.ingestion.loaders import load_documents_from_dir
from src.chunking.chunker import clause_aware_chunk
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_full_pipeline_returns_semantically_relevant_chunk():
    # 1. Ingest
    pages = load_documents_from_dir(SAMPLE_DIR)
    assert len(pages) > 0

    # 2. Chunk
    chunks = clause_aware_chunk(pages)
    assert len(chunks) > 0

    # 3. Embed
    embedder = Embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # 4. Store
    store = VectorStore(embedding_dim=embedder.embedding_dim)
    store.add(chunks, embeddings)

    # 5. Retrieve
    retriever = Retriever(embedder=embedder, vectorstore=store)
    results = retriever.retrieve("What is the monthly rent amount?", top_k=3)

    assert len(results) == 3
    # The top result should be the RENT clause from the lease, not an
    # unrelated clause from a different document.
    top_chunk = results[0].chunk
    assert top_chunk.source == "sample_lease_agreement.pdf"
    assert top_chunk.metadata.get("clause_number") == "3"  # "3. RENT."
    assert results[0].score > results[-1].score  # descending order preserved end-to-end


def test_full_pipeline_save_and_reload_preserves_retrieval_quality():
    import tempfile
    import shutil

    pages = load_documents_from_dir(SAMPLE_DIR)
    chunks = clause_aware_chunk(pages)
    embedder = Embedder()
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = VectorStore(embedding_dim=embedder.embedding_dim)
    store.add(chunks, embeddings)

    tmpdir = tempfile.mkdtemp()
    try:
        store.save(tmpdir)
        reloaded_store = VectorStore.load(tmpdir)
        retriever = Retriever(embedder=embedder, vectorstore=reloaded_store)

        results = retriever.retrieve("Can I have a pet in my apartment?", top_k=1)
        assert results[0].chunk.metadata.get("clause_number") == "7"  # "7. PETS."
    finally:
        shutil.rmtree(tmpdir)
