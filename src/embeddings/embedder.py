"""
Embedding generation for ParaLex.

WHY A WRAPPER CLASS INSTEAD OF CALLING sentence-transformers DIRECTLY:
The rest of the pipeline (vectorstore, retrieval) should depend on a stable
interface — "give me a list of texts, get back vectors" — not on the
specific embedding library or model in use. This is what lets us swap in
OpenAI embeddings, or a different local model, for Phase 3's benchmarking
without touching any other file. It also gives us one place to handle
batching and normalization consistently.

WHY sentence-transformers/all-MiniLM-L6-v2 AS THE DEFAULT:
  - Free and fully local — no API key, no per-token cost, no network
    dependency once the model is downloaded (~80MB, cached after first run).
  - Fast on CPU — this matters for a portfolio project people will actually
    run/demo, not just a benchmark environment with a GPU.
  - 384-dimensional output keeps the FAISS index small and fast to search.
  - Widely used as a strong baseline in retrieval literature, so results
    are easy to sanity-check and explain.
The tradeoff (discussed further in Phase 3) is that larger/proprietary
models like OpenAI's text-embedding-3-large generally score higher on
retrieval benchmarks — we'll quantify that gap directly with real metrics
rather than asserting it.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


class Embedder:
    """
    Wraps a sentence-transformers model behind a stable embed_texts() interface.

    Loading the model happens once, in __init__ — NOT per call — since model
    loading is the expensive part (~1-2 seconds). Reusing one Embedder
    instance across many embed_texts() calls (e.g. once per pipeline run)
    is the intended usage pattern.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = SentenceTransformer(self.model_name)
        # get_embedding_dimension() is the current method name (sentence-transformers
        # >= ~3.x renamed it from get_sentence_embedding_dimension()); fall back for
        # older installed versions so this works across environments.
        if hasattr(self._model, "get_embedding_dimension"):
            self.embedding_dim = self._model.get_embedding_dimension()
        else:
            self.embedding_dim = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of texts into a (n_texts, embedding_dim) numpy array.

        Batching (rather than one-at-a-time calls) is what makes this fast
        for a full document's worth of chunks — sentence-transformers
        parallelizes within a batch on CPU.

        normalize_embeddings=True is important: it L2-normalizes each
        vector so that cosine similarity search (used later in FAISS)
        reduces to a simple dot product, which is both faster and what
        FAISS's IndexFlatIP expects.
        """
        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string. Kept as a separate method (rather than
        making callers wrap a single string in a list) because query
        embedding is a distinct, frequent operation in the retrieval path
        and deserves a clear, purpose-named entry point.
        """
        return self.embed_texts([query])[0]
