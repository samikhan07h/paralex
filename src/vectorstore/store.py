"""
Vector store for ParaLex, backed by FAISS.

WHY FAISS (vs a hosted vector DB like Pinecone/Weaviate):
  - Free and fully local — no account, no network dependency, no usage
    limits. For a portfolio project that needs to be cloned and run by a
    stranger (or an interviewer) in five minutes, that matters a lot.
  - Fast — FAISS is a C++ library with Python bindings, built exactly for
    this job (approximate/exact nearest-neighbor search over dense vectors).
  - Persisted to disk, so the index survives between runs and doesn't need
    to be rebuilt every time the Streamlit app starts.

WHY IndexFlatIP (exact inner-product search) INSTEAD OF AN APPROXIMATE INDEX:
Our corpus size (a handful of documents, low thousands of chunks at most)
is small enough that exact search is essentially free — no accuracy/speed
tradeoff needed. Since embeddings are L2-normalized (see embedder.py),
inner product IS cosine similarity, so IndexFlatIP gives us exact cosine
similarity search with no approximation error. We call this out explicitly
because in a real interview you should be able to say "I used exact search
because the corpus was small enough that approximate search would have
been solving a problem I didn't have" — that's a much stronger answer than
copying an HNSW config because a tutorial did.

WHY WE STORE CHUNK METADATA SEPARATELY (JSON) ALONGSIDE THE FAISS INDEX:
FAISS only stores vectors and returns integer indices on search — it knows
nothing about the text, source, or clause number behind each vector. We
maintain a parallel list of Chunk metadata, indexed identically to the
FAISS index, and persist it alongside the .index file so a search result
(an integer + a score) can be resolved back into "Clause 7 of
sample_lease_agreement.pdf, page 1" without re-running the pipeline.
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.chunking.chunker import Chunk
from src import config


class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP index plus its associated
    chunk metadata, with save/load persistence to disk.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []  # parallel array: chunks[i] <-> index vector i

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their corresponding embeddings to the index.

        embeddings must be shape (len(chunks), embedding_dim) and should
        already be L2-normalized (Embedder does this by default) so that
        inner product search below is equivalent to cosine similarity.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: index expects {self.embedding_dim}, "
                f"got {embeddings.shape[1]}."
            )

        # FAISS requires float32 contiguous arrays.
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[Chunk, float]]:
        """
        Search for the top_k most similar chunks to a query embedding.

        Returns a list of (Chunk, similarity_score) tuples, ordered by
        descending similarity. Since vectors are normalized, scores are
        cosine similarities in [-1, 1] (in practice, close to [0, 1] for
        semantically related text).
        """
        top_k = top_k or config.TOP_K

        if self.index.ntotal == 0:
            return []

        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype("float32")
        )
        # Don't request more neighbors than exist in the index.
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 if fewer than k results exist
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str | Path = None) -> None:
        """
        Persist the FAISS index and chunk metadata to disk as two files:
          - index.faiss   (the FAISS binary index)
          - chunks.pkl    (pickled list of Chunk objects, in index order)

        Pickle is used for chunks (rather than JSON) because Chunk is a
        dataclass with nested dict metadata — pickling avoids writing a
        custom (de)serializer for something that never leaves this codebase.
        """
        directory = Path(directory or config.VECTORSTORE_DIR)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        # A small human-readable manifest — useful for debugging and for
        # anyone (including future-you) inspecting the saved index without
        # loading it into Python.
        manifest = {
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self.chunks),
            "sources": sorted(set(c.source for c in self.chunks)),
        }
        with open(directory / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path = None) -> "VectorStore":
        """Load a previously saved VectorStore from disk."""
        directory = Path(directory or config.VECTORSTORE_DIR)

        index_path = directory / "index.faiss"
        chunks_path = directory / "chunks.pkl"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No saved vector store found at '{directory}'. "
                f"Run the ingestion pipeline first to build one."
            )

        index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        store = cls(embedding_dim=index.d)
        store.index = index
        store.chunks = chunks
        return store
