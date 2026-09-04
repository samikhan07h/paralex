"""
Business logic for the ParaLex Streamlit app, kept separate from
app/streamlit_app.py's UI rendering code.

WHY THIS SEPARATION EXISTS:
streamlit_app.py executes page-level Streamlit calls (st.set_page_config,
sidebar widgets, chat rendering) at import time — code that only runs
correctly inside a live `streamlit run` script context. Importing that
file directly in a test would trigger all of that UI machinery outside
its intended environment. By keeping the actual LOGIC (uploaded-file
processing, cache-key computation, secrets bridging) in this separate,
plain-Python module, it can be imported and unit-tested normally, the
same way pipeline.py's logic is tested independently of run_pipeline.py's
CLI wrapper.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from src.pipeline import load_and_chunk_documents
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever


def secrets_file_exists() -> bool:
    """
    Check whether a Streamlit secrets.toml file actually exists, WITHOUT
    touching st.secrets itself.

    WHY THIS MATTERS: accessing st.secrets when no secrets.toml is present
    causes Streamlit to render a visible "No secrets found" warning box in
    the app UI as a side effect of the property access itself — not just
    a Python exception we could catch. For local development (where
    config comes from a .env file, not Streamlit secrets), this is pure
    noise. Checking for the file's existence first lets us skip touching
    st.secrets entirely in that case, while still working correctly once
    deployed (Streamlit Community Cloud provisions a real secrets.toml
    behind the scenes).
    """
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",
    ]
    return any(p.exists() for p in candidates)


def bridge_secrets_to_env(secrets, keys: Tuple[str, ...] = ("GROQ_API_KEY", "GROQ_MODEL", "EMBEDDING_MODEL", "TOP_K")) -> None:
    """
    Copy relevant keys from a Streamlit secrets-like mapping into
    os.environ, so src/config.py's os.getenv() calls see them identically
    whether running locally (.env via python-dotenv) or deployed on
    Streamlit Community Cloud (secrets.toml via st.secrets).

    Accepts any mapping-like object (not just st.secrets) so this is
    testable with a plain dict.
    """
    for key in keys:
        try:
            if key in secrets:
                os.environ[key] = str(secrets[key])
        except Exception:
            pass  # some secrets objects raise if unconfigured entirely — safe to ignore


def uploaded_files_signature(uploaded_files) -> tuple:
    """
    A cheap, order-independent fingerprint of a set of uploaded files
    (by name + size), used to detect when the user has actually changed
    their upload set — so the index is only rebuilt when needed, not on
    every Streamlit rerun (which happens on almost any UI interaction).
    """
    return tuple(sorted((f.name, f.size) for f in uploaded_files))


def build_retriever_from_uploads(uploaded_files, embedder: Embedder) -> Retriever:
    """
    Build a fresh, in-memory-only retriever from user-uploaded files.

    Uploaded files are written to a temporary directory (removed
    afterward regardless of success or failure), then processed through
    the EXACT SAME load_and_chunk_documents() used for the demo corpus —
    meaning uploaded documents get clause-aware chunking and table
    extraction for free, with no separate ingestion path to maintain.

    `uploaded_files` items only need `.name` (str) and a way to get their
    bytes; in production these are Streamlit's UploadedFile objects
    (`.getbuffer()`), but any object exposing `.name` and `.getbuffer()`
    works — which is what makes this testable with simple fakes.
    """
    tmpdir = tempfile.mkdtemp(prefix="paralex_upload_")
    try:
        for uploaded_file in uploaded_files:
            dest = Path(tmpdir) / uploaded_file.name
            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())

        chunks = load_and_chunk_documents(source_dir=tmpdir, verbose=False)
        if not chunks:
            raise ValueError("No text could be extracted from the uploaded file(s).")

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        store = VectorStore(embedding_dim=embedder.embedding_dim)
        store.add(chunks, embeddings)
        return Retriever(embedder=embedder, vectorstore=store)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
