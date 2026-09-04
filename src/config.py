"""
Central configuration for ParaLex.

WHY THIS FILE EXISTS:
Every "tunable" decision in the pipeline (which embedding model, which LLM,
chunk size, top-k) lives here and is read from environment variables. This
means:
  1. We can swap embedding models or LLM providers for benchmarking (Phase 3)
     without touching pipeline logic — just change .env or override in code.
  2. Nothing sensitive (API keys) is hardcoded into source files.
  3. In an interview, you can point to one file and explain every design
     knob in the system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a local .env file (if present). In production/deployment
# (e.g. Streamlit Community Cloud), these are instead injected as real
# environment variables / secrets, so load_dotenv() is a safe no-op there.
load_dotenv()

# --- Project root & paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
VECTORSTORE_DIR = PROJECT_ROOT / os.getenv("VECTORSTORE_DIR", "data/processed/faiss_index")
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

# --- Retrieval ---
# TOP_K=2 was chosen empirically, not by default/convention — Phase 2's evaluation
# (evaluation/metrics.py, see tests/test_eval_integration.py) measured Recall@k and
# MRR across k=1,2,4,6 on the labeled eval set and found recall/MRR plateau at k=2
# (every correct clause is found within the top 2 results), while precision keeps
# dropping as k grows (more irrelevant chunks passed to the LLM for no retrieval
# benefit). k=2 was the smallest value achieving maximum recall/MRR.
#
# IMPORTANT: this number is only measured/justified against our small, curated
# ~24-chunk demo corpus (Phase 2's eval set). It does NOT generalize to an
# arbitrary user-uploaded document (Phase 4's upload mode), which can be far
# larger and topically diverse — retrieving only 2 chunks from a 300-page
# annual report risks missing the answer entirely even when it's genuinely
# present in the document. UPLOAD_TOP_K below is a separate, more generous
# default used specifically for that less-controlled case, rather than
# changing the one number we've actually validated with real measurements.
TOP_K = int(os.getenv("TOP_K", 2))
UPLOAD_TOP_K = int(os.getenv("UPLOAD_TOP_K", 4))

# Minimum cosine similarity a retrieved chunk must meet to be kept, used only
# for upload mode (see Retriever.retrieve()'s min_score parameter). Left
# unused (None) for the demo corpus so Phase 2's measured behavior is never
# altered. 0.15 is intentionally permissive — a stricter threshold would need
# tuning against real measured data we don't have for arbitrary uploads;
# this exists as a safety net against obviously irrelevant padding, not a
# precisely-tuned relevance filter.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", 0.15))


def ensure_dirs() -> None:
    """Create data directories if they don't exist yet. Safe to call repeatedly."""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORSTORE_DIR, SAMPLE_DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
