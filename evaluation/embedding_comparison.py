"""
Embedding model comparison for ParaLex.

WHY THIS EXISTS:
Phase 1 chose sentence-transformers/all-MiniLM-L6-v2 as the default
embedding model based on well-known, general tradeoffs (fast, free,
local, small). This module tests that choice EMPIRICALLY against a larger
alternative on the exact same labeled eval set used throughout Phase 2 —
turning "MiniLM is a reasonable default" into "MiniLM achieved equivalent
retrieval quality to a 3x larger model while embedding N times faster on
this document set," which is a genuinely defensible, evidence-backed
engineering decision rather than an assumption.

WHY all-mpnet-base-v2 AS THE COMPARISON POINT:
It's the standard "step up" from MiniLM in the sentence-transformers
family — same license, same free/local deployment story (no API key, no
cost, keeping the comparison fair), but roughly 3x the parameters and a
768-dimensional output (vs. MiniLM's 384), and it's frequently cited as
scoring higher on general retrieval benchmarks (MTEB). Comparing against
a model with the same deployment constraints isolates the actual
tradeoff we care about — quality vs. speed/size — rather than conflating
it with a cost or infrastructure difference (as comparing against a paid
API-based embedding model would).

WHY CHUNKING HAPPENS ONCE, OUTSIDE THE PER-MODEL LOOP:
Chunking is entirely model-independent — the same 24 chunks should be fed
to every model being compared. Re-chunking per model would waste time and,
more importantly, risks making the "embedding time" comparison unfair if
chunking time got bundled into it inconsistently.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from src.pipeline import load_and_chunk_documents
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever
from evaluation.test_sets_loader import load_all_test_sets
from evaluation.metrics import evaluate_retrieval
from src import config

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


@dataclass
class ModelComparisonResult:
    """Quality + speed metrics for one embedding model on the eval set."""

    model_name: str
    embedding_dim: int
    num_chunks: int
    embed_time_seconds: float
    chunks_per_second: float
    mrr: float
    recall_at_k: float
    precision_at_k: float
    top_k: int


def compare_embedding_models(
    model_names: Optional[List[str]] = None, top_k: Optional[int] = None
) -> List[ModelComparisonResult]:
    """
    Embed the same chunk set under each candidate model, build a FAISS
    index per model, and evaluate retrieval quality against the full
    labeled eval set — producing a direct, apples-to-apples comparison.
    """
    model_names = model_names or CANDIDATE_MODELS
    top_k = top_k or config.TOP_K

    chunks = load_and_chunk_documents(verbose=False)
    texts = [c.text for c in chunks]
    items = load_all_test_sets()

    results: List[ModelComparisonResult] = []

    for model_name in model_names:
        print(f"\nEvaluating '{model_name}' ...")
        embedder = Embedder(model_name=model_name)

        start = time.time()
        embeddings = embedder.embed_texts(texts, show_progress=True)
        elapsed = time.time() - start

        store = VectorStore(embedding_dim=embedder.embedding_dim)
        store.add(chunks, embeddings)
        retriever = Retriever(embedder=embedder, vectorstore=store)

        retrieval_report = evaluate_retrieval(retriever, items, top_k=top_k)

        results.append(ModelComparisonResult(
            model_name=model_name,
            embedding_dim=embedder.embedding_dim,
            num_chunks=len(chunks),
            embed_time_seconds=elapsed,
            chunks_per_second=(len(chunks) / elapsed) if elapsed > 0 else float("inf"),
            mrr=retrieval_report.mrr,
            recall_at_k=retrieval_report.recall_at_k,
            precision_at_k=retrieval_report.precision_at_k,
            top_k=top_k,
        ))

    return results


def print_comparison(results: List[ModelComparisonResult]) -> None:
    print("\n" + "=" * 88)
    print("Embedding Model Comparison")
    print("=" * 88)
    print(f"{'Model':<38} {'Dim':>5} {'MRR':>7} {'Recall@k':>9} {'Precision@k':>12} {'Embed time':>11}")
    print("-" * 88)
    for r in results:
        short_name = r.model_name.split("/")[-1]
        print(
            f"{short_name:<38} {r.embedding_dim:>5} {r.mrr:>7.3f} {r.recall_at_k:>9.3f} "
            f"{r.precision_at_k:>12.3f} {r.embed_time_seconds:>10.2f}s"
        )
    print("=" * 88)

    if len(results) >= 2:
        baseline, comparison = results[0], results[1]
        mrr_diff = comparison.mrr - baseline.mrr
        speed_ratio = comparison.embed_time_seconds / baseline.embed_time_seconds if baseline.embed_time_seconds > 0 else float("inf")
        print(
            f"\n{comparison.model_name.split('/')[-1]} vs {baseline.model_name.split('/')[-1]}: "
            f"MRR {'+' if mrr_diff >= 0 else ''}{mrr_diff:.3f}, "
            f"{speed_ratio:.1f}x embedding time."
        )


def save_comparison(results: List[ModelComparisonResult], path: Optional[Path] = None) -> Path:
    path = path or (RESULTS_DIR / "embedding_comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return path


def main():
    print("Comparing embedding models on the full labeled eval set...")
    print("This downloads all-mpnet-base-v2 on first run (~420MB) if not already cached.\n")

    results = compare_embedding_models()
    print_comparison(results)
    saved_path = save_comparison(results)
    print(f"\nComparison saved to: {saved_path}")


if __name__ == "__main__":
    main()
