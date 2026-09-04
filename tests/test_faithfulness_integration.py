"""
End-to-end faithfulness evaluation: for every question in the labeled eval
set, run the REAL pipeline (retrieve -> generate), then score the
generated answer's faithfulness to its retrieved context using the REAL
judge. This is the test that produces an actual, defensible faithfulness
number for the README's Evaluation Results section.

Requires GROQ_API_KEY (for both generation and judging) and the cached
embedding model. Skipped automatically if no API key is present, since it
makes real API calls (2 per question: one to generate, one to judge).
"""

import os
import pytest
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from src.ingestion.loaders import load_documents_from_dir
from src.chunking.chunker import clause_aware_chunk
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from evaluation.test_sets_loader import load_all_test_sets
from evaluation.faithfulness import FaithfulnessJudge
from evaluation.rate_limit import call_with_backoff

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@dataclass
class FaithfulnessRunResult:
    item_id: str
    question: str
    answer: str
    score: int
    faithful: bool
    unsupported_claims: List[str] = field(default_factory=list)


def _build_pipeline():
    pages = load_documents_from_dir(SAMPLE_DIR)
    chunks = clause_aware_chunk(pages)
    embedder = Embedder()
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = VectorStore(embedding_dim=embedder.embedding_dim)
    store.add(chunks, embeddings)
    retriever = Retriever(embedder=embedder, vectorstore=store)
    generator = Generator()
    judge = FaithfulnessJudge()
    return retriever, generator, judge


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Requires a real GROQ_API_KEY for generation + judging")
def test_faithfulness_across_full_eval_set():
    """
    Not a strict pass/fail on individual items (LLM outputs have natural
    variance) — but asserts a reasonable AGGREGATE faithfulness bar across
    the whole eval set, and prints per-item results so any low-scoring
    answers are visible for manual review rather than silently averaged away.
    """
    retriever, generator, judge = _build_pipeline()
    items = load_all_test_sets()

    results: List[FaithfulnessRunResult] = []

    for item in items:
        retrieved = retriever.retrieve(item.question)
        # Both the generate and judge calls hit Groq's rate-limited API —
        # wrap each in backoff retry so a transient TPM limit (easy to hit
        # when running 18 questions back-to-back on the free tier) doesn't
        # fail the whole evaluation run partway through.
        generated = call_with_backoff(lambda: generator.generate(item.question, retrieved))
        context_text = "\n\n".join(r.chunk.text for r in retrieved)

        verdict = call_with_backoff(lambda: judge.judge(item.question, context_text, generated.answer))
        results.append(FaithfulnessRunResult(
            item_id=item.id,
            question=item.question,
            answer=generated.answer,
            score=verdict.score,
            faithful=verdict.faithful,
            unsupported_claims=verdict.unsupported_claims,
        ))

    avg_score = sum(r.score for r in results) / len(results)
    faithful_rate = sum(1 for r in results if r.faithful) / len(results)

    print(f"\n\nFaithfulness evaluation across {len(results)} questions:")
    print(f"  Average score:  {avg_score:.2f} / 5")
    print(f"  Faithful rate:  {faithful_rate:.1%}")
    print()

    flagged = [r for r in results if not r.faithful]
    if flagged:
        print(f"Flagged for review ({len(flagged)}):")
        for r in flagged:
            print(f"  [{r.item_id}] score={r.score} \"{r.question}\"")
            print(f"    Answer: {r.answer}")
            print(f"    Unsupported claims: {r.unsupported_claims}")
            print()
    else:
        print("No answers flagged — all met the faithfulness bar.")

    assert avg_score >= 3.5, f"Average faithfulness score {avg_score:.2f} fell below the 3.5 bar."
    assert faithful_rate >= 0.7, f"Faithful rate {faithful_rate:.1%} fell below the 70% bar."
