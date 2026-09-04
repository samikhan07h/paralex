"""
Unified evaluation runner for ParaLex.

WHY THIS FILE EXISTS SEPARATELY FROM THE TEST FILES IN tests/:
test_eval_integration.py and test_faithfulness_integration.py exist to
ASSERT quality bars as part of the test suite (pass/fail, run via pytest,
gate against regressions). This module exists to PRODUCE a report — a
JSON artifact and a human-readable summary — that becomes the actual
evidence you show in interviews and the README. It reuses the exact same
underlying logic (evaluation/metrics.py, evaluation/faithfulness.py) so
the numbers are guaranteed consistent with what the tests verify; this
file is just a different, report-oriented entry point into that logic.

WHY THE REPORT COMBINES RETRIEVAL RANK + FAITHFULNESS SCORE PER QUESTION
(RATHER THAN TWO SEPARATE REPORTS):
Seeing both together per question makes certain patterns visible that
either metric alone would hide — e.g. a question where retrieval found the
right chunk (good rank) but the LLM still produced an unfaithful answer
(low score) points to a generation/prompting problem, not a retrieval
problem, and vice versa. That distinction matters when deciding what to
improve next.
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.ingestion.loaders import load_documents_from_dir
from src.chunking.chunker import clause_aware_chunk
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from evaluation.test_sets_loader import load_all_test_sets, EvalItem
from evaluation.metrics import evaluate_retrieval, find_rank_of_relevant_chunk
from evaluation.faithfulness import FaithfulnessJudge
from evaluation.rate_limit import call_with_backoff
from src import config

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class ItemReport:
    """Combined retrieval + faithfulness outcome for a single eval question."""

    item_id: str
    question: str
    expected_answer: str
    generated_answer: str
    retrieval_rank: Optional[int]  # 1-indexed rank of the correct chunk, or None if missed
    faithfulness_score: int
    faithful: bool
    unsupported_claims: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Full evaluation report: aggregate metrics plus every per-item result."""

    timestamp: str
    top_k: int
    num_questions: int
    mrr: float
    recall_at_k: float
    precision_at_k: float
    avg_faithfulness_score: float
    faithful_rate: float
    embedding_model: str
    llm_model: str
    items: List[ItemReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self) -> None:
        print("=" * 60)
        print("ParaLex Evaluation Report")
        print("=" * 60)
        print(f"Timestamp:           {self.timestamp}")
        print(f"Questions evaluated: {self.num_questions}")
        print(f"Embedding model:     {self.embedding_model}")
        print(f"LLM model:           {self.llm_model}")
        print(f"top_k:               {self.top_k}")
        print()
        print("Retrieval quality:")
        print(f"  MRR:              {self.mrr:.3f}")
        print(f"  Recall@{self.top_k}:        {self.recall_at_k:.3f}")
        print(f"  Precision@{self.top_k}:     {self.precision_at_k:.3f}")
        print()
        print("Answer faithfulness:")
        print(f"  Average score:    {self.avg_faithfulness_score:.2f} / 5")
        print(f"  Faithful rate:    {self.faithful_rate:.1%}")
        print()

        flagged = [item for item in self.items if not item.faithful]
        if flagged:
            print(f"Flagged for review ({len(flagged)}):")
            for item in flagged:
                print(f"  [{item.item_id}] score={item.faithfulness_score}, retrieval_rank={item.retrieval_rank}")
                print(f"    Q: {item.question}")
                print(f"    A: {item.generated_answer}")
                print(f"    Unsupported claims: {item.unsupported_claims}")
        else:
            print("No answers flagged — all met the faithfulness bar.")
        print("=" * 60)


def _build_pipeline():
    pages = load_documents_from_dir(config.SAMPLE_DOCS_DIR)
    chunks = clause_aware_chunk(pages)
    embedder = Embedder()
    embeddings = embedder.embed_texts([c.text for c in chunks], show_progress=True)

    store = VectorStore(embedding_dim=embedder.embedding_dim)
    store.add(chunks, embeddings)
    retriever = Retriever(embedder=embedder, vectorstore=store)
    generator = Generator()
    judge = FaithfulnessJudge()
    return retriever, generator, judge


def run_full_evaluation(top_k: int = None) -> EvaluationReport:
    """
    Run the complete Phase 2 evaluation: retrieval metrics + generation +
    faithfulness scoring, across every question in the labeled eval set.
    Returns a single EvaluationReport combining everything.
    """
    top_k = top_k or config.TOP_K
    retriever, generator, judge = _build_pipeline()
    items: List[EvalItem] = load_all_test_sets()

    retrieval_report = evaluate_retrieval(retriever, items, top_k=top_k)
    rank_by_item_id = {r.item_id: r.rank for r in retrieval_report.per_item_results}

    # NOTE: retriever.retrieve() runs twice per question — once inside
    # evaluate_retrieval() above, once again here for generation. This is
    # a deliberate simplicity tradeoff: retrieval is local (embedding +
    # FAISS search, no API cost), so the duplicate work is cheap, and
    # keeping evaluate_retrieval() self-contained (it doesn't need to
    # return raw RetrievalResults, just rank) keeps that function reusable
    # on its own, e.g. from tests/test_eval_integration.py. If retrieval
    # ever becomes expensive (e.g. a hosted vector DB with network calls),
    # this would be worth refactoring to retrieve once and reuse.
    item_reports: List[ItemReport] = []
    faithfulness_scores: List[int] = []
    faithful_flags: List[bool] = []

    for item in items:
        retrieved = retriever.retrieve(item.question, top_k=top_k)
        generated = call_with_backoff(lambda: generator.generate(item.question, retrieved))
        context_text = "\n\n".join(r.chunk.text for r in retrieved)
        verdict = call_with_backoff(lambda: judge.judge(item.question, context_text, generated.answer))

        item_reports.append(ItemReport(
            item_id=item.id,
            question=item.question,
            expected_answer=item.expected_answer,
            generated_answer=generated.answer,
            retrieval_rank=rank_by_item_id.get(item.id),
            faithfulness_score=verdict.score,
            faithful=verdict.faithful,
            unsupported_claims=verdict.unsupported_claims,
        ))
        faithfulness_scores.append(verdict.score)
        faithful_flags.append(verdict.faithful)

    return EvaluationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        top_k=top_k,
        num_questions=len(items),
        mrr=retrieval_report.mrr,
        recall_at_k=retrieval_report.recall_at_k,
        precision_at_k=retrieval_report.precision_at_k,
        avg_faithfulness_score=sum(faithfulness_scores) / len(faithfulness_scores),
        faithful_rate=sum(faithful_flags) / len(faithful_flags),
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.GROQ_MODEL,
        items=item_reports,
    )


def save_report(report: EvaluationReport, path: Path = None) -> Path:
    """Save the report as JSON, returning the path it was written to."""
    path = path or (RESULTS_DIR / "latest_eval_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run ParaLex's full evaluation suite")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve per question")
    parser.add_argument("--output", type=str, default=None, help="Path to save the JSON report")
    args = parser.parse_args()

    print("Running full evaluation (retrieval + generation + faithfulness)...")
    print("This makes real Groq API calls (2 per question) and may take a few minutes.\n")

    report = run_full_evaluation(top_k=args.top_k)
    report.print_summary()

    output_path = Path(args.output) if args.output else None
    saved_path = save_report(report, output_path)
    print(f"\nFull JSON report saved to: {saved_path}")


if __name__ == "__main__":
    main()
