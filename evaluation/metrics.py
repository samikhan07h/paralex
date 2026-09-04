"""
Retrieval quality metrics for ParaLex's evaluation layer.

WHY MRR IS THE HEADLINE METRIC HERE (NOT JUST PRECISION/RECALL):
Our hand-labeled eval set (Day 1) defines exactly ONE correct source clause
per question — this is realistic for a legal/financial RAG assistant,
where a question like "what is the monthly rent?" has one authoritative
answer location, not several. In that single-relevant-item setting:
  - recall@k degenerates into a simple hit/miss: "was the correct chunk
    anywhere in the top k?" — useful, but doesn't distinguish "found it at
    rank 1" from "found it at rank 4".
  - precision@k degenerates into (1 if hit else 0) / k — it's mostly just
    measuring k, not retrieval quality, since only one relevant item can
    ever exist among k retrieved.
  - Mean Reciprocal Rank (MRR), by contrast, directly rewards ranking the
    correct chunk HIGHER: 1/rank. A system that always finds the answer at
    rank 1 scores 1.0; one that only ever finds it at rank 4 scores 0.25.
    This is exactly the property we care about for a RAG system, since the
    generation step only sees the top-k chunks — rank matters.
We report precision@k and recall@k as well (they're standard, expected in
an ML portfolio, and worth knowing how to compute), but MRR is what we lead
with in the final report because it best reflects what "good retrieval"
actually means for this eval set's structure. Being able to explain *why*
one metric was chosen over another — rather than reporting whatever a
tutorial reports — is exactly the kind of judgment call worth surfacing in
an interview.

WHY RELEVANCE IS DETECTED DIFFERENTLY FOR CLAUSE-NUMBERED VS. FALLBACK DOCS:
For lease/loan documents, clause_aware_chunk() attaches an exact
clause_number to each chunk's metadata, so we can check relevance exactly:
same source document AND same clause number. For the 10-K (which has no
numbered clauses and falls back to recursive chunking), there's no such
exact identifier — so we treat a chunk as relevant if it's from the
correct source document AND contains at least one of the eval item's
expected keywords. This mirrors how a human would judge "is this the right
passage" when there's no clause number to point to.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.chunking.chunker import Chunk
from src.retrieval.retriever import Retriever, RetrievalResult
from evaluation.test_sets_loader import EvalItem


def is_relevant_chunk(chunk: Chunk, item: EvalItem) -> bool:
    """
    Decide whether a retrieved chunk counts as "the correct chunk" for a
    given eval item. See module docstring for why this differs by
    document type.
    """
    if chunk.source != item.expected_source_doc:
        return False

    if item.expected_clause_number is not None:
        return chunk.metadata.get("clause_number") == item.expected_clause_number

    # Fallback (no clause numbers, e.g. the 10-K): relevance is judged by
    # keyword presence within the correct document.
    return any(keyword in chunk.text for keyword in item.expected_keywords)


def find_rank_of_relevant_chunk(
    results: List[RetrievalResult], item: EvalItem
) -> Optional[int]:
    """
    Return the 1-indexed rank of the first relevant chunk in a ranked list
    of retrieval results, or None if no relevant chunk was retrieved at all.
    """
    for rank, result in enumerate(results, start=1):
        if is_relevant_chunk(result.chunk, item):
            return rank
    return None


@dataclass
class ItemRetrievalResult:
    """Per-question retrieval outcome, before aggregation."""

    item_id: str
    question: str
    rank: Optional[int]  # None if the correct chunk was never retrieved
    retrieved_sources: List[str] = field(default_factory=list)  # for debugging/reporting


@dataclass
class RetrievalMetricsReport:
    """Aggregated retrieval metrics across an entire eval set, for a given top_k."""

    top_k: int
    num_items: int
    mrr: float
    precision_at_k: float
    recall_at_k: float
    per_item_results: List[ItemRetrievalResult]

    def summary(self) -> str:
        return (
            f"Retrieval metrics @ k={self.top_k} (n={self.num_items} questions)\n"
            f"  MRR:            {self.mrr:.3f}\n"
            f"  Recall@{self.top_k}:      {self.recall_at_k:.3f}\n"
            f"  Precision@{self.top_k}:   {self.precision_at_k:.3f}"
        )


def evaluate_retrieval(
    retriever: Retriever, items: List[EvalItem], top_k: int = 4
) -> RetrievalMetricsReport:
    """
    Run every eval item's question through the retriever and compute
    aggregate retrieval metrics against the labeled ground truth.
    """
    per_item_results: List[ItemRetrievalResult] = []
    reciprocal_ranks: List[float] = []
    recalls: List[int] = []
    precisions: List[float] = []

    for item in items:
        results = retriever.retrieve(item.question, top_k=top_k)
        rank = find_rank_of_relevant_chunk(results, item)

        per_item_results.append(
            ItemRetrievalResult(
                item_id=item.id,
                question=item.question,
                rank=rank,
                retrieved_sources=[
                    f"{r.chunk.source}"
                    + (f" clause {r.chunk.metadata.get('clause_number')}" if r.chunk.metadata.get("clause_number") else "")
                    for r in results
                ],
            )
        )

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        recalls.append(1 if rank is not None else 0)
        # With exactly one relevant chunk per question, precision@k is
        # (1 if found else 0) / k — see module docstring for why this
        # metric is less informative than MRR in this specific setup, but
        # we compute it anyway since it's a standard, expected metric.
        precisions.append((1.0 / top_k) if rank is not None else 0.0)

    n = len(items)
    return RetrievalMetricsReport(
        top_k=top_k,
        num_items=n,
        mrr=sum(reciprocal_ranks) / n,
        precision_at_k=sum(precisions) / n,
        recall_at_k=sum(recalls) / n,
        per_item_results=per_item_results,
    )
