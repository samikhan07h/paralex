"""
Tests for evaluation/run_eval.py

Mocks retriever, generator, and judge to verify the REPORT ASSEMBLY logic
(combining retrieval rank + faithfulness score per question, computing
aggregates, JSON serialization) deterministically and without real API
calls. The real end-to-end run (via `python -m evaluation.run_eval`) is
what produces the actual numbers for the README — this test suite exists
to make sure the plumbing connecting Days 1-3 together is correct.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.run_eval import run_full_evaluation, save_report, EvaluationReport, ItemReport
from evaluation.test_sets_loader import EvalItem
from evaluation.faithfulness import FaithfulnessResult
from src.chunking.chunker import Chunk
from src.retrieval.retriever import RetrievalResult
from src.generation.generator import GeneratedAnswer


def _make_item(item_id, source, clause_number, question="a question?", expected="an answer"):
    return EvalItem(
        id=item_id, question=question, expected_answer=expected,
        expected_source_doc=source, expected_clause_number=clause_number,
        expected_keywords=["keyword"],
    )


def _make_result(source, clause_number, text="chunk text", score=0.9):
    chunk = Chunk(text=text, source=source, page_number=1, chunk_id="x",
                  chunk_strategy="test", metadata={"clause_number": clause_number})
    return RetrievalResult(chunk=chunk, score=score)


@patch("evaluation.run_eval._build_pipeline")
@patch("evaluation.run_eval.load_all_test_sets")
def test_run_full_evaluation_produces_correct_aggregate_metrics(mock_load_items, mock_build_pipeline):
    items = [_make_item("q1", "lease.pdf", "3"), _make_item("q2", "loan.pdf", "2")]
    mock_load_items.return_value = items

    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = [
        [_make_result("lease.pdf", "3")],  # q1: retrieval metrics pass (evaluate_retrieval loops all items first)
        [_make_result("loan.pdf", "2")],   # q2: retrieval metrics pass
        [_make_result("lease.pdf", "3")],  # q1: generation pass (second loop over all items)
        [_make_result("loan.pdf", "2")],   # q2: generation pass
    ]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(answer="Generated text", sources_used=["Source 1"])

    mock_judge = MagicMock()
    mock_judge.judge.return_value = FaithfulnessResult(score=5, faithful=True, unsupported_claims=[])

    mock_build_pipeline.return_value = (mock_retriever, mock_generator, mock_judge)

    report = run_full_evaluation(top_k=1)

    assert report.num_questions == 2
    assert report.mrr == 1.0  # both found at rank 1
    assert report.avg_faithfulness_score == 5.0
    assert report.faithful_rate == 1.0
    assert len(report.items) == 2


@patch("evaluation.run_eval._build_pipeline")
@patch("evaluation.run_eval.load_all_test_sets")
def test_run_full_evaluation_links_retrieval_rank_to_correct_item(mock_load_items, mock_build_pipeline):
    """
    The key correctness property of the report: each ItemReport's
    retrieval_rank must correspond to THAT item's question, not get mixed
    up with another item's rank when results are assembled.
    """
    items = [_make_item("q1", "lease.pdf", "3"), _make_item("q2", "loan.pdf", "2")]
    mock_load_items.return_value = items

    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = [
        # Retrieval metrics pass (evaluate_retrieval loops all items first)
        [_make_result("lease.pdf", "9"), _make_result("lease.pdf", "3")],  # q1's chunk at rank 2
        [_make_result("loan.pdf", "2")],                                   # q2's chunk at rank 1
        # Generation pass (second loop over all items)
        [_make_result("lease.pdf", "9"), _make_result("lease.pdf", "3")],
        [_make_result("loan.pdf", "2")],
    ]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(answer="text", sources_used=[])

    mock_judge = MagicMock()
    mock_judge.judge.return_value = FaithfulnessResult(score=4, faithful=True, unsupported_claims=[])

    mock_build_pipeline.return_value = (mock_retriever, mock_generator, mock_judge)

    report = run_full_evaluation(top_k=2)

    q1_report = next(item for item in report.items if item.item_id == "q1")
    q2_report = next(item for item in report.items if item.item_id == "q2")

    assert q1_report.retrieval_rank == 2
    assert q2_report.retrieval_rank == 1


@patch("evaluation.run_eval._build_pipeline")
@patch("evaluation.run_eval.load_all_test_sets")
def test_run_full_evaluation_flags_low_faithfulness_items(mock_load_items, mock_build_pipeline):
    items = [_make_item("q1", "lease.pdf", "3")]
    mock_load_items.return_value = items

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_make_result("lease.pdf", "3")]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(answer="A hallucinated answer.", sources_used=[])

    mock_judge = MagicMock()
    mock_judge.judge.return_value = FaithfulnessResult(
        score=2, faithful=False, unsupported_claims=["Invented a fact not in context."]
    )
    mock_build_pipeline.return_value = (mock_retriever, mock_generator, mock_judge)

    report = run_full_evaluation(top_k=1)

    assert report.faithful_rate == 0.0
    assert report.items[0].faithful is False
    assert report.items[0].unsupported_claims == ["Invented a fact not in context."]


def test_report_serializes_to_valid_json():
    report = EvaluationReport(
        timestamp="2026-01-01T00:00:00Z", top_k=2, num_questions=1,
        mrr=1.0, recall_at_k=1.0, precision_at_k=0.5,
        avg_faithfulness_score=4.5, faithful_rate=1.0,
        embedding_model="test-model", llm_model="test-llm",
        items=[ItemReport(
            item_id="q1", question="q?", expected_answer="a",
            generated_answer="a", retrieval_rank=1,
            faithfulness_score=5, faithful=True, unsupported_claims=[],
        )],
    )

    json_str = json.dumps(report.to_dict())
    parsed = json.loads(json_str)
    assert parsed["mrr"] == 1.0
    assert parsed["items"][0]["item_id"] == "q1"


def test_save_report_writes_readable_json_file():
    report = EvaluationReport(
        timestamp="2026-01-01T00:00:00Z", top_k=2, num_questions=1,
        mrr=1.0, recall_at_k=1.0, precision_at_k=0.5,
        avg_faithfulness_score=4.5, faithful_rate=1.0,
        embedding_model="test-model", llm_model="test-llm", items=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.json"
        saved_path = save_report(report, path)

        assert saved_path == path
        with open(saved_path) as f:
            loaded = json.load(f)
        assert loaded["mrr"] == 1.0


def test_print_summary_runs_without_error_and_mentions_key_metrics(capsys):
    report = EvaluationReport(
        timestamp="2026-01-01T00:00:00Z", top_k=2, num_questions=1,
        mrr=0.917, recall_at_k=1.0, precision_at_k=0.5,
        avg_faithfulness_score=4.83, faithful_rate=0.944,
        embedding_model="test-model", llm_model="test-llm",
        items=[ItemReport(
            item_id="q1", question="q?", expected_answer="a",
            generated_answer="a", retrieval_rank=1,
            faithfulness_score=5, faithful=True, unsupported_claims=[],
        )],
    )

    report.print_summary()
    captured = capsys.readouterr()
    assert "MRR" in captured.out
    assert "0.917" in captured.out
    assert "Faithful rate" in captured.out
