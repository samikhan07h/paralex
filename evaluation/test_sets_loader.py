"""
Test set loading for ParaLex's evaluation layer.

WHY A DATACLASS + LOADER INSTEAD OF JUST READING JSON INLINE WHEREVER NEEDED:
Days 2-4 of Phase 2 (retrieval metrics, faithfulness scoring, the eval
runner) all need the same test set data in the same shape. Centralizing
loading here means:
  1. One place to change if the schema evolves (e.g. adding a difficulty
     tag or a document-type filter later).
  2. Type-safe access (`item.expected_clause_number` instead of
     `item["expected_clause_number"]`) catches typos at development time.
  3. Validation happens once, at load time, rather than being silently
     assumed correct throughout the pipeline.

WHY SEPARATE JSON FILES PER DOCUMENT (RATHER THAN ONE BIG FILE):
Keeps each test set easy to review/extend independently — e.g. adding more
lease questions doesn't require touching loan or 10-K questions — and
mirrors how a real team would organize eval data as more document types
are added over time.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

TEST_SETS_DIR = Path(__file__).resolve().parent / "test_sets"


@dataclass
class EvalItem:
    """One labeled test case: a question, its ground-truth answer, and where that answer should come from."""

    id: str
    question: str
    expected_answer: str
    expected_source_doc: str
    expected_clause_number: Optional[str]  # None for documents without numbered clauses (e.g. 10-K)
    expected_keywords: List[str]


def _load_and_validate(file_path: Path) -> List[EvalItem]:
    with open(file_path, "r") as f:
        raw_items = json.load(f)

    items = []
    for raw in raw_items:
        required_fields = {
            "id", "question", "expected_answer",
            "expected_source_doc", "expected_clause_number", "expected_keywords",
        }
        missing = required_fields - raw.keys()
        if missing:
            raise ValueError(
                f"Test item in {file_path.name} is missing required field(s): {missing}. "
                f"Item: {raw.get('id', '<no id>')}"
            )
        items.append(EvalItem(**raw))
    return items


def load_all_test_sets(test_sets_dir: Optional[Path] = None) -> List[EvalItem]:
    """
    Load and combine every *.json test set file in the test_sets directory
    into a single flat list of EvalItem, ready for the evaluation runner.
    """
    test_sets_dir = test_sets_dir or TEST_SETS_DIR
    all_items: List[EvalItem] = []

    for file_path in sorted(test_sets_dir.glob("*.json")):
        all_items.extend(_load_and_validate(file_path))

    if not all_items:
        raise ValueError(f"No test set JSON files found in {test_sets_dir}")

    # IDs should be unique across all test sets combined, since Day 4's
    # report will reference results by ID.
    ids = [item.id for item in all_items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate eval item IDs found across test sets: {duplicates}")

    return all_items
