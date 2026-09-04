"""
Tests for evaluation/faithfulness.py

The Groq call itself is mocked throughout (except the optional live test
at the bottom) so these tests run deterministically and without spending
API quota. Most of these tests focus on _extract_json_object(), since
robustly parsing LLM output is the single most fragile part of any
LLM-as-judge implementation — a judge that "usually" returns clean JSON
but occasionally wraps it in markdown or a stray sentence will silently
break faithfulness scoring in exactly the runs where you're not watching.
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from evaluation.faithfulness import (
    FaithfulnessJudge,
    FaithfulnessResult,
    _extract_json_object,
)


# --- _extract_json_object: the fragile part ---

def test_extract_json_object_handles_clean_json():
    text = '{"score": 5, "faithful": true, "unsupported_claims": [], "reasoning": "Fully supported."}'
    result = _extract_json_object(text)
    assert result["score"] == 5


def test_extract_json_object_handles_markdown_fenced_json():
    text = '```json\n{"score": 3, "faithful": false, "unsupported_claims": ["x"], "reasoning": "y"}\n```'
    result = _extract_json_object(text)
    assert result["score"] == 3
    assert result["unsupported_claims"] == ["x"]


def test_extract_json_object_handles_leading_and_trailing_text():
    text = 'Here is my evaluation:\n{"score": 2, "faithful": false, "unsupported_claims": [], "reasoning": "z"}\nHope that helps!'
    result = _extract_json_object(text)
    assert result["score"] == 2


def test_extract_json_object_raises_on_completely_invalid_input():
    with pytest.raises(json.JSONDecodeError):
        _extract_json_object("This is not JSON at all.")


# --- FaithfulnessJudge.judge() with mocked LLM call ---

def _mock_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_judge_raises_clear_error_with_no_api_key(monkeypatch):
    monkeypatch.setattr("evaluation.faithfulness.config.GROQ_API_KEY", "")
    with pytest.raises(ValueError, match="No Groq API key found"):
        FaithfulnessJudge(api_key="")


def test_judge_returns_faithfulness_result_for_high_score():
    judge = FaithfulnessJudge(api_key="fake_key")
    mock_content = json.dumps({
        "score": 5, "faithful": True, "unsupported_claims": [],
        "reasoning": "All claims match the context exactly.",
    })

    with patch.object(judge.client.chat.completions, "create", return_value=_mock_response(mock_content)):
        result = judge.judge("What is the rent?", "Rent is $2,400/month.", "The rent is $2,400/month.")

    assert isinstance(result, FaithfulnessResult)
    assert result.score == 5
    assert result.faithful is True
    assert result.unsupported_claims == []


def test_judge_flags_unsupported_claims_for_low_score():
    judge = FaithfulnessJudge(api_key="fake_key")
    mock_content = json.dumps({
        "score": 2, "faithful": False,
        "unsupported_claims": ["The answer claims a $3,000 deposit, but context says $2,400."],
        "reasoning": "The answer states an incorrect deposit amount not found in the context.",
    })

    with patch.object(judge.client.chat.completions, "create", return_value=_mock_response(mock_content)):
        result = judge.judge(
            "What is the security deposit?",
            "The security deposit is $2,400.",
            "The security deposit is $3,000.",  # hallucinated number
        )

    assert result.score == 2
    assert result.faithful is False
    assert len(result.unsupported_claims) == 1


def test_judge_raises_clear_error_on_unparseable_response():
    judge = FaithfulnessJudge(api_key="fake_key")

    with patch.object(judge.client.chat.completions, "create", return_value=_mock_response("garbage output, not JSON")):
        with pytest.raises(ValueError, match="Could not parse judge response"):
            judge.judge("q", "context", "answer")


def test_judge_sends_low_temperature_for_deterministic_scoring():
    judge = FaithfulnessJudge(api_key="fake_key")
    mock_content = json.dumps({"score": 5, "faithful": True, "unsupported_claims": [], "reasoning": "ok"})

    with patch.object(judge.client.chat.completions, "create", return_value=_mock_response(mock_content)) as mock_create:
        judge.judge("q", "context", "answer")
        assert mock_create.call_args.kwargs["temperature"] == 0


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Requires a real GROQ_API_KEY to hit the live API")
def test_live_judge_correctly_flags_a_hallucinated_number():
    """
    Optional live test: feeds the judge a deliberately hallucinated answer
    (wrong dollar figure) against real context, and confirms it's flagged
    as unfaithful. This is the most important live test in Phase 2 — it
    proves the judge actually catches the exact failure mode it exists to
    catch, not just that the API call succeeds.
    """
    judge = FaithfulnessJudge()
    context = "Tenant shall pay Landlord monthly rent of $2,400.00, due on the 1st day of each month."
    hallucinated_answer = "The monthly rent is $3,500.00, due on the 15th of each month."

    result = judge.judge("What is the monthly rent?", context, hallucinated_answer)

    assert result.score <= 2
    assert result.faithful is False
