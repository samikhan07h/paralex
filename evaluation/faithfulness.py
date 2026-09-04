"""
Answer faithfulness scoring for ParaLex's evaluation layer, using an
LLM-as-judge approach.

WHY FAITHFULNESS MATTERS MORE THAN "DOES THE ANSWER SOUND RIGHT":
Retrieval metrics (Day 2) tell us whether the RIGHT CONTEXT was found.
Faithfulness tells us something different and equally important: given
that context, did the LLM actually stick to it, or did it add a plausible-
sounding detail that isn't actually in the source? For legal/financial
documents, an unsupported claim (a wrong dollar figure, an invented
condition) is far more dangerous than an admitted "I don't know" — so this
is the metric that most directly measures the core risk of the product.

WHY LLM-AS-JUDGE (RATHER THAN, SAY, SIMPLE KEYWORD OVERLAP):
Faithfulness is fundamentally a semantic judgment — "is this claim implied
by this context" can't be reduced to string matching (an answer can
correctly paraphrase without repeating exact wording, or can subtly
misstate a number that keyword overlap wouldn't catch). Using an LLM to
make this judgment is standard practice (it's what RAGAS and similar
evaluation frameworks do internally) and is free to run at this scale via
Groq. The known weakness — the judge might share blind spots with the
model being judged, especially if both are similar architectures — is
worth being explicit about; a more rigorous setup would use a DIFFERENT,
stronger model as judge, which we call out as a Future Improvement.

WHY THE JUDGE PROMPT ASKS FOR STRUCTURED JSON, NOT FREE TEXT:
We need to aggregate scores across the eval set and flag specific
unsupported claims for manual review (Day 4's report). Free-text judge
output would require additional parsing/guessing; asking directly for a
JSON object with a numeric score and an explicit list of unsupported
claims makes the output immediately usable and testable.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List

from groq import Groq

from src import config


JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge evaluating whether an AI-generated \
answer is fully supported by the context it was given. You are NOT evaluating whether the answer \
is well-written or helpful — only whether every factual claim in it is actually stated in or \
directly implied by the context.

Score the answer's faithfulness on a scale of 1-5:
5 = Every claim is fully and directly supported by the context.
4 = Claims are supported, with only very minor imprecision (e.g. slight rewording that doesn't change meaning).
3 = Mostly supported, but contains at least one claim that is a reasonable inference rather than \
directly stated.
2 = Contains at least one claim not supported by the context, or a meaningful factual error \
(wrong number, wrong date, wrong party).
1 = The answer is largely unsupported by the context or contradicts it.

Respond with ONLY a JSON object in this exact format, with no other text before or after it:
{
  "score": <integer 1-5>,
  "faithful": <true if score >= 4, false otherwise>,
  "unsupported_claims": [<list of specific claims from the answer that are NOT supported by the \
context, as short strings; empty list if none>],
  "reasoning": "<one or two sentence explanation of the score>"
}"""


@dataclass
class FaithfulnessResult:
    """The judge's verdict on one generated answer."""

    score: int
    faithful: bool
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""


def _extract_json_object(text: str) -> dict:
    """
    Extract a JSON object from the judge's response text.

    Even with an explicit "JSON only" instruction, LLMs occasionally wrap
    output in markdown code fences or add a stray sentence. This function
    is defensive against that rather than assuming perfectly clean output,
    since a judge-parsing failure would silently corrupt every downstream
    faithfulness metric.
    """
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Fall back to grabbing the first {...} block in the text.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)

    return json.loads(text)


class FaithfulnessJudge:
    """
    Wraps a Groq LLM call behind a single judge() method that scores an
    answer's faithfulness to its retrieved context.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL

        if not self.api_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in your .env file "
                "(get a free key at https://console.groq.com)."
            )

        self.client = Groq(api_key=self.api_key)

    def judge(self, question: str, context: str, answer: str) -> FaithfulnessResult:
        """
        Score how faithful `answer` is to `context`, given the original
        `question` for reference (helps the judge understand what the
        answer was trying to address).
        """
        user_prompt = f"""Question: {question}

Context:
{context}

Answer to evaluate:
{answer}

Evaluate the faithfulness of the answer to the context, following the scoring rubric exactly."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # deterministic scoring, not creative judgment
        )

        raw_text = response.choices[0].message.content

        try:
            parsed = _extract_json_object(raw_text)
        except (json.JSONDecodeError, AttributeError) as e:
            raise ValueError(
                f"Could not parse judge response as JSON. Raw response:\n{raw_text}"
            ) from e

        return FaithfulnessResult(
            score=int(parsed["score"]),
            faithful=bool(parsed["faithful"]),
            unsupported_claims=list(parsed.get("unsupported_claims", [])),
            reasoning=str(parsed.get("reasoning", "")),
        )
