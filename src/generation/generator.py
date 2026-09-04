"""
Answer generation for ParaLex, using Groq's hosted LLM inference.

WHY GROQ:
Groq's free tier offers fast inference (LPU hardware, not GPU) on strong
open models (Llama 3.x) at no cost for portfolio-scale usage. This keeps
the project deployable and demo-able without anyone (including you) paying
for API credits, while still using a genuinely capable model — not a toy.

WHY A STRICT GROUNDING PROMPT (THE MOST IMPORTANT DESIGN CHOICE IN THIS FILE):
The single biggest risk in a legal/financial RAG assistant is a confident,
plausible-sounding, WRONG answer — hallucinated numbers, invented clause
terms, or a citation to the wrong section. A generic "answer the question"
prompt lets the LLM fall back on its own training knowledge when retrieval
comes up short, which is exactly what we don't want. Our system prompt:
  1. Explicitly instructs the model to answer ONLY using the provided
     context chunks.
  2. Instructs it to say "the answer isn't in the retrieved documents" if
     the retrieved chunks don't contain the answer, rather than guessing.
  3. Instructs it to cite which chunk(s) it used, by the labels we assign.
This turns "trust me" into "here's exactly where this came from" — which
is the actual product value proposition of a legal/financial RAG tool.

WHY CITATION LABELS ARE BUILT INTO THE CONTEXT STRING (NOT ADDED AFTER):
By prefixing each chunk with a label like "[Source 1: sample_lease_agreement.pdf,
Clause 3 (RENT), page 1]" before it reaches the LLM, the model can reference
that exact label in its answer ("According to Source 1..."). This is far
more reliable than trying to reverse-engineer which retrieved chunk an
answer came from after the fact.
"""

from dataclasses import dataclass
from typing import List

from groq import Groq

from src.retrieval.retriever import RetrievalResult
from src import config


SYSTEM_PROMPT = """You are ParaLex, an AI assistant that answers questions about legal and \
financial documents (contracts, leases, loan agreements, financial statements) using ONLY the \
context provided to you.

Rules you must follow:
1. Answer using ONLY the information in the provided context. Do not use outside knowledge, \
even if you know the general subject matter.
2. If the context does not contain enough information to answer the question, say so clearly: \
"The retrieved documents don't contain enough information to answer this question." Do not guess \
or fill in gaps with assumptions.
3. When you use information from a source, cite it by its label, e.g. "(Source 1)" or \
"(Source 2)". Cite every specific fact, number, or clause you reference.
4. Be precise with numbers, dates, and defined terms — do not round, approximate, or paraphrase \
figures like dollar amounts, percentages, or deadlines.
5. If two sources appear to give different numbers for what seems like the same metric, do NOT \
silently present both as if unremarkable, and do NOT merge or average them. First check whether \
they are actually reporting the same thing under a different label or unit — for example, one \
source may report a figure in thousands and another in millions (e.g. $102,658 thousand and \
$102.7 million are the same value), or one may be a GAAP figure and another a Non-GAAP/adjusted \
figure for the same period (these are legitimately different numbers, not an error). If the \
distinction is stated in the source labels or context (e.g. "Table 1 (GAAP)" vs "Table 2 \
(Non-GAAP)"), explain which figure is which rather than listing them side by side with no \
explanation. Only if the sources genuinely conflict with no apparent explanation should you say \
so explicitly, rather than presenting one arbitrarily as the answer.
6. NEVER invent a scale word (thousand/million/billion) for a number pulled from a table unless \
that exact word appears in the SAME source excerpt. Financial tables commonly report figures in \
thousands as a document-wide convention stated once, elsewhere, and not repeated next to every \
individual table — so a bare table figure like "102,658" must NOT be assumed to already be in \
millions, and must NOT be reported as "$102,658 million" (a fabricated unit is a much more \
serious error than reporting the number without any scale word at all). If a DIFFERENT retrieved \
source states the same metric in prose WITH an explicit unit (e.g., "we posted $321.0 million in \
net income"), prefer that explicitly-scaled figure and cite it, rather than reconstructing or \
guessing a scale for the equivalent bare number in a table. If no source in your context states \
an explicit unit for a figure you need, report the raw number exactly as given and note that its \
scale (thousands vs. millions, etc.) was not specified in the retrieved excerpt.
7. Keep answers concise and directly responsive to the question asked."""


@dataclass
class GeneratedAnswer:
    """The LLM's answer plus the source labels it had available, for UI display."""

    answer: str
    sources_used: List[str]  # e.g. ["Source 1: sample_lease_agreement.pdf, Clause 3 (RENT), page 1"]


def _format_source_label(result: RetrievalResult, index: int) -> str:
    """
    Build a human-readable citation label for one retrieved chunk.

    Falls back gracefully when clause metadata isn't present (e.g. for the
    10-K excerpt, which has no numbered clauses — see chunker.py's fallback
    strategy), so labeling works uniformly across document types.
    """
    chunk = result.chunk
    clause_number = chunk.metadata.get("clause_number")
    clause_title = chunk.metadata.get("clause_title")
    table_index = chunk.metadata.get("table_index")
    table_heading = chunk.metadata.get("table_heading")

    location = f"page {chunk.page_number}"
    if clause_number:
        clause_desc = f"Clause {clause_number}"
        if clause_title:
            clause_desc += f" ({clause_title})"
        location = f"{clause_desc}, {location}"
    elif table_index:
        # Table chunks (Phase 3) have no clause number — cite them as
        # "Table N, page P" instead, so a financial-statement answer
        # points to the actual table rather than an unhelpful bare page
        # number (which reads as if it came from ordinary prose).
        #
        # WHEN A HEADING WAS DETECTED (Phase 4 finding, real-world PDFs):
        # some financial documents place two structurally-identical
        # tables on the same page, distinguished only by a short heading
        # above each (e.g. "GAAP" vs "Non-GAAP" results). Without
        # surfacing that heading in the citation, two genuinely different
        # — and both correct — figures (e.g. GAAP vs Non-GAAP net income)
        # look like an unexplained data conflict rather than two
        # deliberately different reporting bases. See
        # src/ingestion/table_extractor.py's _find_heading_above_table.
        table_desc = f"Table {table_index}"
        if table_heading:
            table_desc += f" ({table_heading})"
        location = f"{table_desc}, {location}"

    return f"Source {index}: {chunk.source}, {location}"


def _build_context_block(results: List[RetrievalResult]) -> str:
    """
    Format retrieved chunks into a single context string, each prefixed
    with its citation label, ready to be inserted into the user prompt.
    """
    blocks = []
    for i, result in enumerate(results, start=1):
        label = _format_source_label(result, i)
        blocks.append(f"[{label}]\n{result.chunk.text}")
    return "\n\n---\n\n".join(blocks)


class Generator:
    """
    Wraps the Groq client behind a single generate() method that takes a
    question and its retrieved context, and returns a grounded answer.
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

    def generate(self, question: str, retrieved_results: List[RetrievalResult]) -> GeneratedAnswer:
        """
        Generate a grounded answer to `question` using `retrieved_results`
        as the only permitted source of information.
        """
        if not retrieved_results:
            return GeneratedAnswer(
                answer="The retrieved documents don't contain enough information to answer this question.",
                sources_used=[],
            )

        context_block = _build_context_block(retrieved_results)
        source_labels = [
            _format_source_label(r, i) for i, r in enumerate(retrieved_results, start=1)
        ]

        user_prompt = f"""Context:
{context_block}

Question: {question}

Answer the question using only the context above, citing sources by label."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # low temperature: we want factual consistency, not creativity
        )

        answer_text = response.choices[0].message.content

        return GeneratedAnswer(answer=answer_text, sources_used=source_labels)
