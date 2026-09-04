"""
Custom visual styling for the ParaLex Streamlit app.

DESIGN DIRECTION ("ledger and seal", full dark theme):
A dark, document-archive feel throughout — near-black ink background,
warm paper-toned text, brass gold as the single accent color (used only
for citations, headers, and active states). Citations are rendered as
small bordered "exhibit tag" chips (the one deliberate signature
element) rather than a plain bullet list.

WHY THIS VERSION FIXES REAL CONTRAST BUGS FROM THE PREVIOUS APPROACH:
The earlier design tried to force a dark sidebar onto an otherwise
light-themed app using blanket `[data-testid="stSidebar"] * { color: ... }`
overrides. This broke wherever a Streamlit-native widget (a button, the
file uploader, a code span) had its OWN background we hadn't accounted
for — forcing light text onto an element that also had a light
background, making it invisible. The fix has two parts:
  1. The app now uses Streamlit's actual dark THEME (see
     .streamlit/config.toml, base="dark") as the primary mechanism, so
     Streamlit's own native widgets get correct, automatically-consistent
     contrast — we no longer fight Streamlit's component rendering with
     brute-force overrides.
  2. Custom CSS here is now reserved ONLY for elements Streamlit's theme
     can't control (fonts, the title's hairline rule, citation chips,
     section labels) — and every custom element explicitly pairs its own
     background AND text color together, rather than assuming inherited
     color will be readable against a background it doesn't know about.

WHY THIS IS ITS OWN MODULE:
Keeping CSS and citation-rendering markup separate from streamlit_app.py
mirrors the project's existing "logic vs. entrypoint" separation
(pipeline.py vs run_pipeline.py, app_logic.py vs streamlit_app.py).
"""

import re
from typing import List, Optional, Set

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --text: #EAE7DD;
  --bg-elevated: #1B2438;
  --brass: #C9A227;
  --brass-soft: rgba(201, 162, 39, 0.14);
  --slate: #8E97A8;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
}

/* Main title — serif, with a single hairline rule as the section's structural device */
h1 {
  font-family: 'Spectral', serif !important;
  font-weight: 600 !important;
  color: var(--text) !important;
  border-bottom: 2px solid var(--brass);
  padding-bottom: 0.5rem;
  margin-bottom: 0.3rem !important;
}

.paralex-tagline {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  color: var(--slate);
  letter-spacing: 0.02em;
  margin-top: -0.2rem;
  margin-bottom: 1.5rem;
}

/* Small uppercase section labels in the sidebar — a structural device
   grouping related controls, consistent with the monospace utility face
   used for citations and the tagline. */
.sidebar-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--brass) !important;
  margin-bottom: 0.4rem;
  margin-top: 0.2rem;
}

/* Chat messages — elevated dark card. Background AND text color are set
   together explicitly here (not left to inheritance), which is the fix
   for the "invisible answer text" bug: nested text elements inside
   Streamlit's chat message container must be targeted directly, since a
   parent-level color rule alone doesn't reliably cascade through every
   internal element Streamlit renders. */
[data-testid="stChatMessage"] {
  background-color: var(--bg-elevated);
  border-radius: 6px;
  border-left: 4px solid var(--brass);
  padding: 0.9rem 1.1rem !important;
  margin-bottom: 0.6rem;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
  color: var(--text) !important;
}

/* Source citation chips — the signature element: an "exhibit tag" feel */
.source-chip {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  color: var(--brass);
  background-color: var(--brass-soft);
  border: 1px solid var(--brass);
  border-radius: 4px;
  padding: 4px 10px;
  margin: 3px 6px 3px 0;
}

/* "Sources" expander — brass accent tying it to the citation chips inside it */
[data-testid="stExpander"] summary {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  border-left: 3px solid var(--brass);
  padding-left: 0.6rem !important;
}

/* Evidence blocks — the actual retrieved chunk text behind each citation,
   so grounding is verifiable rather than just claimed. Deliberately more
   muted than the chat message cards (smaller text, softer background) so
   it reads as supporting detail, not competing with the answer itself. */
.evidence-block {
  border-left: 3px solid var(--brass);
  background-color: var(--brass-soft);
  padding: 0.6rem 0.9rem;
  margin: 0.5rem 0;
  border-radius: 4px;
}
.evidence-block.unused {
  border-left-color: var(--slate);
  background-color: rgba(142, 151, 168, 0.08);
  opacity: 0.75;
}
.evidence-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  color: var(--brass);
  margin-bottom: 0.35rem;
}
.evidence-block.unused .evidence-label {
  color: var(--slate);
}
.evidence-used-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  color: var(--brass);
  margin-left: 0.5rem;
}
.evidence-section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--slate);
  margin: 0.8rem 0 0.3rem 0;
}
.evidence-text {
  font-family: 'Inter', sans-serif;
  font-size: 0.87rem;
  color: var(--text);
  white-space: pre-wrap;
  line-height: 1.4;
}
"""


def render_source_chips(sources: List[str]) -> str:
    """
    Render a list of citation strings (e.g. "Source 1: lease.pdf, Clause 3
    (RENT), page 1") as HTML "exhibit tag" chips, for display via
    st.markdown(..., unsafe_allow_html=True).

    Kept as a pure function (no Streamlit calls) so it's directly unit
    testable — the HTML structure is simple enough that verifying its
    output string is meaningful and doesn't require a browser.
    """
    if not sources:
        return "<em>No sources available.</em>"

    chips = "".join(f'<span class="source-chip">{_escape(s)}</span>' for s in sources)
    return f'<div>{chips}</div>'


def extract_cited_source_numbers(answer_text: str) -> Set[int]:
    """
    Parse which "Source N" citations the generated answer text actually
    mentions, e.g. extracts {1, 3} from "...(Source 1)... (Source 3)."

    WHY THIS EXISTS: retrieval returns top_k chunks, but the LLM's answer
    typically only ends up citing a subset of them — the rest were
    retrieved as candidates but weren't actually needed to answer this
    specific question. Distinguishing "used in this answer" from "also
    retrieved" in the evidence panel (see render_evidence_panel) gives a
    much clearer signal than showing all retrieved chunks as if equally
    relevant, especially as top_k grows for upload mode's larger, more
    diverse documents.

    This works by parsing the LLM's own output text rather than requiring
    a second model call — the generation prompt already instructs the
    model to cite sources as "(Source N)" (see src/generation/generator.py's
    SYSTEM_PROMPT), so the citations are already present in plain text.
    """
    return {int(n) for n in re.findall(r"Source\s+(\d+)", answer_text)}


def render_evidence_panel(
    evidence: List[dict],
    max_excerpt_length: int = 320,
    cited_indices: Optional[Set[int]] = None,
) -> str:
    """
    Render the actual retrieved chunk text behind each citation — not just
    its label — as HTML "evidence blocks".

    WHY THIS EXISTS: a bare citation label ("Source 1: lease.pdf, page 1")
    asks the user to trust that the answer came from somewhere real,
    without letting them verify it. Showing the actual retrieved excerpt
    makes the RAG system's grounding checkable rather than just claimed —
    a user (or an interviewer) can directly compare the answer against the
    exact text it was generated from.

    `evidence` is a list of {"label": str, "text": str} dicts, in the same
    order as the citation numbering ("Source 1" is evidence[0], etc.).
    `cited_indices`, if given (see extract_cited_source_numbers), separates
    evidence actually referenced in the answer from evidence that was
    retrieved but not needed — shown as a distinct, visually de-emphasized
    "Other retrieved evidence" group, so the most relevant excerpt isn't
    buried among several the answer didn't end up using.

    Excerpts default to a much shorter length (220 chars) than earlier —
    this is a real usability fix: several large full-paragraph excerpts
    stacked in a row made the panel cumbersome to scan, especially once
    upload mode's larger UPLOAD_TOP_K default returns more sources.
    """
    if not evidence:
        return "<em>No evidence available.</em>"

    def _block(item: dict, used: Optional[bool]) -> str:
        label = _escape(item["label"])
        text = item["text"]
        if len(text) > max_excerpt_length:
            text = text[:max_excerpt_length].rsplit(" ", 1)[0] + "…"
        text = _escape(text)
        # used=True -> mark as cited; used=False -> mark as "other retrieved";
        # used=None -> no citation info available, render plainly with no badge.
        badge = '<span class="evidence-used-badge">★ used in answer</span>' if used else ""
        css_class = "evidence-block unused" if used is False else "evidence-block"
        return (
            f'<div class="{css_class}">'
            f'<div class="evidence-label">{label}{badge}</div>'
            f'<div class="evidence-text">{text}</div>'
            f'</div>'
        )

    if cited_indices is None:
        # No citation info available — render everything as a flat list
        # with no "used"/"other" distinction, same as before this feature existed.
        return "".join(_block(item, used=None) for item in evidence)

    used_items = [item for i, item in enumerate(evidence, start=1) if i in cited_indices]
    other_items = [item for i, item in enumerate(evidence, start=1) if i not in cited_indices]

    html = "".join(_block(item, used=True) for item in used_items)
    if other_items:
        html += '<div class="evidence-section-label">Other retrieved evidence</div>'
        html += "".join(_block(item, used=False) for item in other_items)
    return html


def safe_markdown_text(text: str) -> str:
    """
    Escape literal dollar signs before passing text to st.markdown().

    WHY THIS EXISTS (a real bug this project's answers would otherwise
    hit constantly): st.markdown() interprets text between two "$"
    characters as LaTeX math to render (e.g. "$2.29 billion...$1.33
    billion" gets parsed as a single math expression spanning both
    figures), producing garbled, word-concatenated output — exactly the
    kind of text a financial-document assistant's answers are full of.
    Escaping "$" as "\\$" tells Streamlit's markdown renderer to treat it
    as a literal currency symbol instead of a math delimiter, which is
    always what we want here — this app never intends to render LaTeX.
    """
    return text.replace("$", "\\$")


def _escape(text: str) -> str:
    """Minimal HTML escaping for citation text before embedding in markup."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
