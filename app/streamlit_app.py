"""
ParaLex Streamlit app — Phase 4 UI layer over the tested Phase 1-3 pipeline.

This file handles UI rendering only. See app/app_logic.py for the testable
business logic (uploaded-file processing, secrets bridging, cache-key
computation) — see that module's docstring for why the split exists.

ARCHITECTURE NOTES:

Two document modes, selected in the sidebar:
  - "Demo Documents": queries the pre-built, disk-persisted FAISS index over
    our tested sample lease/loan/10-K/financial-statements corpus.
  - "Upload Your Own": builds a fresh, in-memory-only index from whatever
    the user uploads, scoped to their browser session. Nothing uploaded is
    written to a permanent location — files land in a temp directory that
    gets removed immediately after processing, and the resulting index
    lives only in st.session_state, not on disk.
"""

import sys
from pathlib import Path

# Streamlit adds THIS SCRIPT's own folder (app/) to sys.path when it runs —
# not the project root — so absolute imports like `from app.app_logic import
# ...` and `from src import config` would otherwise fail with
# ModuleNotFoundError regardless of which directory `streamlit run` is
# invoked from. Explicitly prepending the project root (this file's parent's
# parent) fixes this for local runs AND for Streamlit Community Cloud
# deployment (Phase 5), which has the same path behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# set_page_config() MUST be the very first Streamlit command executed in the
# script — calling ANY other st.* command first (including just accessing
# st.secrets, which can render a warning as a side effect) raises
# StreamlitAPIException. So this comes immediately after `import streamlit`,
# before even the secrets-bridging logic below.
st.set_page_config(page_title="ParaLex", page_icon="⚖️", layout="wide")

from app.app_logic import (
    bridge_secrets_to_env,
    secrets_file_exists,
    uploaded_files_signature,
    build_retriever_from_uploads,
)
from app.styles import CUSTOM_CSS, render_source_chips, render_evidence_panel, safe_markdown_text, extract_cited_source_numbers
from app.feedback import FeedbackEntry, log_feedback, get_feedback_summary

# Only touch st.secrets if a real secrets.toml exists — accessing it
# otherwise renders a visible "No secrets found" warning in the app as a
# side effect of the property access itself, which is just noise for local
# development (where config comes from .env instead). See
# app_logic.secrets_file_exists's docstring for details.
if secrets_file_exists():
    bridge_secrets_to_env(st.secrets)

from src import config
from src.pipeline import build_index
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# --- Cached, expensive-to-create resources ---

@st.cache_resource(show_spinner=False)
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner=False)
def get_generator():
    """Returns (Generator, error_message). error_message is None on success."""
    try:
        return Generator(), None
    except ValueError as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def get_demo_retriever(_embedder: Embedder) -> Retriever:
    """
    Load (or build, on first run) the persisted demo index. The leading
    underscore on _embedder tells st.cache_resource not to try hashing
    that argument — the cache key is just this function's identity, which
    is correct since there's only ever one demo index.
    """
    try:
        return Retriever.from_saved_store(embedder=_embedder)
    except FileNotFoundError:
        build_index(embedder=_embedder)
        return Retriever.from_saved_store(embedder=_embedder)


# --- Sidebar ---

with st.sidebar:
    st.title("⚖️ ParaLex")

    st.markdown('<div class="sidebar-label">Document Source</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Document source",
        ["Demo Documents", "Upload Your Own"],
        label_visibility="collapsed",
        help="Try the built-in sample lease, loan, 10-K, and financial statements, "
             "or upload your own PDF/DOCX files to query instead.",
    )

    # Reset chat history when switching modes, so a previous answer's
    # sources are never mistaken for coming from the newly selected corpus.
    if st.session_state.get("_last_mode") != mode:
        st.session_state.messages = []
        st.session_state["_last_mode"] = mode

    uploaded_files = None
    if mode == "Upload Your Own":
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX file(s)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        st.caption("Files are processed in-memory for this session only — nothing is stored permanently.")
    else:
        st.markdown("**Included demo documents:**")
        st.caption(
            "• Residential lease agreement\n\n"
            "• Commercial loan agreement\n\n"
            "• 10-K excerpt (MD&A)\n\n"
            "• Financial statements (income statement + balance sheet)"
        )

    st.divider()
    st.markdown('<div class="sidebar-label">Retrieval Settings</div>', unsafe_allow_html=True)
    # Demo Documents and Upload Your Own get DIFFERENT default top_k values,
    # not one global default. config.TOP_K=2 is empirically measured against
    # our small, curated demo corpus (Phase 2) and stays exact for that mode.
    # An arbitrary user-uploaded document can be far larger and topically
    # diverse — retrieving only 2 chunks risks missing an answer that's
    # genuinely present, which is exactly the failure mode a real test run
    # surfaced (a large annual report needed top_k=4 to answer correctly).
    # See src/config.py's UPLOAD_TOP_K for the full reasoning.
    default_top_k = config.TOP_K if mode == "Demo Documents" else config.UPLOAD_TOP_K
    top_k = st.slider(
        "Chunks retrieved per question (top_k)",
        min_value=1, max_value=6, value=default_top_k,
        key=f"top_k_slider_{mode}",
        help="Demo Documents defaults to 2 (measured to give the best precision on the small "
             "built-in corpus). Upload Your Own defaults to 4, since larger uploaded documents "
             "need more chunks retrieved to reliably surface the right answer.",
    )
    # min_score is an additional relevance filter applied only in upload
    # mode — see Retriever.retrieve()'s docstring for why demo mode
    # intentionally leaves this off (preserving Phase 2's exact measured
    # behavior) while upload mode benefits from a safety net against
    # padding the LLM's context with genuinely irrelevant chunks once a
    # larger top_k exhausts a query's true matches.
    min_score = config.MIN_RELEVANCE_SCORE if mode == "Upload Your Own" else None

    st.divider()
    # Model info, feedback stats, and author details are useful but not
    # what most users need front-and-center every time — collapsing them
    # keeps the sidebar's primary controls (document source, retrieval
    # settings) reachable without scrolling on smaller screens.
    with st.expander("Model Info"):
        st.caption(f"Embedding: `{config.EMBEDDING_MODEL.split('/')[-1]}`")
        st.caption(f"LLM: `{config.GROQ_MODEL}`")

    feedback_summary = get_feedback_summary()
    if feedback_summary["up"] or feedback_summary["down"]:
        with st.expander("Feedback Logged"):
            st.caption(f"👍 {feedback_summary['up']}   👎 {feedback_summary['down']}")

    with st.expander("About"):
        st.markdown(
            "**Md Sami Ahmad**\n\n"
            "📧 [samikhan4jnu@gmail.com](mailto:samikhan4jnu@gmail.com)\n\n"
            "🔗 [GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-username)"
        )


# --- Resolve the active retriever for this render ---

embedder = get_embedder()
generator, generator_error = get_generator()

active_retriever = None
retriever_status = None

if mode == "Demo Documents":
    with st.spinner("Loading demo index..."):
        active_retriever = get_demo_retriever(embedder)
else:
    if not uploaded_files:
        retriever_status = "Upload one or more documents in the sidebar to get started."
    else:
        sig = uploaded_files_signature(uploaded_files)
        if st.session_state.get("_upload_sig") != sig:
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                try:
                    st.session_state["_upload_retriever"] = build_retriever_from_uploads(uploaded_files, embedder)
                    st.session_state["_upload_sig"] = sig
                except Exception as e:
                    st.session_state["_upload_retriever"] = None
                    retriever_status = f"Couldn't process the uploaded file(s): {e}"
        active_retriever = st.session_state.get("_upload_retriever")


# --- Main chat area ---

st.title("ParaLex")
st.markdown(
    '<div class="paralex-tagline">Ask questions about contracts, leases, and financial filings — every claim is grounded and cited to its source.</div>',
    unsafe_allow_html=True,
)

if generator_error:
    st.error(f"⚠️ {generator_error}")
elif retriever_status:
    st.info(retriever_status)

if "messages" not in st.session_state:
    st.session_state.messages = []

AVATARS = {"user": "🗂️", "assistant": "⚖️"}


def render_feedback_buttons(msg_id: str, question: str, answer: str, sources: list, mode_at_time: str, top_k_at_time: int) -> None:
    """
    Render thumbs up/down buttons for one assistant answer, logging the
    result via app.feedback.log_feedback(). Once a message has been rated,
    subsequent reruns show a simple confirmation instead of the buttons —
    both to avoid double-logging the same rating on every Streamlit rerun
    (which happens on nearly any UI interaction) and because there's no
    need to let a rating be re-submitted.
    """
    rated_ids = st.session_state.setdefault("_rated_message_ids", set())
    if msg_id in rated_ids:
        st.caption("✓ Feedback recorded — thank you.")
        return

    col_up, col_down, _ = st.columns([0.06, 0.06, 0.88])
    with col_up:
        if st.button("👍", key=f"fb_up_{msg_id}"):
            log_feedback(FeedbackEntry(
                question=question, answer=answer, sources=sources,
                rating="up", mode=mode_at_time, top_k=top_k_at_time,
            ))
            rated_ids.add(msg_id)
            st.rerun()
    with col_down:
        if st.button("👎", key=f"fb_down_{msg_id}"):
            log_feedback(FeedbackEntry(
                question=question, answer=answer, sources=sources,
                rating="down", mode=mode_at_time, top_k=top_k_at_time,
            ))
            rated_ids.add(msg_id)
            st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
        st.markdown(safe_markdown_text(msg["content"]))
        if msg.get("evidence"):
            with st.expander("Sources & Evidence"):
                st.markdown(render_source_chips([e["label"] for e in msg["evidence"]]), unsafe_allow_html=True)
                # Evidence text is collapsed by default behind a toggle —
                # several full excerpts stacked in a row made the panel
                # cumbersome to scan, especially with upload mode's larger
                # default top_k returning more sources.
                if st.toggle("Show retrieved text", key=f"show_evidence_{msg['id']}"):
                    cited = extract_cited_source_numbers(msg["content"])
                    st.markdown(render_evidence_panel(msg["evidence"], cited_indices=cited), unsafe_allow_html=True)
        if msg["role"] == "assistant":
            render_feedback_buttons(
                msg_id=msg["id"], question=msg["question"], answer=msg["content"],
                sources=[e["label"] for e in msg.get("evidence", [])],
                mode_at_time=msg["mode"], top_k_at_time=msg["top_k"],
            )

question = st.chat_input("Ask a question about your documents...")

if question and active_retriever and generator and not generator_error:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(safe_markdown_text(question))

    msg_id = f"msg_{len(st.session_state.messages)}"

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            results = active_retriever.retrieve(question, top_k=top_k, min_score=min_score)
            generated = generator.generate(question, results)
        st.markdown(safe_markdown_text(generated.answer))

        # results and generated.sources_used are index-aligned (both built
        # from the same retrieved_results list in Generator.generate()),
        # so zipping them pairs each citation label with its actual
        # retrieved text — this is what makes the evidence panel possible.
        evidence = [
            {"label": label, "text": r.chunk.text}
            for label, r in zip(generated.sources_used, results)
        ]
        if evidence:
            with st.expander("Sources & Evidence"):
                st.markdown(render_source_chips([e["label"] for e in evidence]), unsafe_allow_html=True)
                if st.toggle("Show retrieved text", key=f"show_evidence_{msg_id}"):
                    cited = extract_cited_source_numbers(generated.answer)
                    st.markdown(render_evidence_panel(evidence, cited_indices=cited), unsafe_allow_html=True)

        render_feedback_buttons(
            msg_id=msg_id, question=question, answer=generated.answer,
            sources=[e["label"] for e in evidence], mode_at_time=mode, top_k_at_time=top_k,
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": generated.answer,
        "evidence": evidence,
        "question": question,
        "mode": mode,
        "top_k": top_k,
        "id": msg_id,
    })
elif question and not active_retriever:
    st.warning("Please select or upload a document set before asking a question.")
