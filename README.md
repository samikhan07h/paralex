<div align="center">

# ⚖️ ParaLex

**A retrieval-augmented AI assistant for legal and financial documents**

Ask questions about contracts, leases, loan agreements, and financial filings — every answer is grounded in the source document and cited to its exact clause, table, or page.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.38-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Live Demo](#) · [Report a Bug](#) · [Author](#author)

</div>

---

## Project Overview

ParaLex is a Retrieval-Augmented Generation (RAG) system built to answer questions over dense, high-stakes documents — the kind of documents where a wrong answer is worse than no answer at all. It's not a wrapper around a chat API: it's a full pipeline with clause-aware chunking for legal text, real table extraction for financial statements, a measured retrieval and faithfulness evaluation layer, and a production-style Streamlit interface with document upload and feedback logging.

This project was built end-to-end — ingestion, chunking, embeddings, vector search, generation, evaluation, and UI — with every design decision backed by either a measured result or a documented tradeoff. Where something broke against a real document, the fix and its root cause are documented below rather than glossed over.

## Features

- **Clause-aware chunking** for legal documents — numbered clauses (leases, loan agreements) are kept intact as single retrievable units instead of being split mid-sentence by naive fixed-size chunking.
- **Real table extraction** for financial documents, using `pdfplumber` rather than flat text extraction, which preserves row/column relationships that plain text extraction destroys.
- **GAAP / Non-GAAP disambiguation** — when a document places two structurally identical tables side by side (a common real-world pattern), ParaLex detects the heading above each table and cites them distinctly, rather than presenting two different, both-correct figures as an unexplained conflict.
- **Document upload** — query your own PDF or DOCX files in-session, with zero permanent storage, alongside a built-in demo corpus (lease, loan agreement, 10-K excerpt, financial statements).
- **Every answer is cited** to its source document, page, clause, or table — with an optional "evidence" view showing the exact retrieved text behind each citation, distinguishing what the answer actually used from what was retrieved but not needed.
- **Feedback logging** — thumbs up/down on any answer, logged with the full question, answer, sources, and retrieval settings for later analysis.
- **A real evaluation layer** — retrieval quality (MRR, Recall@k, Precision@k) and answer faithfulness (LLM-as-judge) are measured against a hand-labeled test set, not asserted.

## Architecture — How It Works

```
                         ┌─────────────────┐
   PDF / DOCX  ───────►  │   Ingestion     │  pypdf (prose) + pdfplumber (tables)
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │    Chunking     │  clause-aware split (legal) +
                         │                 │  table extraction with heading detection
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   Embedding     │  sentence-transformers (local, free)
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  FAISS Index    │  exact cosine similarity search
                         └────────┬────────┘
                                  │
        Question  ───────►  ┌────▼────┐
                             │Retrieval│  top-k + optional relevance filter
                             └────┬────┘
                                  │
                         ┌────────▼────────┐
                         │   Generation    │  Groq LLM, strict grounding prompt
                         │                 │  (answers only from retrieved context)
                         └────────┬────────┘
                                  │
                             Cited Answer
```

Every stage above is independently tested. The chunking and table-extraction stages in particular were validated against a real, unmodified annual report (not just synthetic test data) — see [Findings from Real-World Testing](#findings-from-real-world-testing) below.

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | Custom pipeline (`src/pipeline.py`) | Full control over the ingest → chunk → embed → retrieve → generate flow, rather than a framework's opinionated chain abstractions |
| Chunking | `langchain-text-splitters` + custom clause-aware/table logic | Recursive splitting as a baseline; clause-aware regex splitting for legal text, since naive splitting measurably fragments legal clauses |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, 384-dim. Benchmarked against `all-mpnet-base-v2` (768-dim) — both achieved identical Recall@k, but MiniLM embedded **6.3× faster**, so the larger model wasn't worth the cost here |
| Vector store | FAISS (`IndexFlatIP`, exact search) | Corpus size doesn't need approximate search — exact search is fast enough and has zero accuracy loss at this scale |
| LLM | Groq (`openai/gpt-oss-120b`) | Free tier, fast inference, strong enough quality for grounded document Q&A |
| Table parsing | `pdfplumber` | Detects real grid structure — `pypdf`'s flat text extraction scrambles table rows/columns into disconnected lines |
| Frontend | Streamlit | Chat interface, document upload, feedback logging, all in pure Python |
| Testing | `pytest` | 150+ tests across ingestion, chunking, retrieval, generation, evaluation, and the app layer |

## Folder Structure

```
paralex/
├── data/
│   ├── sample_docs/          # Demo corpus: lease, loan, 10-K excerpt, financial statements
│   └── processed/            # Generated FAISS index + feedback.db (gitignored)
├── src/
│   ├── ingestion/            # PDF/DOCX loading + table extraction
│   ├── chunking/             # Clause-aware and recursive chunking strategies
│   ├── embeddings/           # Embedding model wrapper
│   ├── vectorstore/          # FAISS wrapper (build, save, load, search)
│   ├── retrieval/            # Query → embed → search → ranked results
│   ├── generation/           # Grounded prompt construction + Groq LLM call
│   ├── config.py             # All tunable settings, env-driven
│   └── pipeline.py           # Orchestrates the full ingest → index flow
├── evaluation/
│   ├── test_sets/            # 23 hand-labeled Q&A pairs across all demo documents
│   ├── metrics.py            # MRR, Recall@k, Precision@k
│   ├── faithfulness.py       # LLM-as-judge answer faithfulness scoring
│   ├── embedding_comparison.py
│   └── run_eval.py           # One command, full evaluation report
├── app/
│   ├── streamlit_app.py      # UI entrypoint
│   ├── app_logic.py          # Testable business logic (upload handling, secrets)
│   ├── styles.py             # Custom theme + citation/evidence rendering
│   └── feedback.py           # SQLite-backed feedback logging
├── tests/                    # 150+ tests
├── scripts/                  # Dev utilities (sample document generation)
├── run_pipeline.py           # CLI entrypoint (no UI needed)
├── requirements.txt
└── .env.example
```

## Installation & Setup

**Prerequisites:** Python 3.11, a free [Groq API key](https://console.groq.com).

```bash
git clone https://github.com/your-username/paralex.git
cd paralex

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your GROQ_API_KEY
```

## Usage

**Command line** (no UI, fastest way to verify the pipeline works):
```bash
python run_pipeline.py --build                        # builds the demo index
python run_pipeline.py --ask "What is the monthly rent?"
```

**Web app:**
```bash
streamlit run app/streamlit_app.py
```
Opens a chat interface at `localhost:8501` with two modes: query the built-in demo documents, or upload your own PDF/DOCX to query instead.

**Run the test suite:**
```bash
pytest tests/ -v
```

**Run the full evaluation** (retrieval quality + faithfulness, real Groq API calls):
```bash
python -m evaluation.run_eval
```

## Evaluation Results

Measured against a hand-labeled set of 23 questions spanning all four demo documents (lease, loan agreement, 10-K excerpt, financial statements).

**Retrieval quality** (`top_k=2`, chosen empirically — see below):

| Metric | Score |
|---|---|
| MRR | 0.917 |
| Recall@2 | 1.000 |
| Precision@2 | 0.500 |

`top_k` was tested at 1, 2, 4, and 6. Recall and MRR plateau at `k=2` — every correct source is found within the top 2 results, and going higher only dilutes precision with irrelevant chunks. `k=2` is the smallest value that achieves maximum retrieval quality on this corpus.

**Answer faithfulness** (LLM-as-judge, scoring whether every claim in an answer is actually supported by its retrieved context):

| Metric | Score |
|---|---|
| Average faithfulness | 4.83 / 5 |
| Faithful rate | 94.4% (17 of 18 answers) |

The one flagged answer wasn't a hallucination — it was a citation attribution mix-up between two adjacent clauses covering related topics (both facts were correct; the labels were swapped). Documented as a known, narrow failure mode rather than smoothed over.

**Embedding model comparison:**

| Model | Dim | MRR | Recall@k | Embed time |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.891 | 1.000 | 0.63s |
| all-mpnet-base-v2 | 768 | 0.935 | 1.000 | 3.93s |

The larger model's modest MRR gain (+0.043) didn't justify a 6.3× slowdown given both achieved perfect recall — MiniLM was kept as the default.

## Findings from Real-World Testing

Everything above was validated against a small, controlled demo corpus. Testing the "upload your own document" feature against a real, unmodified 176-page annual report surfaced three genuine issues that the demo corpus never exercised — each traced to its actual root cause and fixed, not patched around:

1. **Split currency columns.** Real financial tables often typeset the `$` symbol in its own PDF table column, separate from the number. Naive extraction produced a table full of ghost columns and split every figure into two cells (`"$"`, `"1,330,383"`). Fixed by detecting and merging any column consisting entirely of bare currency symbols.

2. **GAAP vs. Non-GAAP tables.** The same document placed two structurally identical financial tables on one page, distinguished only by a heading ("GAAP" / "Non-GAAP") sitting above each — invisible to a table-grid parser. Without it, two different, both-correct figures looked like an unexplained data conflict. Fixed by detecting short, label-like text immediately above each table's bounding box and surfacing it directly in citations (`Table 1 (GAAP), page 24`).

3. **Unit fabrication risk.** Real financial statements often state their unit convention ("in thousands") once, in a footnote elsewhere in the document, and never repeat it near every table. An LLM reading a bare table figure can invent an incorrect scale word. Mitigated with an explicit generation-prompt rule: never invent a scale for a number unless that exact unit appears in the same excerpt, and prefer an explicitly-scaled figure from prose when one is available.

## Known Limitations

- **Retrieval can favor prose over tables.** A full sentence with high lexical overlap to a query can occasionally outrank the structurally correct table, even after enriching table captions with line-item names. Mitigated, not eliminated, by retrieving more than one chunk per query.
- **Feedback and the uploaded-document index are ephemeral on free-tier deployment.** Streamlit Community Cloud's storage resets on app restart — feedback logged and documents uploaded in a session won't survive a redeploy.
- **No reranking or cross-encoder stage.** Retrieval is single-pass dense similarity search. A production system handling much larger or more heterogeneous document sets would likely benefit from a reranking step.

## Future Improvements

- Cross-encoder reranking for retrieval precision on larger, more diverse document sets
- Persistent storage for feedback and uploaded-document indexes (e.g. a hosted database instead of local SQLite/FAISS)
- Multi-document cross-referencing (asking questions that span more than one uploaded document)
- Automatic promotion of well-answered, well-rated questions into the evaluation set over time

## Testing

150+ tests across every layer of the pipeline: ingestion, chunking, embeddings, vector store, retrieval, generation, evaluation metrics, faithfulness scoring, table extraction, and the Streamlit app's business logic. Network-independent tests (the majority) run without any API key; a smaller set of integration tests exercise the real Groq API and are skipped automatically if `GROQ_API_KEY` isn't set.

```bash
pytest tests/ -v
```

## Author

**Md Sami Ahmad**

📧 [samikhan4jnu@gmail.com](mailto:samikhan4jnu@gmail.com)
🔗 GitHub · LinkedIn

---

<div align="center">
<sub>Built as an end-to-end demonstration of production RAG engineering — not a tutorial clone.</sub>
</div>
