"""
CLI entrypoint for ParaLex — lets you build the index and ask questions
from the command line, with no Streamlit UI required.

WHY THIS EXISTS:
Phase 1's goal is a working, testable RAG engine — not a UI. This script
proves the pipeline works end-to-end (ingest -> chunk -> embed -> store ->
retrieve -> generate) independent of any frontend, which is exactly what
you want to demo/screenshot for a portfolio README before Phase 4 UI work
even starts.

USAGE:
    python run_pipeline.py --build              # (re)build the index from data/sample_docs
    python run_pipeline.py --ask "What is the monthly rent?"
    python run_pipeline.py                      # interactive Q&A loop
"""

import argparse
import sys

from src import config
from src.pipeline import build_index, answer_question
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.embeddings.embedder import Embedder


def print_answer(question: str, result) -> None:
    print()
    print(f"Q: {question}")
    print(f"A: {result.answer}")
    if result.sources_used:
        print()
        print("Sources:")
        for s in result.sources_used:
            print(f"  - {s}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ParaLex CLI")
    parser.add_argument("--build", action="store_true", help="(Re)build the vector index from data/sample_docs")
    parser.add_argument("--ask", type=str, help="Ask a single question and exit")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.build:
        build_index()
        if not args.ask:
            return

    # Load the shared embedder/generator once, reuse across all questions
    # in this session — this matters for interactive mode's responsiveness.
    try:
        embedder = Embedder()
        retriever = Retriever.from_saved_store(embedder=embedder)
    except FileNotFoundError:
        print("No index found. Run with --build first: python run_pipeline.py --build")
        sys.exit(1)

    try:
        generator = Generator()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.ask:
        result = answer_question(args.ask, retriever=retriever, generator=generator, top_k=args.top_k)
        print_answer(args.ask, result)
        return

    # Interactive loop
    print("ParaLex CLI — ask a question about the sample documents (Ctrl+C or 'exit' to quit)")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        result = answer_question(question, retriever=retriever, generator=generator, top_k=args.top_k)
        print_answer(question, result)


if __name__ == "__main__":
    main()
