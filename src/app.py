import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ingest import run_ingestion, INDEX_PATH
from retriever import Retriever
from qa_chain import QAChain


BANNER = """
╔══════════════════════════════════════════════════════╗
║        EU Taxonomy RAG — Question Answering CLI      ║
║  Answers are based solely on the EU Taxonomy FAQ.    ║
║  Type 'exit' or 'quit' to stop.                      ║
╚══════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EU Taxonomy RAG — CLI interface"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Re-build the vector store from data/faq.md",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--faq",
        type=str,
        default="data/faq.md",
        help="Path to the FAQ markdown file (default: data/faq.md)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        dest="show_context",
        help="Print the retrieved context passages before the answer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ingestion
    if args.ingest or not INDEX_PATH.exists():
        if not INDEX_PATH.exists():
            print("[app] Vector store not found — running ingestion first …")
        run_ingestion(faq_path=args.faq)

    # Load retriever & QA chain
    retriever = Retriever(top_k=args.top_k)
    qa = QAChain()

    print(BANNER)

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[app] Goodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        # Retrieve relevant chunks
        chunks = retriever.retrieve(question)

        if args.show_context:
            print("\n── Retrieved context ──────────────────────────────────")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n[Passage {i}]\n{chunk}")
            print("\n── Answer ─────────────────────────────────────────────")

        # Generate answer
        answer = qa.answer(question, chunks)
        print(f"\nAnswer: {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
