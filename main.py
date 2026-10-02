import argparse

from config import INDEX_DIR, PDF_DIR, TOP_K
from rag import answer, build_index


def _print_result(result: dict, show_text: bool = False) -> None:
    print(f"\n{result['answer'].strip()}\n")
    print("Sources (L2 distance, lower is closer):")
    for i, src in enumerate(result["sources"], start=1):
        print(f"  [{i}] {src['source']} p.{src['page']} score={src['score']:.4f}")
        if show_text:
            excerpt = " ".join(src["text"].split())
            print(f"      {excerpt}\n")


def cmd_ingest(args: argparse.Namespace) -> None:
    build_index(args.pdf_dir, args.index_dir, force=args.force)


def cmd_ask(args: argparse.Namespace) -> None:
    build_index()
    _print_result(answer(args.question, top_k=args.top_k), args.show_text)


def cmd_chat(args: argparse.Namespace) -> None:
    build_index()
    print("Ask a question about your documents. Type 'exit' or press Enter on an empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        try:
            result = answer(question, top_k=args.top_k)
        except Exception as e:  # e.g. a transient 503 from Gemini; keep the session alive
            print(f"Error: {e}")
            continue
        _print_result(result, args.show_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Build the index (skips if PDFs are unchanged)")
    ingest.add_argument("--pdf-dir", default=PDF_DIR)
    ingest.add_argument("--index-dir", default=INDEX_DIR)
    ingest.add_argument("--force", action="store_true", help="Rebuild even if nothing changed")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Answer one question")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="Interactive question loop")
    chat.set_defaults(func=cmd_chat)

    for p in (ask, chat):
        p.add_argument("--top-k", type=int, default=TOP_K)
        p.add_argument("--show-text", action="store_true", help="Print each source excerpt")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
