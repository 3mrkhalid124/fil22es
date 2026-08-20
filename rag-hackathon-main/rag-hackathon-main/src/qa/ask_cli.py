"""Interactive question runner for the Prompting + Multi-Query phase.

Lets you ask arbitrary questions against the indexed corpus and see the
full formatted answer (Recommendation / Evidence / Citation / Uncertainty).

Run from the project root (with GROQ_API_KEY set):
    python src/qa/ask_cli.py                # interactive loop
    python src/qa/ask_cli.py "your question" # one-shot
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.config import AppConfig  # noqa: E402
from src.pipeline import build_retrieval_stack  # noqa: E402
from src.qa.engine import QAEngine, format_answer  # noqa: E402
from src.qa.logger import TraceLogger  # noqa: E402

LOG_DIR = Path(r"C:\Users\Amr\Desktop\fil22es\qa_logs")


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Set it first, e.g.:")
        print("  $env:GROQ_API_KEY = 'your_key_here'")
        sys.exit(1)

    cfg = AppConfig.from_yaml(ROOT / "config" / "config.yaml")
    print("[setup] building retrieval stack (cached chunks + embeddings)...")
    retriever = build_retrieval_stack(cfg)

    from groq import Groq

    engine = QAEngine(retriever, Groq(api_key=os.environ["GROQ_API_KEY"]))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def run(question: str) -> None:
        logger = TraceLogger(path=str(LOG_DIR / f"cli_{len(list(LOG_DIR.glob('cli_*'))):03d}_trace.json"))
        engine.logger = logger
        result = engine.ask(question, show_details=True)
        logger.set("result_summary", {
            "gate": result.gate_decision,
            "overall_score": result.overall_score,
            "overall_level": result.overall_level.value if result.overall_level else None,
            "n_claims": len(result.claims),
            "validation": result.validation,
        })
        logger.save()
        print("\n" + format_answer(result))
        print(f"\n[log] trace saved to: {logger.path}")

    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
        return

    print("\nAsk questions about the corpus (Ctrl+C to quit):\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        run(question)
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()