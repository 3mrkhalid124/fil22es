#!/usr/bin/env python3
"""Medical RAG CLI.

Usage:
    python main.py process            # ingest + chunk + enrich + save chunks
    python main.py embed              # embed chunks (or load cached) + select retriever
    python main.py serve              # interactive question loop over the index
    python main.py eval               # run the saved retriever on the eval set
    python main.py run all            # everything, end to end
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import AppConfig  # noqa: E402
from src.evaluation import DEFAULT_TEST_QUERIES, run_evaluation  # noqa: E402
from src.models import ProcessedChunk  # noqa: E402
from src.pipeline import build_retrieval_stack, process_corpus  # noqa: E402


def get_retriever(cfg: AppConfig):
    return build_retrieval_stack(cfg)


def cmd_process(cfg: AppConfig) -> None:
    process_corpus(cfg)


def cmd_embed(cfg: AppConfig) -> None:
    build_retrieval_stack(cfg)


def cmd_serve(cfg: AppConfig) -> None:
    retriever = get_retriever(cfg)
    print("\nAsk questions (Ctrl+C to quit):\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        results = retriever.retrieve(question, retriever.cfg.k_values[0],
                                     retriever.cfg.rerank_k)
        for r in results:
            print(
                f"\n  [{r.rank}] (dense={r.dense_score:.3f}, "
                f"rerank={r.rerank_score:.2f}) "
                f"{r.chunk.metadata.source_file} p{r.chunk.metadata.page_numbers}"
            )
            print(f"      {r.chunk.original_text[:300]}...")


def cmd_eval(cfg: AppConfig) -> None:
    retriever = get_retriever(cfg)
    from src.embeddings import build_embedder  # noqa: E402
    embedder_name = (
        cfg.embeddings.groq_model
        if cfg.embeddings.use_groq
        else cfg.embeddings.index_embedder
    )
    run_evaluation(
        retriever, DEFAULT_TEST_QUERIES, embedder_name,
        cfg.embeddings.reranker_model,
        Path(cfg.paths.output_dir) / cfg.paths.evaluation_json,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Medical RAG pipeline")
    parser.add_argument("command", choices=["process", "embed", "serve", "eval", "run"],
                        nargs="?", default="run", help="default: run (full pipeline)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="path to config.yaml")
    parser.add_argument("--data-dir", help="override data/ dir in config")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(Path(args.config))
    if args.data_dir:
        cfg.paths.data_dir = args.data_dir

    if args.command == "run":
        cmd_process(cfg)
        cmd_embed(cfg)
        cmd_eval(cfg)
    else:
        {"process": cmd_process, "embed": cmd_embed,
         "serve": cmd_serve, "eval": cmd_eval}[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
