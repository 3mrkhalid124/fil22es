"""End-to-end pipeline orchestrator.

Wires ingestion -> cleaning -> structural units -> semantic chunking ->
enrichment -> embeddings -> retriever selection -> evaluation into one
reproducible flow with checkpointing at every stage.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_experimental.text_splitter import SemanticChunker

from .chunking import (
    build_base_documents,
    build_structural_units,
    merge_structural_units,
)
from .cleaning import prepare_pdf_pages
from .config import AppConfig
from .embeddings import build_embedder, build_index, load_index
from .enrich import (
    enrich_and_build,
    file_coverage_report,
    load_acronym_dictionary,
)
from .ingestion import PdfInfo, ingest_all
from .models import ProcessedChunk
from .retrieval import Retriever, Reranker, RetrievalConfig, select_best_retriever

# ---------------------------------------------------------------------------
# Final safety cleanup (notebook Block 8A end-of-pipeline pass)
# ---------------------------------------------------------------------------

_STANDALONE_SECTION_RE = re.compile(
    r"^[\[\(]?\s*Section\s*:\s*\d+(?:\.\d+)*\.?\s*[\]\)]?$", re.IGNORECASE
)


def final_safety_cleanup(docs):
    """Remove standalone section fragments and extraction artifacts that
    survived every earlier filter."""
    cleaned = []
    for doc in docs:
        text = doc.page_content.strip()
        text = re.sub(r"(?im)^\s*e\d{3,5}\s*$", "", text)
        text = text.replace("-]", "]").strip()
        if not text or _STANDALONE_SECTION_RE.match(text):
            continue
        doc.page_content = text
        cleaned.append(doc)
    return cleaned


# ---------------------------------------------------------------------------
# Stage 1: process PDFs -> chunks JSON
# ---------------------------------------------------------------------------


def process_corpus(cfg: AppConfig) -> List[ProcessedChunk]:
    """Full ingest -> chunk -> enrich -> save pipeline."""
    data_dir, output_dir = Path(cfg.paths.data_dir), Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Ingestion ----------------------------------------------------
    print("\n[1/6] Ingesting PDFs...")
    pdf_infos: Dict[str, PdfInfo] = {}
    for info in ingest_all(data_dir):
        pdf_infos[Path(info.name).name.lower()] = info
    if not pdf_infos:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    # --- 2. Cleaning + structural analysis + semantic chunking -----------
    print("\n[2/6] Cleaning and chunking...")
    pre_cfg, chunk_cfg = cfg.preprocessing, cfg.chunking
    all_final_docs = []

    for name, info in pdf_infos.items():
        print(f"\n  Processing {info.name}")
        prepared, _ = prepare_pdf_pages(info.pages, pre_cfg)
        raw_units = build_structural_units(prepared)
        grouped = merge_structural_units(raw_units, pre_cfg)
        print(f"    Structural units: {len(raw_units)} raw -> {len(grouped)} grouped")

        chunker_embedder = build_embedder(cfg.embeddings, role="chunker")
        semantic_chunker = SemanticChunker(
            chunker_embedder,
            breakpoint_threshold_type=chunk_cfg.semantic_breakpoint_type,
            breakpoint_threshold_amount=chunk_cfg.semantic_percentile,
            add_start_index=chunk_cfg.add_start_index,
        )

        best_holder, variant_summary = build_base_documents(
            grouped, {}, info.name, str(info.path.resolve()),
            info.total_pages, semantic_chunker, chunk_cfg, pre_cfg,
        )
        docs = best_holder.metadata["docs"]
        docs = final_safety_cleanup(docs)
        all_final_docs.extend(docs)
        print(f"    Final chunks from {info.name}: {len(docs)}")

    # --- 3. Enrichment ---------------------------------------------------
    print("\n[3/6] Enriching metadata...")
    acronym_dictionary = load_acronym_dictionary(Path("config/acronyms.yaml"))
    chunks, per_file = enrich_and_build(all_final_docs, pdf_infos, acronym_dictionary)
    file_coverage_report(chunks, pdf_infos)

    # --- 4. Persist chunks -----------------------------------------------
    ProcessedChunk.save_all(chunks, output_dir / cfg.paths.chunks_json)
    print(f"\n[4/6] Saved {len(chunks)} chunks to {output_dir / cfg.paths.chunks_json}")
    return chunks


# ---------------------------------------------------------------------------
# Stage 2: embeddings + retriever
# ---------------------------------------------------------------------------


def build_retrieval_stack(
    cfg: AppConfig,
    chunks: Optional[List[ProcessedChunk]] = None,
):
    """Embed chunks (or load cached), select best retriever config,
    and return a ready-to-query Retriever."""
    cache_dir, output_dir = Path(cfg.paths.cache_dir), Path(cfg.paths.output_dir)

    if chunks is None:
        chunks = ProcessedChunk.load_all(output_dir / cfg.paths.chunks_json)
    print(f"\n[5/6] Embedding {len(chunks)} chunks...")
    matrix, chunks = build_index(
        chunks, cfg.embeddings,
        cache_path=cache_dir / cfg.paths.embedding_matrix_npz,
    )

    embedder = build_embedder(cfg.embeddings, role="index")
    reranker = Reranker(cfg.embeddings.reranker_model, device=cfg.embeddings.device)

    if (output_dir / cfg.paths.retriever_config_json).exists():
        import json as _json
        with (output_dir / cfg.paths.retriever_config_json).open() as fh:
            saved = _json.load(fh)
        best_cfg = RetrievalConfig(
            search_types=[saved["search_type"]],
            k_values=[saved["k"]],
            rerank_k=saved["rerank_k"],
            mmr_fetch_k_cap=cfg.retrieval.mmr_fetch_k_cap,
            mmr_diversity=cfg.retrieval.mmr_diversity,
        )
        retriever = Retriever(chunks, matrix, best_cfg, embedder, reranker)
        print(f"    Loaded saved retriever config: {saved['search_type']} k={saved['k']}")
    else:
        print("\n[6/6] Selecting best retriever configuration...")
        from .evaluation import DEFAULT_TEST_QUERIES
        best_name, best = select_best_retriever(
            chunks, matrix, cfg.retrieval, embedder, reranker,
            DEFAULT_TEST_QUERIES, output_dir / cfg.paths.retriever_config_json,
        )
        retriever = Retriever(chunks, matrix, RetrievalConfig(
            search_types=[best.search_type],
            k_values=[best.k],
            rerank_k=best.rerank_k,
            mmr_fetch_k_cap=cfg.retrieval.mmr_fetch_k_cap,
            mmr_diversity=cfg.retrieval.mmr_diversity,
        ), embedder, reranker)

    return retriever
