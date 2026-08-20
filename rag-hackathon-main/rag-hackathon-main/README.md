# Medical RAG — from Notebook to Professional Project

A modular, reproducible retrieval-augmented-generation pipeline for medical PDFs, refactored from the original Colab notebook (`RAG_SYSTEM(4).ipynb`). The entire ~21,000-line notebook has been reorganized into six focused modules behind a single CLI, with every tunable parameter moved to configuration files.

## What changed and why

| Notebook problem | Project solution |
|---|---|
| One 21k-line notebook; ~8k-line monolithic cell | `src/chunking.py` (~250 lines), one function per responsibility |
| Cell-to-cell dependencies via `globals()` checks (`if "final_docs" not in globals()`) | Explicit function arguments and typed dataclasses; zero runtime coupling |
| Hard-coded Colab paths (`/content/DATA`, `/content/...`) | `config/config.yaml` `paths` section; works anywhere |
| Magic numbers scattered across blocks (250, 300, 65, 0.3...) | Single `config/config.yaml`; change a threshold, re-run one command |
| Two disconnected embedding models (all-MiniLM for chunking, bge for indexing) | Documented dual-embedder design in `EmbeddingConfig` with Groq as optional backend |
| Acronym dictionary buried in code | `config/acronyms.yaml` — add terms without touching Python |
| Chunk list lost on crash | `ProcessedChunk.save_all/load_all` + compressed `.npz` embedding cache |

## Project structure

```text
rag_project/
├── main.py                  # CLI: process / embed / serve / eval / run
├── requirements.txt
├── config/
│   ├── config.yaml          # ALL tunables: thresholds, models, paths
│   └── acronyms.yaml        # extend the acronym expansion dictionary
├── src/
│   ├── __init__.py
│   ├── models.py            # ChunkMetadata + ProcessedChunk (with save/load)
│   ├── config.py            # YAML loading → typed dataclasses
│   ├── ingestion.py         # PDF discovery, SHA-256 hashing, extraction, cleaning
│   ├── cleaning.py          # headers/footers, front matter, TOC, section hierarchy
│   ├── chunking.py          # structural units → SemanticChunker → size-control grid
│   ├── enrich.py            # visual refs, clinical flags, acronyms, JSON build
│   ├── embeddings.py        # local (bge) + Groq (nomic-embed) backends, caching
│   ├── retrieval.py         # similarity, MMR, cross-encoder rerank, config sweep
│   ├── evaluation.py        # eval-set runner + JSON report export
│   └── pipeline.py          # orchestrator wiring all stages with checkpoints
├── data/                    # <-- drop your additional PDFs here
├── output/                  # processed_semantic_chunks.json, eval reports
└── cache/                   # embeddings.npz (never recompute after crash)
```

## How to use it

### 1. Install

```bash
cd rag_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your documents

Copy any PDF (textbook chapters, journal articles, guideline PDFs) into `data/`. No code changes are needed — the ingestion stage auto-discovers all `*.pdf` files and fingerprints each with SHA-256 for provenance.

### 3. Run the pipeline

```bash
python main.py process   # ingest + clean + chunk + enrich -> output/processed_semantic_chunks.json
python main.py embed     # embed (cached after first run) + pick best retriever config
python main.py eval      # run the 5-question evaluation set -> output/rag_evaluation_results.json
python main.py run     # everything end to end (command is optional)
```

### 4. Ask questions

```bash
python main.py serve
```

This opens an interactive loop over your index using the *automatically selected* best configuration (the same `similarity × k × rerank_k` sweep from notebook Block 11, now persisted to `output/best_retriever_config.json` and reused on every subsequent run).

### 5. Programmatic usage (for your next notebook or a FastAPI service)

```python
from src.config import AppConfig
from src.pipeline import process_corpus, build_retrieval_stack
from src.models import ProcessedChunk

cfg = AppConfig.from_yaml("config/config.yaml")
chunks = process_corpus(cfg)                          # or: ProcessedChunk.load_all("output/processed_semantic_chunks.json")
retriever = build_retrieval_stack(cfg, chunks=chunks)

for r in retriever.retrieve("How do ACE inhibitors work?", k=5, rerank_k=3):
    print(r.rank, r.chunk.metadata.section_title, r.chunk.original_text[:200])
```

## Configuration quick reference

The most impactful settings, with their notebook origins:

| Setting | Default | Meaning |
|---|---|---|
| `chunking.semantic_percentile` | 65 | The notebook's selected breakpoint (was hard-coded) |
| `embeddings.index_embedder` | BAAI/bge-base-en-v1.5 | Final retrieval index (768-d) |
| `embeddings.use_groq` | false | Flip to true to use Groq `nomic-embed-text-v1_5` (8192-token context, cheap) |
| `retrieval.k_values` | [3,5,10,20] | Grid searched automatically; best k persisted |
| `preprocessing.remove_front_matter` | true | Skips TOC/copyright pages detected per-file |

## Extending the corpus

When you add new PDFs, simply re-run `python main.py process`. Because chunks are saved with file hashes and the embedding matrix is cached by chunk count, you can also script incremental additions — re-run processes the whole `data/` folder, and unchanged files re-extract in seconds. To change the acronym dictionary, edit `config/acronyms.yaml` and re-run `process` (acronym expansion happens at chunk-construction time).

## Adding new capabilities

The module boundaries double as extension points: a new parser (e.g., unstructured.io partition) belongs in `src/ingestion.py` returning the same per-page list; a new retriever (hybrid BM25, RRF) belongs in `src/retrieval.py` alongside `similarity_search`/`mmr_search`; and any new metadata field goes in `ChunkMetadata` in `src/models.py`, surviving JSON round-trips automatically.
