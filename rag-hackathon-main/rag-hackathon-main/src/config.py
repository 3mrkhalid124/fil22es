"""Central, file-based configuration.

Every magic number and model choice that was hardcoded in the notebook now
lives in config/config.yaml and is read here. The CLI and every module only
ever touch this single object, so re-running with different settings no
longer requires editing source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    cache_dir: str = "cache"

    # Artifact filenames (relative to cache/output dirs).
    chunks_json: str = "processed_semantic_chunks.json"
    embedding_matrix_npz: str = "embeddings.npz"
    evaluation_json: str = "rag_evaluation_results.json"
    retriever_config_json: str = "best_retriever_config.json"

    @property
    def chunks_path(self) -> Path:
        return Path(self.output_dir) / self.chunks_json

    @property
    def embedding_matrix_path(self) -> Path:
        return Path(self.cache_dir) / self.embedding_matrix_npz

    @property
    def evaluation_path(self) -> Path:
        return Path(self.output_dir) / self.evaluation_json

    @property
    def retriever_config_path(self) -> Path:
        return Path(self.output_dir) / self.retriever_config_json


@dataclass
class PreprocessingConfig:
    """Text cleaning and structural analysis."""

    remove_front_matter: bool = True
    max_front_scan_ratio: float = 0.35
    header_footer_repeat_ratio: float = 0.12
    min_structural_chars: int = 250
    min_section_content: int = 300
    max_section_group_chars: int = 2500
    # Tiny-fragment protection (characters / words).
    tiny_chunk_char_limit: int = 250
    tiny_chunk_word_limit: int = 40
    # Final chunk quality bands.
    min_chunk_chars: int = 250
    large_chunk_chars: int = 3000
    ideal_min_chars: int = 250
    ideal_max_chars: int = 1800


@dataclass
class ChunkingConfig:
    """Semantic chunker + post-hoc size control."""

    # LangChain SemanticChunker (percentile method).
    semantic_breakpoint_type: str = "percentile"
    semantic_percentile: float = 60.0
    add_start_index: bool = True

    # Grid search over size-control strategies (from Block 8A).
    target_max_options: List[int] = field(default_factory=lambda: [1000, 1500, 2000])
    overlap_options: List[int] = field(default_factory=lambda: [0, 1])


@dataclass
class EmbeddingConfig:
    """Dual-embedder setup.

    chunker_embedder : used ONLY by the SemanticChunker to break structural
                       units (cheap, local).
    index_embedder   : used to build the retrieval index (higher quality).
    reranker_model   : cross-encoder used for second-stage reranking.
    """

    chunker_embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_embedder: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    batch_size: int = 8
    normalize_embeddings: bool = True
    # Optional Groq cloud embedding (alternative index_embedder).
    groq_api_key: str = ""
    groq_model: str = "nomic-embed-text-v1_5"
    use_groq: bool = False


@dataclass
class RetrievalConfig:
    """Evaluation-driven retriever selection."""

    k_values: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    search_types: List[str] = field(default_factory=lambda: ["similarity", "mmr"])
    rerank_k: int = 3
    mmr_fetch_k_cap: int = 20
    mmr_diversity: float = 0.3
    # Overall-score weights: 0.7 relevance rate + 0.3 sigmoid(rerank).
    relevance_weight: float = 0.7
    rerank_weight: float = 0.3


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        def nested(obj, mapping: Dict[str, Any]) -> None:
            for key, value in mapping.items():
                if isinstance(value, dict) and hasattr(obj, key):
                    sub = getattr(obj, key)
                    nested(sub, value)
                    setattr(obj, key, sub)
                elif hasattr(obj, key):
                    setattr(obj, key, value)

        cfg = cls()
        nested(cfg, raw)
        return cfg
