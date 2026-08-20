"""Embedding backends and index construction.

Two backends are supported, matching the notebook's dual-model setup:
  - Local  : SentenceTransformer (chunker + bge-base-en-v1.5 index embedder)
  - Groq   : nomic-embed-text via the Groq REST API (cheap cloud alternative)

Embeddings are saved once to cache as a single .npz file so a crash or
restart never forces re-embedding.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import EmbeddingConfig
from .models import ProcessedChunk

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """SentenceTransformer-based embedder (chunker or index).

    Also implements the LangChain Embeddings interface (embed_documents /
    embed_query) so it can be passed directly to SemanticChunker, which
    requires a LangChain embedder.
    """

    def __init__(self, model_name: str, cfg: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=cfg.device)
        self.batch_size = cfg.batch_size
        self.normalize = cfg.normalize_embeddings

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list) -> list:
        return self.encode(list(texts)).tolist()

    def embed_query(self, text: str) -> list:
        return self.encode([text])[0].tolist()


class GroqEmbedder:
    """Groq-hosted nomic-embed-text. Uses "search_document"/"search_query"
    task prefixes, which measurably improve retrieval matching."""

    def __init__(self, cfg: EmbeddingConfig):
        from groq import Groq

        api_key = cfg.groq_api_key or Path.home().joinpath(".groq_key").read_text().strip()
        self.client = Groq(api_key=api_key)
        self.model = cfg.groq_model
        self._dimension = 768

    def encode(self, texts: List[str], prefix: str = "search_document: ") -> np.ndarray:
        all_vecs: List[np.ndarray] = []
        for start in range(0, len(texts), 200):
            batch = texts[start : start + 200]
            resp = self.client.embeddings.create(
                model=self.model, input=[prefix + t for t in batch]
            )
            batch_vecs = {d.index: np.array(d.embedding, dtype=np.float32) for d in resp.data}
            all_vecs.extend(batch_vecs[i] for i in sorted(batch_vecs))
        return np.stack(all_vecs)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], prefix="search_query: ")[0]

    @property
    def dimension(self) -> int:
        return self._dimension


def build_embedder(cfg: EmbeddingConfig, role: str = "index"):
    """role='chunker' uses the cheap local chunker embedder; 'index' uses
    the configured index backend (Groq if enabled, else local bge)."""
    if role == "chunker":
        return LocalEmbedder(cfg.chunker_embedder, cfg)
    if cfg.use_groq:
        return GroqEmbedder(cfg)
    return LocalEmbedder(cfg.index_embedder, cfg)


# ---------------------------------------------------------------------------
# Index construction + persistence
# ---------------------------------------------------------------------------


def build_index(
    chunks: List[ProcessedChunk],
    cfg: EmbeddingConfig,
    embedder=None,
    cache_path: Optional[Path] = None,
) -> tuple:
    """Embed chunk.original_text, validate, and save to cache.

    Returns (embedding_matrix, chunks). If a cached .npz exists and matches
    chunk count, it is loaded instead of re-embedding.
    """
    embedder = embedder or build_embedder(cfg, role="index")
    texts = [c.original_text.strip() for c in chunks]

    if cache_path and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as data:
            matrix = data["matrix"]
            if matrix.shape[0] == len(chunks):
                print(f"    Loaded cached embeddings: {matrix.shape}")
                return matrix, chunks

    matrix = embedder.encode(texts)
    assert matrix.shape[0] == len(chunks), (
        f"Embedding/chunk count mismatch: {matrix.shape[0]} vs {len(chunks)}"
    )

    # Validation: no NaN/zero vectors, normalized if requested.
    norms = np.linalg.norm(matrix, axis=1)
    assert (norms > 0).all(), "Zero-norm embedding detected"
    if (np.isnan(matrix)).any():
        raise ValueError("NaN values in embedding matrix")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, matrix=matrix)
        print(f"    Saved embeddings to {cache_path}")

    return matrix, chunks


def load_index(cache_path: Path):
    with np.load(cache_path, allow_pickle=True) as data:
        return data["matrix"]
