"""Retrieval: dense similarity, MMR, cross-encoder reranking, and the
evaluation-driven retriever configuration sweep from Block 11.

The design keeps a single public entry point — `Retriever.retrieve()` —
while the config-selection sweep is a separate, optional stage
(`select_best_retriever`).
"""

from __future__ import annotations
from .nlp_utils import get_nlp
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import re
from .config import RetrievalConfig
from .embeddings import build_embedder
from .models import ProcessedChunk

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def dense_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity (matrix must be normalized) via dot product."""
    return matrix @ query_vec


def similarity_search(scores: np.ndarray, k: int) -> List[int]:
    k = min(k, len(scores))
    return list(np.argsort(scores)[::-1][:k])


def mmr_search(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    k: int,
    fetch_k: int = 20,
    diversity: float = 0.3,
) -> List[int]:
    """Maximal Marginal Relevance over pre-normalized vectors.

    diversity 0 -> pure relevance; 1 -> pure diversity.
    """
    scores = matrix @ query_vec
    fetch_k = min(fetch_k, len(matrix))
    candidates = list(np.argsort(scores)[::-1][:fetch_k])
    if not candidates:
        return []

    selected = [candidates[0]]
    while len(selected) < k and len(selected) < len(candidates):
        best_idx, best_score = None, -float("inf")
        for cand in candidates:
            if cand in selected:
                continue
            relevance = float(scores[cand])
            max_sim = max(float(matrix[cand] @ matrix[s]) for s in selected)
            score = (1 - diversity) * relevance - diversity * max_sim  # يطابق التوثيق
            if score > best_score:
                best_idx, best_score = cand, score
        selected.append(best_idx)
    return selected


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Thin wrapper around a cross-encoder."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, texts: List[str], top_k: Optional[int] = None) -> List[tuple]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[: top_k or len(texts)]
        return [(int(idx), float(score)) for idx, score in ranked]


# ---------------------------------------------------------------------------
# Relevance heuristic (the notebook's keyword-based oracle)
# ---------------------------------------------------------------------------


# Domain abbreviation/synonym equivalences that lemmatization alone
# can't bridge (these aren't inflectional variants, they're different
# surface forms of the same concept).
_SYNONYM_CANONICAL = {
    "cyclic gmp": "cgmp",
    "cgmp": "cgmp",
}


def _lemmatize_phrase(nlp, phrase: str) -> List[str]:
    # Disable components we don't need for lemmatization — parser/NER are
    # the expensive parts and add nothing here.
    with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
        doc = nlp(phrase.lower())
    return [tok.lemma_.lower() for tok in doc if tok.is_alpha]


def is_relevant(chunk: ProcessedChunk, keywords: List[str]) -> bool:
    """Keyword-based relevance oracle using spaCy lemmatization, so
    "nitrates" matches "Nitrate" and "treated" matches "treatment" without
    relying on brittle suffix-stripping rules."""
    nlp = get_nlp()
    text_tokens = _lemmatize_phrase(nlp, chunk.original_text)
    text_token_set = set(text_tokens)

    matches = 0
    for kw in keywords:
        kw_tokens = _lemmatize_phrase(nlp, kw)
        if not kw_tokens:
            continue

        canonical = _SYNONYM_CANONICAL.get(" ".join(kw_tokens))
        if canonical and canonical in text_token_set:
            matches += 1
            continue

        if len(kw_tokens) == 1:
            hit = kw_tokens[0] in text_token_set
        else:
            hit = any(
                text_tokens[i : i + len(kw_tokens)] == kw_tokens
                for i in range(len(text_tokens) - len(kw_tokens) + 1)
            )
        if hit:
            matches += 1

    return matches >= min(2, len(keywords))

def extract_query_keywords(question: str, stop_words: tuple = (
    "what", "is", "are", "how", "which", "that", "the", "and", "or",
    "of", "in", "to", "for", "on", "with", "by", "does", "do", "it",
    "a", "an", "their", "their", "its", "this", "was", "were", "from",
    "when", "why", "can", "should", "would", "determines",
)) -> List[str]:
    """Lowercased content words from the question (drops question words)."""
    tokens = re.sub(r"[^a-z0-9 &/-]", " ", question.lower()).split()
    return [t for t in tokens if t not in stop_words and len(t) >= 3]


def is_low_quality_candidate(question: str, text: str) -> bool:
    """True if the candidate is a junk/low-information match that should be
    removed before reranking (e.g. short generic text riding on a single
    shared keyword)."""
    body = re.sub(r"\s+", " ", text).strip()
    keywords = extract_query_keywords(question)
    if len(body) < 200:
        # Short text only survives if it contains at least one question keyword.
        if not any(kw in body.lower() for kw in keywords):
            return True
    # A chunk whose ENTIRE overlap with the query is one 3-4 letter token
    # and nothing else meaningful is almost certainly a keyword trap.
    shared = [kw for kw in keywords if kw in body.lower()]
    if shared and len(shared) == 1 and len(shared[0]) <= 4 and len(body) < 400:
        return True
    return False

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    chunk: ProcessedChunk
    rank: int
    dense_score: float
    rerank_score: Optional[float]


class Retriever:
    """End-to-end retriever: embed query -> candidate search -> rerank."""

    def __init__(
        self,
        chunks: List[ProcessedChunk],
        matrix: np.ndarray,
        cfg: RetrievalConfig,
        embedder,
        reranker: Reranker,
    ):
        self.chunks = chunks
        self.matrix = matrix
        self.cfg = cfg
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, k: int, rerank_k: int) -> List[RetrievalResult]:
        query_vec = np.asarray(
            self.embedder.encode([query])[0], dtype=np.float32
        )
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        search_type = self.cfg.search_types[0]
        if search_type == "mmr":
            indices = mmr_search(
                query_vec, self.matrix, k, fetch_k=self.cfg.mmr_fetch_k_cap,
                diversity=self.cfg.mmr_diversity,
            )
        else:
            indices = similarity_search(dense_similarity(query_vec, self.matrix), k)

        # Keep candidate_matrix_indices[i] as the ONLY source of truth for
        # "which row of `matrix` does candidates[i] correspond to". Filtering
        # out low-quality candidates must never be allowed to desync
        # `candidates` from `indices` -- that was the bug: `idx` returned by
        # the reranker indexes into `candidates` (post-filter), but was
        # previously used to index into `indices` (pre-filter) when
        # computing dense_score, silently attaching the wrong chunk's dense
        # score whenever any candidate got filtered out.
        candidate_matrix_indices = [
            i for i in indices
            if not is_low_quality_candidate(query, self.chunks[i].original_text)
        ]
        if not candidate_matrix_indices:
            # Total-quality-filter failure: fall back to all original candidates
            # (never return an empty result set).
            candidate_matrix_indices = list(indices)
        candidates = [self.chunks[i] for i in candidate_matrix_indices]

        reranked = (
            self.reranker.rerank(query, [c.original_text for c in candidates], top_k=rerank_k)
            if self.cfg.rerank_k > 0 else [(j, 0.0) for j in range(len(candidates))]
        )

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(reranked, start=1):
            results.append(
                RetrievalResult(
                    chunk=candidates[idx],
                    rank=rank,
                    dense_score=float(
                        np.dot(self.matrix[candidate_matrix_indices[idx]], query_vec)
                    ),
                    rerank_score=score,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Configuration selection sweep
# ---------------------------------------------------------------------------


@dataclass
class ConfigScore:
    search_type: str
    k: int
    rerank_k: int
    relevance_rate: float
    average_rerank_score: float
    normalized_rerank: float
    overall_score: float


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def select_best_retriever(
    chunks: List[ProcessedChunk],
    matrix: np.ndarray,
    eval_cfg: RetrievalConfig,
    embedder,
    reranker: Reranker,
    test_queries: List[Dict[str, str]],
    output_path: Path,
) -> tuple:
    """Grid sweep: search_type × k × rerank_k. Returns (best_name, best_cfg)
    and persists the chosen configuration to JSON."""
    retriever_results: Dict[str, ConfigScore] = {}

    for search_type in eval_cfg.search_types:
        for k in eval_cfg.k_values:
            # Per-config retriever with its own search type / rerank_k.
            cfg = RetrievalConfig(
                search_types=[search_type],
                k_values=[k],
                rerank_k=eval_cfg.rerank_k,
                mmr_fetch_k_cap=eval_cfg.mmr_fetch_k_cap,
                mmr_diversity=eval_cfg.mmr_diversity,
            )
            ret = Retriever(chunks, matrix, cfg, embedder, reranker)
            relevance_scores, top_scores = [], []
            for item in test_queries:
                results = ret.retrieve(item["question"], k, eval_cfg.rerank_k)
                if results:
                    top = results[0]
                    relevance_scores.append(
                        is_relevant(top.chunk, item.get("keywords", []))
                    )
                    top_scores.append(top.rerank_score or 0.0)

            relevance_rate = float(np.mean(relevance_scores)) if relevance_scores else 0.0
            avg_rerank = float(np.mean(top_scores)) if top_scores else 0.0
            normalized = sigmoid(avg_rerank)
            overall = (
                eval_cfg.relevance_weight * relevance_rate
                + eval_cfg.rerank_weight * normalized
            )
            name = f"{search_type}_k{k}"
            retriever_results[name] = ConfigScore(
                search_type=search_type, k=k, rerank_k=eval_cfg.rerank_k,
                relevance_rate=relevance_rate, average_rerank_score=avg_rerank,
                normalized_rerank=normalized, overall_score=overall,
            )
            print(
                f"      {name} ({search_type}): relevance={relevance_rate:.3f} "
                f"rerank={avg_rerank:.2f} overall={overall:.3f}"
            )

    best_name = max(retriever_results, key=lambda n: retriever_results[n].overall_score)
    best = retriever_results[best_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "search_type": best.search_type,
                "k": best.k,
                "rerank_k": best.rerank_k,
                "relevance_rate": best.relevance_rate,
                "average_rerank_score": best.average_rerank_score,
                "normalized_rerank": best.normalized_rerank,
                "overall_score": best.overall_score,
                "all_configurations": {
                    n: {
                        "search_type": s.search_type, "k": s.k,
                        "rerank_k": s.rerank_k, "relevance_rate": s.relevance_rate,
                        "average_rerank_score": s.average_rerank_score,
                        "normalized_rerank": s.normalized_rerank,
                        "overall_score": s.overall_score,
                    }
                    for n, s in retriever_results.items()
                },
            },
            fh, indent=2,
        )
    print(f"    Selected retriever: {best_name} (overall={best.overall_score:.3f})")
    return best_name, best