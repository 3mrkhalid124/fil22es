"""Prompting + Multi-Query engine.

Builds on the logic of blocks 12-14 (grounding prompt, pre-generation
refusal, metadata-built citations, clean-refusal safeguard) and the
project's `Retriever` interface — without modifying retrieval.py or
pipeline.py.

Pipeline (per question):
    1. multi-query expansion (LLM, Groq) -> [original, variants...]
    2. retrieve each sub-query through the project's Retriever
    3. merge: max rerank score per chunk across sub-queries
    4. confidence gate: best merged score < 0.77 -> refuse before generation
    5. grounded generation (per-claim contract, temperature 0)
    6. block_12 safeguard: escape-hatch text anywhere -> clean refusal
    7. parse claims + map passage references back to chunks
    8. per-claim scoring: min over the claim's supporting chunk scores
    9. drop claims below 0.77; metadata-built citations + evidence excerpts
   10. overall uncertainty = min over retained claims
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.qa.confidence import (
    REFUSE_THRESHOLD,
    REFUSAL_MESSAGE,
    WEAKLY_SUPPORTED_CLAUSE,
    SupportLevel,
    classify_support,
)
from src.qa.logger import TraceLogger
from src.qa.multi_query import call_with_retry, expand_queries
from src.qa.prompts import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT_TEMPLATE,
    build_context_block,
)
from src.retrieval import extract_query_keywords

GROQ_MODEL = "openai/gpt-oss-120b"
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 1024
DEFAULT_N_VARIANTS = 4
DEFAULT_TOP_CONTEXT = 4
# Retrieval pool size. k=5 (project default) misses exact table chunks that
# the reranker scores highly once they enter the candidate pool; k=20 recovers
# them (validated: smoking table 0.95, cholesterol row 0.98) without lifting
# any refusal case above the 0.77 gate.
DEFAULT_K = 20
DEFAULT_RERANK_K = 10
EXCERPT_LIMIT = 300

_PASSAGE_RE = re.compile(r"Passage\s+(\d+)", re.IGNORECASE)
_CLAIM_LINE_RE = re.compile(r"^C\d+\s*\|.*$", re.IGNORECASE)


# ==========================================================================
# Results
# ==========================================================================


@dataclass
class Claim:
    claim_id: str
    text: str
    passage_numbers: List[int]
    support_score: float
    support_level: SupportLevel
    citations: List[str]
    evidence: List[str]
    dropped: bool = False
    note: str = ""


@dataclass
class QAResult:
    question: str = ""
    sub_queries: List[str] = field(default_factory=list)
    gate_decision: str = ""
    gate_best_score: Optional[float] = None
    refusal_message: Optional[str] = None
    claims: List[Claim] = field(default_factory=list)
    overall_level: Optional[SupportLevel] = None
    overall_score: Optional[float] = None
    raw_answer: Optional[str] = None
    validation: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_refused(self) -> bool:
        return self.gate_decision == "refused"


# ==========================================================================
# Deterministic helpers (block_12 logic, reused verbatim)
# ==========================================================================


def citation_from_metadata(chunk) -> str:
    """Citation built from trusted metadata, never from the LLM's text."""
    document = getattr(chunk.metadata, "source_file", "Unknown Document")
    pages = getattr(chunk.metadata, "page_numbers", [])
    page_str = ", ".join(str(p) for p in pages) if pages else "Unknown Page"
    section = getattr(chunk.metadata, "section_title", None)
    if section and section != "General Context":
        return f"[{document}, {section}, Page {page_str}]"
    return f"[{document}, Page {page_str}]"


def _excerpt(text: str, limit: int = EXCERPT_LIMIT) -> str:
    body = re.sub(r"\s+", " ", text).strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + " ..."


def _locate_evidence(
    chunk_text: str, claim_text: str, width: int = 240
) -> str:
    """Return the window of the chunk that best supports the claim.

    Scans the chunk for the segment with the most claim keywords, so the
    Evidence field shows the supporting sentence region instead of the
    chunk's beginning. Falls back to the chunk head on ties/no keywords.
    """
    body = re.sub(r"\s+", " ", chunk_text).strip()
    if len(body) <= width:
        return body
    keywords = set(extract_query_keywords(claim_text))
    if not keywords:
        return body[:width].rstrip() + " ..."
    lower = body.lower()
    step = max(1, width // 4)
    best_start, best_score = 0, -1
    for start in range(0, max(1, len(lower) - width + 1), step):
        window = lower[start : start + width]
        score = sum(1 for kw in keywords if kw in window)
        if score > best_score:
            best_score, best_start = score, start
    if best_score <= 0:
        return body[:width].rstrip() + " ..."
    start = max(0, best_start)
    return body[start : start + width].rstrip() + " ..."


def build_keyword_df(chunks) -> Dict[str, int]:
    """Document frequency per content keyword over the whole corpus.

    Used by the grounding check: rare keywords (low document frequency)
    are the discriminative signal that a claim's text actually comes from
    its cited chunks rather than from the model's training knowledge.
    """
    df: Dict[str, int] = {}
    for chunk in chunks:
        body = re.sub(r"\s+", " ", chunk.original_text).lower()
        seen = set(extract_query_keywords(body))
        for keyword in seen:
            df[keyword] = df.get(keyword, 0) + 1
    return df


def _claim_grounded(
    claim_text: str,
    chunk_texts,
    keyword_df: Dict[str, int],
    n_chunks: int,
) -> bool:
    """Deterministic grounding check.

    A claim is kept only if its distinctive (corpus-rare) content words
    actually appear in the cited chunks. If the claim has no rare words,
    a lenient generic-overlap rule is used instead. This does NOT rely on
    the model's own judgment.
    """
    keywords = sorted(set(extract_query_keywords(claim_text)))

    bodies = [re.sub(r"\s+", " ", b).lower() for b in chunk_texts]

    def hit(keyword: str) -> bool:
        return any(keyword in body for body in bodies)

    # Numeric grounding: any quantity in the claim (dose, value, number)
    # must literally appear in the cited evidence. Fabricated doses are the
    # classic medical hallucination, and keyword overlap cannot catch them.
    for number in re.findall(r"\d+(?:\.\d+)?", claim_text):
        if not any(number in body for body in bodies):
            return False

    if not keywords:
        return True  # nothing to verify against -> keep (lenient)

    rare_threshold = max(1, int(round(0.05 * n_chunks)))
    rare = [kw for kw in keywords if keyword_df.get(kw, n_chunks) <= rare_threshold]
    common = [kw for kw in keywords if kw not in rare]

    if rare:
        # The distinctive entity words of the claim must be present in the
        # cited evidence. At least ONE rare word must hit: model phrasing
        # legitimately echoes question words (e.g. "systematic", "reported")
        # that are absent from a table chunk, so requiring a majority of the
        # rare words rejects perfectly grounded numeric answers (validated:
        # smoking-table answer "69.1% men vs 30.1% women"). A hallucinated
        # claim still fails because none of its rare entity words appear in
        # the cited chunk, and fabricated numbers are caught by the numeric
        # check above.
        rare_hits = sum(1 for kw in rare if hit(kw))
        return rare_hits >= 1

    common_hits = sum(1 for kw in common if hit(kw))
    need = max(2, int(round(0.4 * len(common))))
    return common_hits >= min(need, len(common))


# ==========================================================================
# Rendering
# ==========================================================================


def format_answer(result: QAResult) -> str:
    """Render the mandatory 3-part output + honest Uncertainty Score."""
    if result.is_refused:
        return f"QUESTION: {result.question}\n\n{result.refusal_message}"

    lines: List[str] = [f"QUESTION: {result.question}", ""]

    lines.append("Recommendation:")
    for claim in result.claims:
        weak = (
            f"  [{WEAKLY_SUPPORTED_CLAUSE}]"
            if claim.support_level is SupportLevel.WEAKLY_SUPPORTED
            else ""
        )
        lines.append(f"  {claim.claim_id}: {claim.text}{weak}")

    lines.append("")
    lines.append("Evidence:")
    for claim in result.claims:
        lines.append(f"  {claim.claim_id}:")
        for evidence in claim.evidence:
            lines.append(f"    - \"{evidence}\"")

    lines.append("")
    lines.append("Citation:")
    for claim in result.claims:
        for citation in claim.citations:
            lines.append(f"  {claim.claim_id}: {citation}")

    lines.append("")
    lines.append("Uncertainty Score:")
    for claim in result.claims:
        lines.append(
            f"  {claim.claim_id}: {claim.support_score:.3f} "
            f"({claim.support_level.value})"
        )
    lines.append(
        f"  Overall: {result.overall_score:.3f} "
        f"({result.overall_level.value})"
    )
    if result.overall_level is SupportLevel.WEAKLY_SUPPORTED:
        lines.append(f"  {WEAKLY_SUPPORTED_CLAUSE}")

    dropped = result.validation.get("n_dropped_below_threshold", 0)
    ungrounded = result.validation.get("n_dropped_ungrounded", 0)
    if dropped or ungrounded:
        lines.append("")
        if dropped:
            lines.append(
                f"  Note: {dropped} claim(s) were removed because their supporting "
                "evidence was below the confidence threshold (0.77)."
            )
        if ungrounded:
            lines.append(
                f"  Note: {ungrounded} claim(s) were removed because their text "
                "was not grounded in the cited passages."
            )

    lines.append("")
    lines.append(
        "This information is for educational purposes only and is not a "
        "medical diagnosis or a treatment decision. Always consult a "
        "qualified clinician for any medical decision."
    )
    return "\n".join(lines)


# ==========================================================================
# Engine
# ==========================================================================


class QAEngine:
    def __init__(
        self,
        retriever,
        groq_client,
        model: str = GROQ_MODEL,
        temperature: float = GENERATION_TEMPERATURE,
        n_variants: int = DEFAULT_N_VARIANTS,
        top_context: int = DEFAULT_TOP_CONTEXT,
        k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        logger: Optional[TraceLogger] = None,
    ) -> None:
        self.retriever = retriever
        self.groq_client = groq_client
        self.model = model
        self.temperature = temperature
        self.n_variants = n_variants
        self.top_context = top_context
        self.k = k if k is not None else DEFAULT_K
        self.rerank_k = rerank_k if rerank_k is not None else DEFAULT_RERANK_K
        self.logger = logger if logger is not None else TraceLogger()
        self._keyword_df: Optional[Dict[str, int]] = None

    def _grounding_df(self) -> Dict[str, int]:
        if self._keyword_df is None:
            self._keyword_df = build_keyword_df(self.retriever.chunks)
        return self._keyword_df

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def ask(self, question: str, show_details: bool = False) -> QAResult:
        self.logger.start(question)
        result = QAResult(question=question)

        # 1. Multi-query expansion
        sub_queries = expand_queries(
            question,
            self.groq_client,
            self.model,
            temperature=self.temperature,
            n_variants=self.n_variants,
        )
        result.sub_queries = sub_queries
        self.logger.set("sub_queries", sub_queries)

        # 2. Retrieve per sub-query
        per_query = []
        for query in sub_queries:
            retrieved = self.retriever.retrieve(query, self.k, self.rerank_k)
            per_query.append((query, retrieved))
            self.logger.add_query(query, retrieved)

        # 3. Merge: max rerank per chunk across sub-queries
        merged = self._merge_results(per_query)
        self.logger.set(
            "merged_chunks",
            [
                {
                    "chunk_id": m["chunk"].chunk_id,
                    "source": m["chunk"].metadata.source_file,
                    "pages": m["chunk"].metadata.page_numbers,
                    "max_rerank": round(m["max_rerank"], 4),
                    "queries_hit": len(m["queries"]),
                }
                for m in merged.values()
            ],
        )

        # 4. Confidence gate BEFORE generation (block_13 logic)
        best_score = max((m["max_rerank"] for m in merged.values()), default=None)
        result.gate_best_score = best_score
        if best_score is None or classify_support(best_score) is SupportLevel.UNSUPPORTED:
            result.gate_decision = "refused"
            result.refusal_message = REFUSAL_MESSAGE
            result.validation = {"gate": "refused before generation"}
            self.logger.set(
                "gate", {"decision": "refused", "best_score": best_score,
                         "reason": "best retrieval score below 0.77"}
            )
            if show_details:
                self._print_result(result)
            return result

        result.gate_decision = "generated"
        self.logger.set(
            "gate", {"decision": "generated", "best_score": best_score}
        )

        # 5. Top context chunks (only chunks that can actually support a
        #    retained claim). A chunk below the 0.77 gate can never survive
        #    the per-claim min rule, so offering it in context just invites
        #    the model to cite it and self-inflict a refusal.
        ordered = sorted(
            merged.values(), key=lambda m: m["max_rerank"], reverse=True
        )
        context_eligible = [
            m for m in ordered if m["max_rerank"] >= REFUSE_THRESHOLD
        ]
        if not context_eligible:
            context_eligible = ordered[:1]
        top_chunks = [m["chunk"] for m in context_eligible[: self.top_context]]
        self.logger.set(
            "top_context",
            [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.metadata.source_file,
                    "pages": c.metadata.page_numbers,
                }
                for c in top_chunks
            ],
        )

        # 6. Grounded generation
        raw_answer = self._generate(question, top_chunks)
        result.raw_answer = raw_answer
        self.logger.set("raw_answer", raw_answer)

        # 7. block_12 hard safeguard: refusal text anywhere -> clean refusal
        if NO_CONTEXT_ANSWER in raw_answer:
            result.gate_decision = "refused"
            result.refusal_message = NO_CONTEXT_ANSWER
            result.validation = {"refusal": "escape hatch triggered in raw answer"}
            self.logger.set(
                "gate", {"decision": "refused", "reason": "escape hatch"}
            )
            if show_details:
                self._print_result(result)
            return result

        # 8. Parse claims + per-claim scoring
        claims = self._parse_claims(raw_answer, top_chunks, merged)
        result.claims = [c for c in claims if not c.dropped]

        # 9. Validation
        result.validation = self._validate(raw_answer, claims)

        # 10. Overall = min over retained claims; refuse if nothing remains
        retained = [c for c in claims if not c.dropped]
        if not retained:
            result.gate_decision = "refused"
            result.refusal_message = REFUSAL_MESSAGE
            result.validation["fallback"] = (
                "no claim met the 0.77 support threshold"
            )
            self.logger.set(
                "gate",
                {"decision": "refused", "reason": "no retained claims"},
            )
            if show_details:
                self._print_result(result)
            return result

        overall_score = min(c.support_score for c in retained)
        result.overall_score = overall_score
        result.overall_level = classify_support(overall_score)
        self.logger.set(
            "overall",
            {"score": overall_score, "level": result.overall_level.value},
        )

        if show_details:
            self._print_result(result)
        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_results(per_query):
        """Max rerank (and max dense) per chunk across all sub-queries."""
        merged: Dict[str, Dict[str, Any]] = {}
        for query, results in per_query:
            for r in results:
                chunk_id = r.chunk.chunk_id
                entry = merged.setdefault(
                    chunk_id,
                    {
                        "chunk": r.chunk,
                        "max_rerank": -float("inf"),
                        "max_dense": -float("inf"),
                        "queries": [],
                        "scores": [],
                    },
                )
                entry["max_rerank"] = max(
                    entry["max_rerank"], r.rerank_score or 0.0
                )
                entry["max_dense"] = max(entry["max_dense"], r.dense_score)
                entry["queries"].append(query)
                entry["scores"].append(round(r.rerank_score or 0.0, 4))
        return merged

    def _generate(self, question: str, top_chunks) -> str:
        context_block = build_context_block(top_chunks)
        user_message = f"{context_block}\n\nQuestion: {question}"

        def _call():
            return self.groq_client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=GENERATION_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                    {"role": "user", "content": user_message},
                ],
            )

        response = call_with_retry(_call)
        return response.choices[0].message.content

    def _parse_claims(
        self, raw_answer: str, top_chunks, merged
    ) -> List[Claim]:
        """Parse C<n>| text |Passage n|Passage m| lines and score each claim.

        Per-claim score = MIN over the max-rerank scores of the chunks the
        claim references (weakest supporting evidence decides). A claim is
        also dropped when its text is not grounded in the cited chunks.
        """
        keyword_df = self._grounding_df()
        n_chunks = len(self.retriever.chunks)
        passage_to_chunk_id = {
            i: chunk.chunk_id for i, chunk in enumerate(top_chunks, start=1)
        }
        claims: List[Claim] = []

        for line in raw_answer.splitlines():
            line = line.strip()
            if not _CLAIM_LINE_RE.match(line):
                continue

            tokens = [t.strip() for t in line.split("|") if t.strip() != ""]
            if len(tokens) < 2:
                continue

            claim_id = tokens[0]
            passages: List[int] = []
            text_parts: List[str] = []
            for token in tokens[1:]:
                match = _PASSAGE_RE.fullmatch(token)
                if match:
                    passages.append(int(match.group(1)))
                else:
                    text_parts.append(token)
            text = " ".join(text_parts).strip()
            if not text or not passages:
                continue

            support_chunks = []
            valid_passages = []
            for p in sorted(set(passages)):
                chunk_id = passage_to_chunk_id.get(p)
                if chunk_id is None:
                    continue
                support_chunks.append(merged[chunk_id])
                valid_passages.append(p)

            if not support_chunks:
                claims.append(
                    Claim(
                        claim_id=claim_id,
                        text=text,
                        passage_numbers=list(passages),
                        support_score=0.0,
                        support_level=SupportLevel.UNSUPPORTED,
                        citations=[],
                        evidence=[],
                        dropped=True,
                        note="passage references outside provided context",
                    )
                )
                continue

            score = min(m["max_rerank"] for m in support_chunks)
            level = classify_support(score)
            citations, evidence = QAEngine._cite(valid_passages, top_chunks, text)

            grounded = _claim_grounded(
                text,
                [m["chunk"].original_text for m in support_chunks],
                keyword_df,
                n_chunks,
            )
            dropped = (level is SupportLevel.UNSUPPORTED) or not grounded
            if dropped:
                note = (
                    "below 0.77 support threshold"
                    if level is SupportLevel.UNSUPPORTED and grounded
                    else "claim text not grounded in the cited chunks"
                )
            else:
                note = ""
            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=text,
                    passage_numbers=valid_passages,
                    support_score=score,
                    support_level=level,
                    citations=citations,
                    evidence=evidence,
                    dropped=dropped,
                    note=note,
                )
            )

        return claims

    @staticmethod
    def _cite(passage_numbers, top_chunks, claim_text=""):
        citations: List[str] = []
        evidence: List[str] = []
        for p in passage_numbers:
            if 1 <= p <= len(top_chunks):
                chunk = top_chunks[p - 1]
                citations.append(citation_from_metadata(chunk))
                evidence.append(_locate_evidence(chunk.original_text, claim_text))
        return citations, evidence

    @staticmethod
    def _validate(raw_answer: str, claims: List[Claim]) -> Dict[str, Any]:
        parsed = len(claims) > 0
        return {
            "has_claims": parsed,
            "n_claims": len(claims),
            "n_retained": sum(1 for c in claims if not c.dropped),
            "n_dropped_below_threshold": sum(
                1 for c in claims if c.dropped and "0.77" in c.note
            ),
            "n_dropped_ungrounded": sum(
                1 for c in claims if c.dropped and "not grounded" in c.note
            ),
            "refusal_text_present": NO_CONTEXT_ANSWER in raw_answer,
            "all_claims_grounded": parsed and all(
                c.passage_numbers for c in claims
            ),
        }

    def _print_result(self, result: QAResult) -> None:
        print("\n" + "=" * 80)
        print("QUESTION:", result.question)
        print("SUB-QUERIES:", " | ".join(result.sub_queries))
        print("=" * 80)
        if result.is_refused:
            print("[REFUSED]", result.refusal_message)
        else:
            print(format_answer(result))
        print("=" * 80)