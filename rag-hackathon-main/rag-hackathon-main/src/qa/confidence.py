"""Confidence thresholds for the Prompting phase.

The five support levels and their cutoff values are the project's required
LITERAL specification (0.77 / 0.83 / 0.89 / 0.96). They are applied to the
real cross-encoder rerank score (0..1) returned by the project's Reranker —
never approximated, never changed, never averaged.

Aggregation rule (as required):
- per chunk: max rerank score across all sub-queries that retrieved it;
- per claim: min over the supporting chunks of the claim;
- overall answer: min over all retained claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class SupportLevel(Enum):
    UNSUPPORTED = "Unsupported"
    WEAKLY_SUPPORTED = "Weakly Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    WELL_SUPPORTED = "Well Supported"
    STRONGLY_SUPPORTED = "Strongly Supported"


# Literal threshold values — required as-is.
REFUSE_THRESHOLD = 0.77
WEAK_THRESHOLD = 0.83
PARTIAL_THRESHOLD = 0.89
WELL_THRESHOLD = 0.96

# Mandatory explicit sentence appended to weakly-supported claims.
WEAKLY_SUPPORTED_CLAUSE = "Better to check with a doctor"

# Refusal message (kept from block_13, reuse verbatim).
REFUSAL_MESSAGE = (
    "I don't have enough relevant information in the indexed documents "
    "to answer this confidently. Try rephrasing the question, or "
    "consult a clinician directly."
)

# Ordered from weakest to strongest (used for comparisons / display).
SUPPORT_LEVELS: Tuple[SupportLevel, ...] = (
    SupportLevel.UNSUPPORTED,
    SupportLevel.WEAKLY_SUPPORTED,
    SupportLevel.PARTIALLY_SUPPORTED,
    SupportLevel.WELL_SUPPORTED,
    SupportLevel.STRONGLY_SUPPORTED,
)


def classify_support(score: float) -> SupportLevel:
    """Map a real rerank score (0..1) to a support level.

    Semantics (literal, per spec):
        score < 0.77                -> Unsupported (refuse entirely)
        0.77 <= score < 0.83        -> Weakly Supported
        0.83 <= score < 0.89        -> Partially Supported
        0.89 <= score < 0.96        -> Well Supported
        score >= 0.96               -> Strongly Supported
    """
    if score < REFUSE_THRESHOLD:
        return SupportLevel.UNSUPPORTED
    if score < WEAK_THRESHOLD:
        return SupportLevel.WEAKLY_SUPPORTED
    if score < PARTIAL_THRESHOLD:
        return SupportLevel.PARTIALLY_SUPPORTED
    if score < WELL_THRESHOLD:
        return SupportLevel.WELL_SUPPORTED
    return SupportLevel.STRONGLY_SUPPORTED


@dataclass
class SupportDecision:
    score: float
    level: SupportLevel

    @property
    def is_refused(self) -> bool:
        return self.level is SupportLevel.UNSUPPORTED