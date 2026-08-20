"""Prompting + Multi-Query phase.

Built on the logic of blocks 12-14 (grounding prompt, pre-generation
refusal, metadata-built citations) and the project's `Retriever`
interface — without modifying retrieval.py / pipeline.py.
"""

from __future__ import annotations

from src.qa.confidence import (
    REFUSAL_MESSAGE,
    SUPPORT_LEVELS,
    WEAKLY_SUPPORTED_CLAUSE,
    SupportDecision,
    SupportLevel,
    classify_support,
)
from src.qa.engine import (
    EXCERPT_LIMIT,
    Claim,
    QAEngine,
    QAResult,
    citation_from_metadata,
    format_answer,
)
from src.qa.multi_query import QUERY_EXPANSION_SYSTEM_PROMPT, expand_queries
from src.qa.prompts import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT_TEMPLATE,
    build_context_block,
)

__all__ = [
    "Claim",
    "NO_CONTEXT_ANSWER",
    "QAEngine",
    "QAResult",
    "QUERY_EXPANSION_SYSTEM_PROMPT",
    "REFUSAL_MESSAGE",
    "SUPPORT_LEVELS",
    "SYSTEM_PROMPT_TEMPLATE",
    "SupportDecision",
    "SupportLevel",
    "WEAKLY_SUPPORTED_CLAUSE",
    "build_context_block",
    "citation_from_metadata",
    "classify_support",
    "expand_queries",
    "format_answer",
    "EXCERPT_LIMIT",
]