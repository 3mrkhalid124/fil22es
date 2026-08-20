"""Multi-Query: expand the user's question into several search phrasings.

Every generated phrasing is logged for traceability, and the original
question is always included as the first query. Expansion only produces new
SEARCH strings — it never adds facts and never relaxes the context boundary.
"""

from __future__ import annotations

import re
import time
from typing import Callable, List, Optional

EXPANSION_MAX_TOKENS = 256

QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You are a medical information retrieval query-expansion assistant.\n"
    "Your only job is to rewrite a user's question into several distinct "
    "SEARCH QUERIES that improve retrieval over a cardiology document index.\n"
    "Rules:\n"
    "- Produce exactly {n} numbered search phrasings (one per line).\n"
    "- Cover different search angles: mechanism, definition, drug/class name,\n"
    "  physiology, treatment/management, and alternate wording.\n"
    "- If the question asks for numbers, statistics, percentages, rates,\n"
    "  prevalence or comparisons (e.g. % men vs women), add phrasings that\n"
    "  target TABLES and DATA, e.g. 'prevalence by sex percentage table',\n"
    "  'smoking percentage men women', 'statistics figures'.\n"
    "- Stay strictly within the facts and entities of the original question.\n"
    "  Do NOT add new medical facts, drugs, symptoms, or numbers that are not\n"
    "  already in the question.\n"
    "- Each phrasing is a standalone search string. No explanations, no\n"
    "  preamble, no bullet points, no surrounding quotes.\n"
    "- Use only numbered lines: 1. ...  2. ...  3. ..."
)


def _retry_seconds_from_error(exc: Exception) -> Optional[float]:
    """Extract Groq's 'try again in XmYs' hint from a rate-limit error."""
    message = str(exc)
    match = re.search(r"try again in\s+(\d+)m([\d.]+)s", message)
    if match:
        return float(match.group(1)) * 60 + float(match.group(2))
    match = re.search(r"try again in\s+([\d.]+)s", message)
    if match:
        return float(match.group(1))
    return None


def call_with_retry(
    fn: Callable,
    max_retries: int = 3,
    margin_seconds: float = 30.0,
):
    """Call `fn`, transparently waiting out Groq 429 rate-limit windows.

    Raises the last error after max_retries attempts (or immediately for
    non-rate-limit errors).
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — deliberate broad catch
            message = str(exc)
            is_rate_limit = "429" in message or "rate limit" in message.lower()
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait = _retry_seconds_from_error(exc)
            if wait is None:
                wait = 60.0
            wait += margin_seconds
            print(
                f"[groq] rate limit reached ({type(exc).__name__}); "
                f"waiting {wait / 60:.1f} min before retrying "
                "(Ctrl+C to abort)...",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError("call_with_retry exhausted its retries")


def _parse_variants(raw: str) -> List[str]:
    variants: List[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[.)\]]\s*(.+)$", line)
        if match:
            cleaned = match.group(1).strip().strip("\"'`*")
            if cleaned:
                variants.append(cleaned)
    return variants


def _deterministic_variants(question: str) -> List[str]:
    """Rule-based search phrasings used when the LLM expansion yields nothing.

    Recombines only the question's own content words (never new facts); for
    numeric/statistical questions it adds a table/statistics-targeted
    phrasing, which is what surfaces data-rich (table) chunks.
    """
    from src.retrieval import extract_query_keywords

    keywords = extract_query_keywords(question)
    if len(keywords) < 3:
        return []
    variants = [" ".join(keywords[:6])]
    if re.search(
        r"\d|percent|percentage|prevalence|how many|compared|rates?",
        question,
        re.IGNORECASE,
    ):
        variants.append("statistics percentage " + " ".join(keywords[:4]))
    return variants


def expand_queries(
    question: str,
    groq_client,
    model: str,
    temperature: float = 0.0,
    n_variants: int = 4,
) -> List[str]:
    """Return [original_question, variant_1, ..., variant_n].

    Expansion is an optimization, never a blocker: it makes a single fast
    attempt and, on ANY failure (rate limit, network, parse), falls back
    immediately to deterministic rule-based phrasings of the question — the
    retrieval + confidence gate + generation pipeline still runs normally.
    """
    queries: List[str] = [question]

    try:
        response = groq_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=EXPANSION_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": QUERY_EXPANSION_SYSTEM_PROMPT.format(n=n_variants),
                },
                {"role": "user", "content": f"Original question: {question}"},
            ],
        )
        raw = response.choices[0].message.content
    except Exception:
        return queries

    for variant in _parse_variants(raw):
        if variant.lower() == question.lower():
            continue
        if variant not in queries:
            queries.append(variant)

    if len(queries) <= 1:
        for variant in _deterministic_variants(question):
            if variant not in queries:
                queries.append(variant)

    return queries[: n_variants + 1]