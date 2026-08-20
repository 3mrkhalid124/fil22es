"""Page-aware cleaning: artifacts, headers/footers, front matter, sections.

This module consolidates the notebook's entire Block 8A cleaning stage
(hyphen repair, page-artifact detection, journal headers, repeated-line
header/footer detection, TOC detection, section hierarchy, front-matter
classification, clinical-start detection) into composable, testable units.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import PreprocessingConfig

# ---------------------------------------------------------------------------
# Line-level artifact detection
# ---------------------------------------------------------------------------

_PAGE_ARTIFACT_RE = re.compile(r"^\s*page\s*\d{1,4}\s*$", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_EPAGE_RE = re.compile(r"^\s*e\d{3,5}\s*$")
_JOURNAL_HEADER_RE = re.compile(
    r"(J\s*A\s*C\s*C|JAMA|Circulation|Heart|vol|issue|DOI|ISSN)", re.IGNORECASE
)
_TOC_TITLE_RE = re.compile(
    r"^\s*(table\s+of\s+contents|contents)\s*:?\s*$", re.IGNORECASE
)
_TOC_ENTRY_RE = re.compile(r"^\s*.{3,60}(\.{2,}|\s{2,})(\d+|e?\d+)\s*$")

_GARBAGE_PATTERNS = [
    # Copyright / legal boilerplate lines
    re.compile(r"(unauthorized\s+use\s+(is\s+)?prohibited|all\s+rights\s+reserved)", re.IGNORECASE),
    re.compile(r"(no\s+part\s+of\s+(this\s+(publication|document))|(may\s+not\s+be\s+reproduced|stored\s+in\s+a\s+retrieval))", re.IGNORECASE),
    # Internal document codes / tracking numbers (e.g. "WF618229", "JACC-12345-2026")
    re.compile(r"^[A-Z]{2,6}\d{4,8}$"),
    re.compile(r"^\s*[©]\s*$|^(www\.[\w.-]+\.\w{2,}|doi\s*:?\s*\S+)\s*$"),
]
_GARBAGE_BOOST_RE = re.compile(r"(copyright|©|prohibited|reproduction|WF\d{4,}|JACC-?\d)", re.IGNORECASE)


def is_garbage_line(line: str) -> bool:
    """True for pure junk lines: copyright boilerplate, document codes, etc."""
    text = line.strip()
    if not text:
        return True
    if any(p.match(text) or p.search(text) for p in _GARBAGE_PATTERNS):
        return True
    # If a short line is mostly made of junk tokens, drop it too.
    if len(text) < 150 and len(_GARBAGE_BOOST_RE.findall(text)) >= 2:
        return True
    return False

def is_page_artifact(line: str) -> bool:
    text = line.strip()
    if not text:
        return True
    return bool(_PAGE_ARTIFACT_RE.match(text)) or bool(_PAGE_NUMBER_RE.match(text))


def is_journal_header(line: str) -> bool:
    return bool(_JOURNAL_HEADER_RE.search(line)) and len(line.strip()) < 100


def is_toc_title(line: str) -> bool:
    return bool(_TOC_TITLE_RE.match(line))


def is_toc_entry(line: str) -> bool:
    return bool(_TOC_ENTRY_RE.match(line))


# ---------------------------------------------------------------------------
# Repeated header/footer detection (per-file)
# ---------------------------------------------------------------------------


def find_repeated_lines(
    pages: List[str], min_ratio: float = 0.12, threshold: int = 3
) -> set:
    """Lines that appear on >= threshold pages (a fraction of the book)
    are almost certainly running headers/footers."""
    counts: Counter[str] = Counter()
    for page in pages:
        seen = set()
        for raw_line in page.splitlines():
            line = raw_line.strip()
            if len(line) < 4:
                continue
            if line not in seen:
                counts[line] += 1
                seen.add(line)
    n_pages = len(pages) or 1
    minimum = max(threshold, int(n_pages * min_ratio))
    return {line for line, count in counts.items() if count >= minimum}

def find_recurring_heading_texts(
    prepared_pages: List["PreparedPage"],
    min_occurrences: int = 3,
) -> set:
    """Detect heading-like lines that recur many times across one document.

    A genuine section heading appears once (or occasionally repeats across
    a handful of nested subsections). A line that matches a heading pattern
    but recurs many times across many pages — e.g. "You should understand:"
    used as a recurring in-text prompt in a patient booklet — is almost
    certainly NOT a structural boundary, and must not be allowed to swallow
    unrelated content between its occurrences.

    Must be called BEFORE the real per-line pass, using detect_section
    with no context args (context-based bullet filtering still applies).
    """
    counts: Counter[str] = Counter()
    for page in prepared_pages:
        for line in page.lines:
            heading = detect_section(line)
            if heading:
                counts[heading.strip().lower()] += 1
    return {text for text, count in counts.items() if count >= min_occurrences}


def strip_repeated_lines(page: str, repeated: set) -> str:
    lines = [ln for ln in page.splitlines() if ln.strip() not in repeated]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section detection + hierarchy
# ---------------------------------------------------------------------------

KNOWN_HEADINGS = {
    "abstract", "introduction", "anatomy", "physiology", "pathology",
    "pharmacology", "embryology", "discussion", "methods", "results",
    "conclusion", "summary", "epidemiology", "treatment", "diagnosis",
    "management", "prevention", "complications", "risk factors",
}

END_SECTION_HEADINGS = {
    "references", "appendix", "acknowledgments", "acknowledgement",
    "conflict of interest", "conflicts of interest", "funding",
    "bibliography", "glossary", "index",
}

_HEADING_PATTERNS = [
    re.compile(r"^\s*chapter\s+\d+.*$", re.IGNORECASE),
    re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][^\.\n]{0,55}$"),
    re.compile(r"^[A-Z][A-Z0-9\s:,()&/-]{4,100}$"),
    re.compile(r"^\s*(?:▪|❖|•)\s*[A-Z][^\.\n]{0,60}$"),
    re.compile(r"^[A-Z][^\.\n]{3,60}:$"),
]

_PROSE_WORDS = {
    "is a", "are a", "is the", "are the", "was a", "were a",
    "the patient", "the left", "the right", "of the", "in the",
}


def detect_section(
    line: str,
    prev_line: str = "",
    next_line: str = "",
    recurring_headings: Optional[set] = None,
) -> Optional[str]:
    text = line.strip()
    if not text or len(text) > 70:
        return None
    if text.lower() in END_SECTION_HEADINGS:
        return None
    if re.match(r"^\s*\d+[\.]?\s+[A-Z]", text) and any(w in text.lower() for w in _PROSE_WORDS):
        return None

    # Reject bullet lines that are part of a list, not a heading.
    bullet_match = re.match(r"^\s*(▪|❖|•)\s*[A-Z][^\.\n]{0,60}$", text)
    if bullet_match:
        marker = bullet_match.group(1)
        prev_stripped = prev_line.strip()
        next_stripped = next_line.strip()
        if prev_stripped.startswith(marker) or next_stripped.startswith(marker):
            return None

    for pattern in _HEADING_PATTERNS:
        if pattern.match(text):
            # NEW: reject headings that recur too often across the file —
            # a real section boundary shouldn't repeat verbatim many times.
            if recurring_headings and text.strip().lower() in recurring_headings:
                return None
            return text

    if text.lower() in KNOWN_HEADINGS:
        candidate = text.capitalize()
        if recurring_headings and candidate.strip().lower() in recurring_headings:
            return None
        return candidate

    return None

def is_strong_end_heading(line: str) -> bool:
    return line.strip().lower() in END_SECTION_HEADINGS


def update_hierarchy(hierarchy: Dict[str, Dict], heading: str) -> None:
    """Maintain a parent -> children tree and set current path."""
    depth = heading.count(".") + (1 if re.match(r"^\d", heading) else 0)
    hierarchy.setdefault("_current", {})["heading"] = heading
    hierarchy["_current"]["depth"] = depth


def get_context(hierarchy: Dict[str, Dict]) -> str:
    """Return a readable breadcrumb like 'Chapter 1 → Physiology'."""
    current = hierarchy.get("_current", {})
    return current.get("heading", "General Context")


# ---------------------------------------------------------------------------
# Front-matter classification
# ---------------------------------------------------------------------------

_AUTHOR_PATTERNS = [
    re.compile(r"(writing committee|study group|investigators)", re.IGNORECASE),
    re.compile(r"\b(MD|PhD|MSc|FACC|FAHA|FRCP)\b"),
    re.compile(r"[\w.-]+@[\w.-]+\.\w{2,}"),
    re.compile(r"(Department of|University|Hospital|School of Medicine)"),
]
_COPYRIGHT_RE = re.compile(r"(copyright|©|\ball rights reserved)", re.IGNORECASE)
_JOURNAL_METADATA_RE = re.compile(r"(doi\s*:?\s*\S+|issn|pii\s*:?\s*\S+)", re.IGNORECASE)


def classify_front_page(page: str) -> str:
    """Classify a page as front matter type or 'clinical' / 'unknown'.

    Returns one of: blank, toc, authors_membership, copyright,
    journal_metadata, title, clinical, front_matter, unknown.
    """
    text = page.strip()
    if not text:
        return "blank"
    if is_toc_title(text.splitlines()[0]) or any(
        is_toc_entry(ln) for ln in text.splitlines()[:10]
    ):
        return "toc"
    author_hits = sum(bool(p.search(text)) for p in _AUTHOR_PATTERNS)
    if author_hits >= 2:
        return "authors_membership"
    if _COPYRIGHT_RE.search(text):
        return "copyright"
    if _JOURNAL_METADATA_RE.search(text):
        return "journal_metadata"
    if _EPAGE_RE.match(text.splitlines()[0]) and len(text) < 400:
        return "title"
    if author_hits >= 1:
        return "front_matter"
    return "clinical"


CLINICAL_TERMS = {
    "patient", "patients", "diagnosis", "treatment", "management",
    "recommendation", "clinical", "therapy", "disease", "heart",
    "blood pressure", "trial", "guideline",
}

STRONG_CLINICAL_HEADINGS = {"methods", "results", "conclusion", "abstract"}


def find_clinical_start_page(
    pages: List[str], max_scan_ratio: float = 0.35
) -> int:
    """First page index that is clearly clinical content (or 0)."""
    max_scan = max(1, int(len(pages) * max_scan_ratio))
    for idx, page in enumerate(pages[:max_scan]):
        lower = page.lower()
        heading = (page.splitlines() or [""])[0].strip().lower()
        score = sum(term in lower for term in CLINICAL_TERMS)
        _CHAPTER_KW = ("anatomy", "embryology", "physiology", "pathology",
                       "pharmacology", "chapter")
        if (heading in STRONG_CLINICAL_HEADINGS
                or score >= 2
                or any(kw in heading for kw in _CHAPTER_KW)):
            return idx

    return 0


def detect_end_boundary(pages: List[str]) -> int:
    """Page index of the first end section (References etc.), or len(pages)."""
    for idx, page in enumerate(pages):
        for line in page.splitlines():
            if is_strong_end_heading(line):
                return idx
    return len(pages)


# ---------------------------------------------------------------------------
# Page preparation (the main entry point)
# ---------------------------------------------------------------------------


@dataclass
class PreparedPage:
    page_number: int
    lines: List[str]


def prepare_pdf_pages(
    pages: List[str],
    cfg: Optional[PreprocessingConfig] = None,
) -> Tuple[List[PreparedPage], List[Tuple[int, str]]]:
    """Strip artifacts, headers/footers, and front matter from raw pages.

    Returns (prepared_pages, removed_pages_report).
    """
    cfg = cfg or PreprocessingConfig()
    removed: List[Tuple[int, str]] = []
    prepared: List[PreparedPage] = []

    # 1. Find repeated header/footer lines globally.
    repeated = find_repeated_lines(pages, cfg.header_footer_repeat_ratio)

    # 2. Locate clinical start / end boundary (skip TOC, copyright, etc.).
    start_idx = 0
    if cfg.remove_front_matter:
        start_idx = find_clinical_start_page(pages, cfg.max_front_scan_ratio)
    end_idx = detect_end_boundary(pages)

    for offset, page in enumerate(pages):
        page_number = offset + 1
        lines = []

        # Front-matter and end-section pages are dropped wholesale.
        if offset < start_idx or offset >= end_idx:
            removed.append((page_number, "front_or_end_matter"))
            continue

        page = strip_repeated_lines(page, repeated)

        pending: str = ""
        for raw_line in page.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if is_page_artifact(line):
                continue
            if _EPAGE_RE.match(line):
                continue
            if is_journal_header(line):
                continue
            if is_garbage_line(line):
                continue
            # Box-boundary repair: the previous line ended mid-sentence
            # (no terminal punctuation) and this line starts lowercase —
            # the PDF split the sentence across text boxes, so join them.
            if (
                    pending
                    and line[0].islower()
                    and pending[-1] not in ".!?:"
                    and len(pending) < 120
            ):
                pending = pending + " " + line
                continue

            if pending:
                lines.append(pending)
            pending = line

        if pending:
            lines.append(pending)

        # Cross-page sentence repair: sentence split across a page break.
        if (
                prepared
                and lines
                and lines[0][0].islower()
                and prepared[-1].lines
                and prepared[-1].lines[-1][-1] not in ".!?:"
        ):
            prepared[-1].lines[-1] = (
                    prepared[-1].lines[-1] + " " + lines[0]
            )
            lines = lines[1:]

        if not lines:
            removed.append((page_number, "empty_after_cleanup"))
            continue

        prepared.append(PreparedPage(page_number=page_number, lines=lines))

    print(
        f"    Retained pages: {len(prepared)} / {len(pages)} "
        f"({len(removed)} removed)"
    )
    return prepared, removed
