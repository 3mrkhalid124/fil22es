"""Chunking pipeline: structural units -> semantic chunks -> size control.

Implements the notebook's Block 8A logic as pure functions:

    prepared_pages -> build_structural_units -> merge_structural_units
    -> (per unit) SemanticChunker -> size control grid -> quality scoring
    -> best configuration
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
try:
    from langchain.docstore.document import Document  # legacy
except ModuleNotFoundError:
    from langchain_core.documents import Document  # langchain>=0.3
from langchain_experimental.text_splitter import SemanticChunker

from .nlp_utils import get_nlp as _get_nlp

from .cleaning import (
    PreparedPage,
    detect_section,
    find_recurring_heading_texts,   # <-- NEW import
    get_context,
    is_strong_end_heading,
    is_toc_entry,
    is_toc_title,
    update_hierarchy,
)

from .config import ChunkingConfig, PreprocessingConfig








def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    doc = _get_nlp()(text)
    sents = [sent.text.strip() for sent in doc.sents]
    return [s for s in sents if s]


def repair_mid_sentence_starts(docs: List[Document]) -> List[Document]:
    result: List[Document] = []
    for doc in docs:
        text = doc.page_content.strip()
        if (
            result
            and text
            and text[0].islower()
            and result[-1].metadata.get("source_file") == doc.metadata.get("source_file")
            and result[-1].metadata.get("section_title") == doc.metadata.get("section_title")
        ):
            result[-1].page_content = (result[-1].page_content + " " + text).strip()
            old = result[-1].metadata.get("page_numbers") or []
            new = doc.metadata.get("page_numbers") or []
            combined = sorted(set(old + new))
            result[-1].metadata["page_numbers"] = combined
            if combined:
                result[-1].metadata["start_page"] = min(combined)
                result[-1].metadata["end_page"] = max(combined)
            continue
        result.append(doc)
    return result


def split_large_chunk(text: str, max_chars: int, overlap_ratio: float = 0.15) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return [text]
    chunks, current = [], []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and len(candidate) > max_chars:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            overlap_chars = max(1, int(len(chunk_text) * overlap_ratio))
            new_current: List[str] = []
            acc = 0
            for s in reversed(current):
                if acc + len(s) + 1 > overlap_chars and new_current:
                    break
                new_current.append(s)
                acc += len(s) + 1
            current = list(reversed(new_current))
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Structural units
# ---------------------------------------------------------------------------


@dataclass
class StructuralUnit:
    section: str
    lines: List[Tuple[str, int]]  # (line_text, page_number), in original order

    @property
    def text(self) -> str:
        return "\n".join(line for line, _ in self.lines)

    @property
    def pages(self) -> List[int]:
        return sorted({page for _, page in self.lines})


def build_structural_units(prepared_pages: List[PreparedPage]) -> List[StructuralUnit]:
    """Group page lines into section-keyed units; skip TOC pages."""
    units: List[StructuralUnit] = []
    hierarchy: Dict[str, Any] = {}
    current_section = "General Context"
    current_lines: List[Tuple[str, int]] = []
    inside_toc = False

    # NEW: first pass — find heading-like lines that recur too often to be
    # real structural boundaries (e.g. a repeated in-text prompt).
    recurring_headings = find_recurring_heading_texts(prepared_pages)

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            units.append(StructuralUnit(section=current_section, lines=list(current_lines)))
        current_lines = []

    for page in prepared_pages:
        for line_idx, line in enumerate(page.lines):
            prev_line = page.lines[line_idx - 1] if line_idx > 0 else ""
            next_line = page.lines[line_idx + 1] if line_idx + 1 < len(page.lines) else ""

            if is_toc_title(line):
                flush()
                inside_toc = True
                continue
            if inside_toc:
                if is_toc_entry(line):
                    continue
                heading = detect_section(line, prev_line, next_line, recurring_headings)
                if heading:
                    inside_toc = False
                    update_hierarchy(hierarchy, heading)
                    current_section = get_context(hierarchy)
                    continue
                continue
            if is_strong_end_heading(line):
                flush()
                continue
            heading = detect_section(line, prev_line, next_line, recurring_headings)
            if heading:
                flush()
                update_hierarchy(hierarchy, heading)
                current_section = get_context(hierarchy)
                continue
            current_lines.append((line, page.page_number))
    flush()
    return units


def merge_structural_units(
    units: List[StructuralUnit],
    cfg: PreprocessingConfig,
) -> List[StructuralUnit]:
    """Merge units smaller than min_section_content into the previous
    same/related section when the combined size stays under the cap."""
    result: List[StructuralUnit] = []
    for unit in units:
        if not unit.lines:
            continue
        text_len = len(unit.text.strip())
        if text_len >= cfg.min_section_content:
            result.append(StructuralUnit(unit.section, list(unit.lines)))
            continue
        if result:
            prev = result[-1]
            related = (
                prev.section == unit.section
                or prev.section.startswith(unit.section)
                or unit.section.startswith(prev.section)
            )
            combined_len = len(prev.text) + 1 + text_len
            if related and combined_len <= cfg.max_section_group_chars:
                prev.lines = prev.lines + unit.lines
                continue
        if text_len >= cfg.min_structural_chars:
            result.append(StructuralUnit(unit.section, list(unit.lines)))
    return result


# ---------------------------------------------------------------------------
# Tiny-fragment protection (post-chunking)
# ---------------------------------------------------------------------------


def is_tiny_fragment(text: str, cfg: PreprocessingConfig) -> bool:
    text = text.strip()
    if not text:
        return True
    return len(text) < cfg.tiny_chunk_char_limit or len(text.split()) < cfg.tiny_chunk_word_limit


_MEANINGFUL_TERMS = {
    "patient", "patients", "disease", "diagnosis", "treatment",
    "management", "recommendation", "clinical", "therapy",
}


def merge_tiny_documents(
    docs: List[Document], cfg: PreprocessingConfig
) -> List[Document]:
    """Two-pass tiny-fragment merge (same file + same section), ported from
    the notebook's merge_tiny_documents."""
    result: List[Document] = []
    hold: Document | None = None
    for doc in docs:
        text = doc.page_content.strip()
        if is_tiny_fragment(text, cfg):
            # Hold the first tiny fragment for later merge attempts.
            if not result and hold is None:
                hold = doc
                continue
            if hold is not None:
                # Merge held fragment into the first normal chunk if adjacent.
                result.append(hold)
                hold = None
            if result:
                prev = result[-1]
                same_file = (
                    prev.metadata.get("source_file") == doc.metadata.get("source_file")
                )
                same_section = (
                    prev.metadata.get("section_title")
                    == doc.metadata.get("section_title")
                )
                if same_file and same_section:
                    prev.page_content = f"{prev.page_content} {text}".strip()
                    _merge_pages(prev, doc)
                    continue
            result.append(doc)
        else:
            if hold is not None:
                result.append(hold)
                hold = None
            result.append(doc)
    # Second pass: if the very first doc is still tiny, merge into the second.
    if len(result) >= 2 and is_tiny_fragment(result[0].page_content, cfg):
        first, second = result[0], result[1]
        if (
            first.metadata.get("source_file") == second.metadata.get("source_file")
            and first.metadata.get("section_title")
            == second.metadata.get("section_title")
        ):
            second.page_content = f"{first.page_content} {second.page_content}".strip()
            _merge_pages(first, second)
            result.pop(0)
    return result


def _merge_pages(prev: Document, doc: Document) -> None:
    old = prev.metadata.get("page_numbers") or []
    new = doc.metadata.get("page_numbers") or []
    combined = sorted(set(old + new))
    prev.metadata["page_numbers"] = combined
    if combined:
        prev.metadata["start_page"] = min(combined)
        prev.metadata["end_page"] = max(combined)


def _chunk_quality_score(lengths: List[int], cfg: PreprocessingConfig) -> float:
    if not lengths:
        return -1.0
    total = len(lengths)
    ideal = sum(cfg.ideal_min_chars <= x <= cfg.ideal_max_chars for x in lengths)
    small = sum(x < cfg.min_chunk_chars for x in lengths)
    large = sum(x > cfg.large_chunk_chars for x in lengths)
    return 100 * ideal / total - 100 * small / total - 80 * large / total


def chunk_quality_report(lengths: List[int], cfg: PreprocessingConfig) -> Dict[str, Any]:
    if not lengths:
        return {}
    return {
        "num_chunks": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "average": round(sum(lengths) / len(lengths), 2),
        "median": sorted(lengths)[len(lengths) // 2],
        "small": sum(x < cfg.min_chunk_chars for x in lengths),
        "large": sum(x > cfg.large_chunk_chars for x in lengths),
    }


def _line_offsets(lines: List[Tuple[str, int]]) -> List[Tuple[int, int, int]]:
    """(char_start, char_end, page_number) for each line as it sits inside
    '\\n'.join(line for line, _ in lines) -- i.e. StructuralUnit.text."""
    spans = []
    pos = 0
    for line, page in lines:
        start = pos
        end = start + len(line)
        spans.append((start, end, page))
        pos = end + 1  # +1 for the newline joiner
    return spans


def _pages_for_span(
    start: int, end: int, line_spans: List[Tuple[int, int, int]]
) -> List[int]:
    """Pages touched by the [start, end) character range, based on which
    lines' own spans overlap it."""
    pages = {
        page
        for line_start, line_end, page in line_spans
        if line_start < end and line_end > start
    }
    return sorted(pages)


# ---------------------------------------------------------------------------
# Main chunking stage
# ---------------------------------------------------------------------------


def build_base_documents(
    units: List[StructuralUnit],
    units_pages: Dict[str, List[int]],
    pdf_name: str,
    pdf_path: str,
    total_pages: int,
    semantic_chunker: SemanticChunker,
    chunking_cfg: ChunkingConfig,
    preprocessing_cfg: PreprocessingConfig,
) -> Tuple[Document, Dict[str, Any]]:
    """Run the semantic-chunker + size-control grid for ONE pdf.

    Returns (best_document_list_holder, results_dict). The holder's
    `.page_content` / `.metadata` pattern is preserved from the notebook so
    downstream enrichment code needs no changes.
    """
    configuration_results: Dict[str, Any] = {}

    for unit in units:
        text = unit.text.strip()
        if not text:
            continue

        line_spans = _line_offsets(unit.lines)

        try:
            semantic_docs = semantic_chunker.create_documents([text])
        except Exception as exc:  # noqa: BLE001
            print(f"    Warning: semantic chunking failed for a unit: {exc}")
            semantic_docs = [Document(page_content=text, metadata={})]

        base_docs: List[Document] = []
        search_pos = 0
        for s_doc in semantic_docs:
            chunk_text = s_doc.page_content.strip()
            if not chunk_text:
                continue

            # Locate this sub-chunk within the unit's text so we can derive
            # the pages it actually spans, instead of inheriting every page
            # the whole (possibly much larger) unit touches.
            idx = text.find(chunk_text, search_pos)
            if idx == -1:
                idx = text.find(chunk_text)  # fallback: search from the start
            if idx == -1:
                idx = search_pos  # last resort

            chunk_start = idx
            chunk_end = idx + len(chunk_text)
            search_pos = chunk_end

            chunk_pages = _pages_for_span(chunk_start, chunk_end, line_spans) or unit.pages

            base_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": pdf_name,
                        "source_path": pdf_path,
                        "total_pages": total_pages,
                        "start_page": (min(chunk_pages) if chunk_pages else None),
                        "end_page": (max(chunk_pages) if chunk_pages else None),
                        "page_numbers": chunk_pages,
                        "section_title": unit.section,
                    },
                )
            )

        # Post-hoc size control: keep every split/merge variant, score them,
        # and select the best per-pdf (same grid as the notebook).
        docs_by_variant: Dict[str, List[Document]] = {}
        for max_chars in chunking_cfg.target_max_options:
            for overlap in chunking_cfg.overlap_options:
                variant_docs: List[Document] = []
                for doc in base_docs:
                    pieces = split_large_chunk(doc.page_content, max_chars, overlap)
                    for piece in pieces:
                        variant_docs.append(
                            Document(
                                page_content=piece,
                                metadata=dict(doc.metadata),
                            )
                        )
                merged = merge_tiny_documents(variant_docs, preprocessing_cfg)
                merged = repair_mid_sentence_starts(merged)
                # Drop remaining tiny fragments without meaningful content.
                kept: List[Document] = []
                for doc in merged:
                    if is_tiny_fragment(doc.page_content, preprocessing_cfg):
                        if not any(t in doc.page_content.lower() for t in _MEANINGFUL_TERMS):
                            continue
                    kept.append(doc)
                name = f"max_{max_chars}_overlap_{overlap}"
                docs_by_variant[name] = kept

        # Best variant for this unit's family: evaluated jointly below is
        # not possible, so we accumulate per variant across units instead.
        # (The notebook ran the grid per-file on ALL units; here we do the
        # same by accumulating variant docs across units.)

        for name, docs in docs_by_variant.items():
            bucket = configuration_results.setdefault(name, [])
            bucket.extend(docs)

    # Score every variant and pick the best.
    best_name, best_docs = None, []
    summary = {}
    for name, docs in configuration_results.items():
        lengths = [len(d.page_content) for d in docs]
        score = _chunk_quality_score(lengths, preprocessing_cfg)
        summary[name] = {
            **chunk_quality_report(lengths, preprocessing_cfg),
            "quality_score": round(score, 3),
        }
        if best_name is None or score > summary[best_name]["quality_score"]:
            best_name, best_docs = name, docs

    print(f"    Best size-control variant: {best_name}")
    return Document(
        page_content="", metadata={"docs": best_docs},
    ), summary
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
"""Metadata enrichment: visual references, clinical-note flags, acronym
expansion, stable ProcessedChunk construction, JSON persistence.

Port of the notebook's Block 9. The acronym dictionary is loaded from
config/acronyms.yaml so new terms can be added without touching code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ingestion import PdfInfo
from .models import ChunkMetadata, ProcessedChunk

# ---------------------------------------------------------------------------
# Visual reference detection
# ---------------------------------------------------------------------------

VISUAL_REFERENCE_PATTERNS = [
    re.compile(r"figure\s+\d+[A-Z]?\b", re.IGNORECASE),
    re.compile(r"fig\.\s*\d+[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\btable\s+\d+[A-Z]?\b", re.IGNORECASE),
    re.compile(r"\bpanel\s+[A-Z]\b", re.IGNORECASE),
    re.compile(r"(?:supplementary\s+)?(?:online\s+)?(?:figure|table)\s+\d+", re.IGNORECASE),
    re.compile(r"\bvideo\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bappendix\s+figure\s+\d+", re.IGNORECASE),
]


def detect_visual_references(text: str) -> List[str]:
    hits: List[str] = []
    for pattern in VISUAL_REFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(match.group(0).strip())
    return sorted(set(hits))


# ---------------------------------------------------------------------------
# Clinical note detection
# ---------------------------------------------------------------------------

CLINICAL_INDICATORS = {
    "patient", "patients", "diagnosis", "treatment", "management",
    "recommendation", "clinical", "therapy", "disease", "trial",
    "guideline", "outcome", "mortality", "prognosis", "follow-up",
}


def is_clinical_note(text: str, threshold: int = 2) -> bool:
    lower = text.lower()
    return sum(term in lower for term in CLINICAL_INDICATORS) >= threshold


# ---------------------------------------------------------------------------
# Acronym expansion (dictionary-driven, matches notebook semantics)
# ---------------------------------------------------------------------------

DEFAULT_ACRONYM_DICTIONARY: Dict[str, str] = {
    "PDA": "patent ductus arteriosus",
    "LAD": "left anterior descending",
    "SVR": "systemic vascular resistance",
    "TOF": "tetralogy of Fallot",
    "HR": "heart rate",
    "ASD": "atrial septal defect",
    "VSD": "ventricular septal defect",
    "CAD": "coronary artery disease",
    "HF": "heart failure",
    "HFrEF": "heart failure with reduced ejection fraction",
    "HFpEF": "heart failure with preserved ejection fraction",
    "NYHA": "New York Heart Association",
    "LV": "left ventricle",
    "RV": "right ventricle",
    "EF": "ejection fraction",
    "ACEI": "ACE inhibitor",
    "ARB": "angiotensin receptor blocker",
    "BB": "beta-blocker",
    "MRA": "mineralocorticoid receptor antagonist",
    "SGLT2": "sodium-glucose cotransporter-2",
    "CKM": "cardiovascular-kidney-metabolic",
    "AHA": "American Heart Association",
    "ACC": "American College of Cardiology",
    "ESC": "European Society of Cardiology",
    "BP": "blood pressure",
    "SBP": "systolic blood pressure",
    "DBP": "diastolic blood pressure",
    "MI": "myocardial infarction",
    "CHF": "congestive heart failure",
    "AF": "atrial fibrillation",
    "VT": "ventricular tachycardia",
    "VF": "ventricular fibrillation",
    "RAAS": "renin-angiotensin-aldosterone system",
    "LVH": "left ventricular hypertrophy",
    "CMR": "cardiovascular magnetic resonance",
    "ECG": "electrocardiogram",
    "CXR": "chest X-ray",
}


def load_acronym_dictionary(path: Optional[Path]) -> Dict[str, str]:
    """Merge the built-in dictionary with config/acronyms.yaml if present."""
    merged = dict(DEFAULT_ACRONYM_DICTIONARY)
    if path and path.exists():
        import yaml  # local import: optional dependency
        with path.open(encoding="utf-8") as fh:
            user_dict = yaml.safe_load(fh) or {}
        for key, value in user_dict.items():
            merged[str(key).upper()] = str(value)
    return dict(sorted(merged.items(), key=lambda kv: -len(kv[0])))


def expand_medical_acronyms(text: str, dictionary: Dict[str, str]) -> str:
    """Expand acronyms that appear WITHOUT a following parenthetical full
    term, exactly like the notebook's MedicalAcronymExpander."""
    words = re.split(r"(\s+)", text)
    out: List[str] = []
    for i, word in enumerate(words):
        stripped = word.strip()
        upper = stripped.strip(".,;:)")
        if upper in dictionary:
            # Skip when the next non-empty word starts with the parenthetical
            # full term (e.g. "HF (heart failure)").
            rest = " ".join(words[i + 1 : i + 4])
            if re.search(r"\(\s*" + re.escape(dictionary[upper].split()[0]), rest):
                out.append(word)
                continue
            out.append(f"{stripped} ({dictionary[upper]})")
        else:
            out.append(word)
    return " ".join(w.strip() for w in out if w.strip())


# ---------------------------------------------------------------------------
# Enrichment + stable chunk construction
# ---------------------------------------------------------------------------


def enrich_and_build(
    final_docs: list,
    pdf_infos: Dict[str, PdfInfo],
    acronym_dictionary: Dict[str, str],
) -> Tuple[List[ProcessedChunk], Dict[str, int]]:
    """Convert post-chunking Documents into validated ProcessedChunks.

    final_docs : List[langchain Document] (the best-variant output of
                 chunking.build_base_documents, plus any final safety
                 cleanup applied by the caller).
    pdf_infos  : PdfInfo by normalized file name.
    Returns (chunks, per-file chunk counts).
    """
    file_index: Dict[str, int] = {name: 0 for name in pdf_infos}
    chunks: List[ProcessedChunk] = []
    global_index = 0
    seen_ids = set()

    for doc in final_docs:
        if not hasattr(doc, "page_content") or not doc.page_content.strip():
            continue

        meta = doc.metadata or {}
        source_file = str(meta.get("source_file", "")).strip()
        if not source_file:
            continue

        norm = Path(source_file).name.lower()
        info = pdf_infos.get(norm)
        page_numbers = meta.get("page_numbers") or []

        text = doc.page_content.strip()
        chunk_id = f"{norm}:{meta.get('section_title','general')}:{len(chunks)}"
        if chunk_id in seen_ids:
            chunk_id = f"{chunk_id}:{global_index}"
        seen_ids.add(chunk_id)

        file_index[norm] += 1
        local_index = file_index[norm]
        total_in_file = 0  # filled in a second pass below

        chunk_metadata = ChunkMetadata(
            source_file=source_file,
            file_path=meta.get("source_path", ""),
            file_hash_sha256=info.hash_sha256 if info else "",
            file_size_bytes=info.size_bytes if info else 0,
            mime_type=info.mime_type if info else "application/pdf",
            start_page=meta.get("start_page"),
            end_page=meta.get("end_page"),
            page_numbers=page_numbers,
            total_doc_pages=info.total_pages if info else meta.get("total_pages", 0),
            section_title=str(meta.get("section_title", "General Context")),
            contains_clinical_note=is_clinical_note(text),
            has_visual_reference=bool(detect_visual_references(text)),
            visual_references=detect_visual_references(text),
            char_count=len(text),
            word_count=len(text.split()),
            estimated_tokens=int(len(text.split()) * 1.3),
        )
        text = " ".join(text.split())
        chunks.append(
            ProcessedChunk(
                chunk_id=chunk_id,
                original_text=text,
                expanded_text=expand_medical_acronyms(text, acronym_dictionary),
                metadata=chunk_metadata,
            )
        )
        global_index += 1

    # Second pass: fill per-file totals and indices.
    file_counts: Dict[str, int] = {}
    for chunk in chunks:
        file_counts.setdefault(chunk.metadata.source_file, 0)
    for chunk in chunks:
        file_counts[chunk.metadata.source_file] += 1
    per_file_seen: Dict[str, int] = {}
    for chunk in chunks:
        key = chunk.metadata.source_file
        per_file_seen[key] = per_file_seen.get(key, 0) + 1
        chunk.metadata.local_chunk_index = per_file_seen[key]
        chunk.metadata.total_chunks_in_file = file_counts[key]
    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata.global_chunk_index = i

    per_file = {norm: file_counts.get(norm, 0) for norm in pdf_infos}
    return chunks, per_file


def file_coverage_report(chunks: List[ProcessedChunk], pdf_infos: Dict[str, PdfInfo]) -> None:
    counts: Dict[str, int] = {}
    for chunk in chunks:
        norm = Path(chunk.metadata.source_file).name.lower()
        counts[norm] = counts.get(norm, 0) + 1
    print("\n    File coverage:")
    for name, info in pdf_infos.items():
        print(f"      {info.name}: {counts.get(name, 0)} chunks")
"""Final retrieval evaluation export (notebook Block 12).

Runs the selected retriever configuration over the evaluation question set
and writes a self-describing JSON document (evidence + chunk metadata) that
can be consumed by notebooks, dashboards, or CI checks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .retrieval import Retriever


DEFAULT_TEST_QUERIES: List[Dict[str, str]] = [
    {
        "question": "What alternative medicines to clopidogrel exist and who can prescribe them?",
        "keywords": [
            "prasugrel",
            "ticagrelor",
            "clopidogrel",
            "specialist"
        ],
        "note": "Answer spans a 163-char chunk ending mid-sentence ('T hese need to be...') plus the next chunk."
    },
    {
        "question": "How does prasugrel work as an antiplatelet medicine?",
        "keywords": [
            "prasugrel",
            "platelet inhibitor",
            "clumping",
            "blood clot"
        ],
        "note": "245-char chunk cut off mid-sentence ('instead of')."
    },
    {
        "question": "What should you do if you need to stop taking beta-blockers?",
        "keywords": [
            "beta-blockers",
            "stop",
            "medical advice",
            "calcium channel blockers"
        ],
        "note": "Two unrelated topics (beta-blocker warning + calcium channel blocker intro) got merged into one 174-char chunk."
    },
    {
        "question": "What is the funny current (If) and what role does it play in phase 4 of the cardiac action potential?",
        "keywords": [
            "funny current",
            "phase 4",
            "diastolic depolarization",
            "SA node"
        ],
        "note": "207-char chunk cut off mid-word ('ve...')."
    },
    {
        "question": "How do acetylcholine and catecholamines affect heart rate through the SA node?",
        "keywords": [
            "acetylcholine",
            "catecholamines",
            "SA node",
            "heart rate",
            "depolarization"
        ],
        "note": "248-char chunk, sympathetic/parasympathetic content likely continues into next chunk."
    },
    {
        "question": "What is cardiac cell depolarization and how does it occur in pacemaker cells?",
        "keywords": [
            "depolarization",
            "pacemaker cells",
            "spontaneously",
            "polarized"
        ],
        "note": "Two tiny chunks (137 and 154 chars) that are really one continuous idea about resting potential -> depolarization."
    },
    {
        "question": "How do nitrates cause vasodilation at the molecular level?",
        "keywords": [
            "nitrates",
            "nitric oxide",
            "guanylate cyclase",
            "cyclic GMP"
        ],
        "note": "241-char chunk, mechanism description likely continues into the next chunk for the full pathway."
    },
    {
        "question": "How do niacin's effects on lipolysis change plasma lipid levels?",
        "keywords": [
            "niacin",
            "lipolysis",
            "VLDL",
            "LDL",
            "HDL"
        ],
        "note": "167-char chunk with no drug name stated -- context (which drug this describes) lives in a preceding chunk."
    },
    {
        "question": "How does septum primum development contribute to atrial septation?",
        "keywords": [
            "septum primum",
            "endocardial cushion",
            "primitive atrium"
        ],
        "note": "Whole section (5/5 chunks) is fragmented; answer requires stitching together septum primum, foramen primum, and foramen secundum chunks."
    },
    {
        "question": "What is the relationship between the foramen primum and the foramen secundum during atrial septation?",
        "keywords": [
            "foramen primum",
            "foramen secundum",
            "septum primum",
            "shunt"
        ],
        "note": "Answer requires combining 3 separate ~190-220 char chunks describing sequential embryological steps."
    },
    {
        "question": "What do MRAs (mineralocorticoid receptor antagonists) do for heart failure patients?",
        "keywords": [
            "MRA",
            "spironolactone",
            "eplerenone",
            "blood pressure",
            "salt"
        ],
        "note": "161-char chunk; drug class name and mechanism are compressed into a very short fragment."
    },
    {
        "question": "What do diuretics do for heart failure patients and what are some examples?",
        "keywords": [
            "diuretics",
            "furosemide",
            "fluid",
            "lungs"
        ],
        "note": "237-char chunk cut off mid-sentence ('and other')."
    },
    {
        "question": "What happens to the ductus arteriosus shunt direction after birth in patent ductus arteriosus?",
        "keywords": [
            "ductus arteriosus",
            "shunt",
            "pulmonary vascular resistance",
            "left to right"
        ],
        "note": "120-char chunk -- one of the smallest in the corpus, describing a specific physiological transition."
    },
    {
        "question": "How is patent ductus arteriosus treated pharmacologically, and what keeps it open when needed?",
        "keywords": [
            "indomethacin",
            "PGE",
            "patent ductus arteriosus",
            "prostaglandin"
        ],
        "note": "249-char chunk cut off mid-word ('Narrowing')."
    },
    {
        "question": "What causes the infantile (preductal) form of aortic coarctation?",
        "keywords": [
            "coarctation",
            "aorta",
            "tunica media",
            "ductus arteriosus",
            "preductal"
        ],
        "note": "230-char chunk cut off mid-sentence."
    },
    {
        "question": "What class Ic antiarrhythmic effect do these drugs have on the cardiac action potential?",
        "keywords": [
            "class Ic",
            "action potential",
            "conduction",
            "tachycardia"
        ],
        "note": "169-char chunk with no drug name given -- requires context from a preceding chunk to know which drug class."
    },
    {
        "question": "How does ezetimibe lower LDL cholesterol and what are its side effects?",
        "keywords": [
            "cholesterol absorption",
            "LDL",
            "side effect",
            "gastrointestinal",
            "LFTs"
        ],
        "note": "216-char chunk -- drug name likely never appears in this specific chunk despite describing ezetimibe's mechanism."
    },
    {
        "question": "Why is medication adherence important for heart failure patients?",
        "keywords": [
            "medication adherence",
            "heart failure",
            "prescriptions",
            "health care team"
        ],
        "note": "Two near-duplicate ~150-char chunks describing the same medications list, likely a chunking/dedup artifact."
    },
    {
        "question": "What are antiplatelet medicines used for and who should avoid them?",
        "keywords": [
            "antiplatelet",
            "high risk",
            "recommended",
            "treatment"
        ],
        "note": "202-char chunk cut off mid-sentence ('aren't at high ri...')."
    },
    {
        "question": "What does the slope of phase 4 depolarization in the SA node control?",
        "keywords": [
            "phase 4",
            "SA node",
            "heart rate",
            "slope"
        ],
        "note": "Tests whether a 248-char chunk fragment retrieves correctly despite starting mid-topic."
    }
]


def run_evaluation(
    retriever: Retriever,
    questions: List[Dict[str, str]],
    embedding_model_name: str,
    reranker_model_name: str,
    output_path: Path,
) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "retriever_config": {
            "search_type": retriever.cfg.search_types[0],
            "k": retriever.cfg.k_values[0],
            "rerank_k": retriever.cfg.rerank_k,
        },
        "questions": [],
    }

    for item in questions:
        results = retriever.retrieve(item["question"], retriever.cfg.k_values[0],
                                     retriever.cfg.rerank_k)
        report["questions"].append({
            "question": item["question"],
            "keywords": item.get("keywords", []),
            "results": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk.chunk_id,
                    "dense_score": round(r.dense_score, 4),
                    "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
                    "source_file": r.chunk.metadata.source_file,
                    "section_title": r.chunk.metadata.section_title,
                    "page_numbers": r.chunk.metadata.page_numbers,
                    "text": r.chunk.original_text,
                    "expanded_text": r.chunk.expanded_text,
                }
                for r in results
            ],
        })

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"    Evaluation report saved: {output_path}")
    return report
"""PDF ingestion: discovery, hashing, extraction, cleaning.

Mirrors the notebook's "PDF Extraction" block but as a pure, testable
module with no dependence on notebook globals. Output per file:
    {"pages": List[str], "total_pages": int}
"""

from __future__ import annotations

import hashlib
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


@dataclass
class PdfInfo:
    name: str
    path: Path
    hash_sha256: str
    size_bytes: int
    mime_type: str
    total_pages: int
    pages: List[str] = field(default_factory=list)


def compute_file_hash(path: Path) -> str:
    """SHA-256 of the raw file — used for cache invalidation and provenance."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_pdf_text(text: str) -> str:
    """Basic text repair copied from the notebook: line endings, whitespace,
    hyphen-joined words split across lines."""
    if not text:
        return text
    # Hard line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Hyphenated word continuation: "cardio- \n vascular" -> "cardiovascular"
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(path: Path) -> Tuple[List[str], int]:
    """Extract per-page text. Returns (pages, total_pages)."""
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(clean_pdf_text(page.extract_text() or ""))
    return pages, len(pages)


def discover_pdfs(data_dir: Path) -> List[Path]:
    """Find all PDFs in the data directory (non-recursive)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    return sorted(p for p in data_dir.glob("*.pdf") if not p.name.startswith("."))


def ingest_pdf(path: Path) -> PdfInfo:
    """Full ingestion of a single PDF: metadata + per-page text."""
    pages, total = extract_pdf_pages(path)
    mime, _ = mimetypes.guess_type(str(path))
    return PdfInfo(
        name=path.name,
        path=path.resolve(),
        hash_sha256=compute_file_hash(path),
        size_bytes=path.stat().st_size,
        mime_type=mime or "application/pdf",
        total_pages=total,
        pages=pages,
    )


def ingest_all(data_dir: Path) -> List[PdfInfo]:
    results = []
    for path in discover_pdfs(data_dir):
        print(f"  Ingesting {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
        info = ingest_pdf(path)
        results.append(info)
        print(
            f"    {info.total_pages} pages, "
            f"{sum(len(p) for p in info.pages)} chars extracted"
        )
    return results
"""Shared domain models.

ChunkMetadata + ProcessedChunk were originally dataclasses scattered inside
the notebook's Block 9. They are promoted to a first-class, serializable
module used by every downstream stage (enrichment, embedding, retrieval).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChunkMetadata:
    """Everything the pipeline knows about one chunk's provenance."""

    source_file: str
    file_path: str
    file_hash_sha256: str
    file_size_bytes: int
    mime_type: str
    start_page: Optional[int]
    end_page: Optional[int]
    page_numbers: List[int]
    total_doc_pages: int
    section_title: str
    chunk_index: int = 0
    local_chunk_index: int = 0
    global_chunk_index: int = 0
    total_chunks_in_file: int = 0
    char_count: int = 0
    word_count: int = 0
    estimated_tokens: int = 0
    contains_clinical_note: bool = False
    has_visual_reference: bool = False
    visual_references: List[str] = field(default_factory=list)
    parser_type: str = "complex"
    content_type: str = "text"
    table_atomic: bool = False
    specialized_guideline: bool = False
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=now_utc_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedChunk:
    """One atomic, retrievable unit of knowledge.

    original_text : the exact text emitted by the chunker (what gets embedded)
    expanded_text : original_text with medical acronyms expanded (for LLM context)
    """

    chunk_id: str
    original_text: str
    expanded_text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "original_text": self.original_text,
            "expanded_text": self.expanded_text,
            "metadata": self.metadata.to_dict(),
        }

    # ------------------------------------------------------------------ #
    # Persistence helpers — replaces fragile "save final chunk list via
    # whatever is in globals()" from the notebook.
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_all(chunks: List["ProcessedChunk"], path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in chunks], fh, ensure_ascii=False, indent=2)

    @staticmethod
    def load_all(path: Path) -> List["ProcessedChunk"]:
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        chunks: List[ProcessedChunk] = []
        for item in data:
            metadata = ChunkMetadata(**item["metadata"])
            chunks.append(
                ProcessedChunk(
                    chunk_id=item["chunk_id"],
                    original_text=item["original_text"],
                    expanded_text=item.get("expanded_text", item["original_text"]),
                    metadata=metadata,
                )
            )
        return chunks
"""Shared spaCy pipeline loader.

Loaded once per process and reused everywhere (sentence splitting in
chunking.py, lemmatization-based keyword matching in retrieval.py)
instead of each module loading its own copy of the model.
"""

from __future__ import annotations

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = spacy.blank("en")
        if "sentencizer" not in _nlp.pipe_names and "parser" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp
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
#!/usr/bin/env python3
"""Medical RAG CLI.

Usage:
    python main.py process            # ingest + chunk + enrich + save chunks
    python main.py embed              # embed chunks (or load cached) + select retriever
    python main.py serve              # interactive question loop over the index
    python main.py eval               # run the saved retriever on the eval set
    python main.py run all            # everything, end to end
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import AppConfig  # noqa: E402
from src.evaluation import DEFAULT_TEST_QUERIES, run_evaluation  # noqa: E402
from src.models import ProcessedChunk  # noqa: E402
from src.pipeline import build_retrieval_stack, process_corpus  # noqa: E402


def get_retriever(cfg: AppConfig):
    return build_retrieval_stack(cfg)


def cmd_process(cfg: AppConfig) -> None:
    process_corpus(cfg)


def cmd_embed(cfg: AppConfig) -> None:
    build_retrieval_stack(cfg)


def cmd_serve(cfg: AppConfig) -> None:
    retriever = get_retriever(cfg)
    print("\nAsk questions (Ctrl+C to quit):\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        results = retriever.retrieve(question, retriever.cfg.k_values[0],
                                     retriever.cfg.rerank_k)
        for r in results:
            print(
                f"\n  [{r.rank}] (dense={r.dense_score:.3f}, "
                f"rerank={r.rerank_score:.2f}) "
                f"{r.chunk.metadata.source_file} p{r.chunk.metadata.page_numbers}"
            )
            print(f"      {r.chunk.original_text[:300]}...")


def cmd_eval(cfg: AppConfig) -> None:
    retriever = get_retriever(cfg)
    from src.embeddings import build_embedder  # noqa: E402
    embedder_name = (
        cfg.embeddings.groq_model
        if cfg.embeddings.use_groq
        else cfg.embeddings.index_embedder
    )
    run_evaluation(
        retriever, DEFAULT_TEST_QUERIES, embedder_name,
        cfg.embeddings.reranker_model,
        Path(cfg.paths.output_dir) / cfg.paths.evaluation_json,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Medical RAG pipeline")
    parser.add_argument("command", choices=["process", "embed", "serve", "eval", "run"],
                        nargs="?", default="run", help="default: run (full pipeline)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="path to config.yaml")
    parser.add_argument("--data-dir", help="override data/ dir in config")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(Path(args.config))
    if args.data_dir:
        cfg.paths.data_dir = args.data_dir

    if args.command == "run":
        cmd_process(cfg)
        cmd_embed(cfg)
        cmd_eval(cfg)
    else:
        {"process": cmd_process, "embed": cmd_embed,
         "serve": cmd_serve, "eval": cmd_eval}[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
