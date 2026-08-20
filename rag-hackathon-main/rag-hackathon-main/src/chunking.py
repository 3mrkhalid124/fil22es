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