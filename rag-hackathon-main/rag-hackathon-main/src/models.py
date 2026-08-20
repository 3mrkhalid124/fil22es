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
