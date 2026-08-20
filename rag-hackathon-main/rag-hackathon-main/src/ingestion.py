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
