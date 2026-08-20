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
