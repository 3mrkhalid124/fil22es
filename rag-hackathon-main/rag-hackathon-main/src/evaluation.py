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
