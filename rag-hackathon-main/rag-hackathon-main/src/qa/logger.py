"""Traceability logger for the Prompting phase.

Records the original question, every sub-query, per-query retrieval scores,
the merged (max-per-chunk) scores, the threshold gate decision, parsed
claims and their per-claim confidence. Printed to the console and optionally
persisted as JSON for the mandatory test report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TraceLogger:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._data: Dict[str, Any] = {"events": []}

    # ------------------------------------------------------------------ #
    # Populating
    # ------------------------------------------------------------------ #

    def start(self, question: str) -> None:
        self._data["question"] = question

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def add_event(self, message: str) -> None:
        self._data["events"].append(message)

    def add_query(self, query: str, results) -> None:
        entry = {
            "query": query,
            "top_results": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk.chunk_id,
                    "source": r.chunk.metadata.source_file,
                    "pages": r.chunk.metadata.page_numbers,
                    "rerank": round(r.rerank_score, 4)
                    if r.rerank_score is not None
                    else None,
                    "dense": round(r.dense_score, 4),
                }
                for r in results
            ],
        }
        self._data.setdefault("sub_query_results", []).append(entry)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def summary_line(self) -> str:
        gate = self._data.get("gate", {})
        overall = self._data.get("overall", {})
        return (
            f"question={self._data.get('question')!r} | "
            f"sub_queries={len(self._data.get('sub_queries', []))} | "
            f"gate={gate.get('decision')} (best={gate.get('best_score')}) | "
            f"overall={overall.get('level')} ({overall.get('score')})"
        )

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path or self.path) if (path or self.path) else None
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, ensure_ascii=False, indent=2)