from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import PaperCandidate, ResearchCase


class LongTermMemory:
    def __init__(self, root: str | Path = "data/memory"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.case_log = self.root / "cases.jsonl"
        self.paper_log = self.root / "papers.jsonl"
        self.dialogue_log = self.root / "analogy_dialogues.jsonl"

    def remember_case(self, case: ResearchCase) -> None:
        item: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "input_type": case.input_type,
            "mode": case.mode,
            "user_input": case.user_input,
            "problem_frame": asdict(case.problem_frame) if case.problem_frame else {},
            "routes": [asdict(x) for x in case.routes],
            "stage_recommendation": case.stage_recommendation,
            "judge_audit": asdict(case.judge_audit),
            "reflection_notes": case.reflection_notes,
        }
        with self.case_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def remember_papers(self, papers: list[PaperCandidate]) -> None:
        with self.paper_log.open("a", encoding="utf-8") as f:
            for p in papers:
                item = asdict(p)
                item["ts"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def retrieve_related_notes(self, query: str, top_k: int = 5) -> list[str]:
        if not self.case_log.exists():
            return []
        q_terms = _tokenize(query)
        scored: list[tuple[int, str]] = []
        for line in self.case_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = f"{row.get('user_input', '')} {row.get('problem_frame', {})}"
            score = _overlap_score(q_terms, _tokenize(text))
            if score > 0:
                scored.append((score, text[:800]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def remember_analogy_dialogue(self, case: ResearchCase) -> None:
        if not case.analogy or not case.analogy.dialogue_log:
            return
        item: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user_input": case.user_input,
            "task": case.problem_frame.task if case.problem_frame else "",
            "dialogue": [asdict(x) for x in case.analogy.dialogue_log],
        }
        with self.dialogue_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in text.replace("\n", " ").split() if t.strip()}


def _overlap_score(a: set[str], b: set[str]) -> int:
    return len(a.intersection(b))
