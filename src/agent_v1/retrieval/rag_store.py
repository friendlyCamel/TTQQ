from __future__ import annotations

import json
from pathlib import Path

from ..schemas import PaperCandidate


class SimpleRAGStore:
    def __init__(self, db_path: str | Path = "data/memory/rag_chunks.jsonl"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def upsert_papers(self, papers: list[PaperCandidate]) -> None:
        with self.db_path.open("a", encoding="utf-8") as f:
            for p in papers:
                text = p.extracted_text or p.abstract
                if not text:
                    continue
                chunks = _chunk_text(text, size=500)
                for i, chunk in enumerate(chunks[:20]):
                    row = {
                        "title": p.title,
                        "url": p.url,
                        "chunk_id": i,
                        "text": chunk,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def search(self, query: str, top_k: int = 6) -> list[str]:
        if not self.db_path.exists():
            return []
        q = _tokenize(query)
        hits: list[tuple[int, str]] = []
        for line in self.db_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            score = len(q.intersection(_tokenize(text)))
            if score > 0:
                hits.append((score, f"{row.get('title', '')}: {text[:260]}"))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in hits[:top_k]]


def _chunk_text(text: str, size: int = 500) -> list[str]:
    words = text.split()
    out: list[str] = []
    for i in range(0, len(words), size):
        out.append(" ".join(words[i : i + size]))
    return out


def _tokenize(text: str) -> set[str]:
    return {x.lower() for x in text.split() if x.strip()}
