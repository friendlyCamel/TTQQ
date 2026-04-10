from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from ..llm import OpenAICompatLLM
from ..schemas import PaperCandidate


class VectorRAGStore:
    def __init__(self, llm: OpenAICompatLLM, db_path: str | Path = "data/memory/rag_vectors.db"):
        self.llm = llm
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    url TEXT,
                    chunk_id INTEGER,
                    text TEXT,
                    vec TEXT,
                    uniq_key TEXT UNIQUE
                )
                """
            )
            conn.commit()

    def upsert_papers(self, papers: list[PaperCandidate]) -> None:
        rows: list[tuple[str, str, int, str, str, str]] = []
        for p in papers:
            text = p.extracted_text or p.abstract
            if not text:
                continue
            chunks = _chunk_text(text, size=280)
            for i, chunk in enumerate(chunks[:16]):
                vec = self.llm.embed(chunk)
                uniq_key = f"{p.title}::{i}"
                rows.append((p.title, p.url, i, chunk, json.dumps(vec), uniq_key))

        if not rows:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO rag_chunks(title, url, chunk_id, text, vec, uniq_key)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def search(self, query: str, top_k: int = 6) -> list[str]:
        q_vec = self.llm.embed(query)
        scored: list[tuple[float, str]] = []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT title, text, vec FROM rag_chunks")
            for title, text, vec_raw in cur.fetchall():
                vec = json.loads(vec_raw)
                score = _cosine(q_vec, vec)
                if score > 0.2:
                    scored.append((score, f"{title}: {text[:260]}"))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]


def _chunk_text(text: str, size: int = 280) -> list[str]:
    words = text.split()
    out: list[str] = []
    for i in range(0, len(words), size):
        out.append(" ".join(words[i : i + size]))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n))) or 1.0
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n))) or 1.0
    return dot / (na * nb)
