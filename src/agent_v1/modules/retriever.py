from __future__ import annotations

from pathlib import Path

from ..llm import OpenAICompatLLM
from ..memory import LongTermMemory
from ..retrieval.pdf_reader import PDFPipeline
from ..retrieval.semantic_scholar import SemanticScholarRetriever
from ..retrieval.vector_store import VectorRAGStore
from ..schemas import AnalogyResult, PaperCandidate, ProblemFrame


class CrossDomainRetriever:
    def __init__(self, llm: OpenAICompatLLM, project_root: str | Path = "data/default"):
        root = Path(project_root)
        mem_root = root / "memory"
        paper_root = root / "papers"
        self.llm = llm
        self.ss = SemanticScholarRetriever()
        self.pdf = PDFPipeline(paper_root)
        self.rag = VectorRAGStore(llm, mem_root / "rag_vectors.db")
        self.mem = LongTermMemory(mem_root)

    def build_queries(self, frame: ProblemFrame, analogy: AnalogyResult, k: int = 8) -> list[str]:
        base = analogy.query_families[:k]
        rag_hints = self.rag.search(frame.user_input, top_k=3)
        if rag_hints:
            hint = " ".join(x[:120] for x in rag_hints)
            base = [*base, f"{frame.task} {hint}"]

        if base:
            return _dedup(base)[:k]

        system = "输出 JSON: {queries: string[]}"
        user = (
            f"task={frame.task}\n"
            f"goal={frame.current_goal}\n"
            f"mechanisms={analogy.constraint_filtered_mechanisms}\n"
            "请生成 6-8 个跨领域英文检索 query。"
        )
        data = self.llm.complete_json(system, user)
        arr = data.get("queries", [])
        return [str(x) for x in arr if str(x).strip()]

    def retrieve(
        self,
        queries: list[str],
        max_papers: int = 16,
        enrich_pdf: bool = True,
        progress_callback=None,
    ) -> list[PaperCandidate]:
        papers: list[PaperCandidate] = []
        seen: set[str] = set()

        def emit(stage: str, detail: str) -> None:
            if progress_callback:
                progress_callback(stage, detail)

        for q in queries:
            emit("retrieve", f"Searching Semantic Scholar: {q[:80]}")
            try:
                found = self.ss.search(q, limit=4)
            except Exception:
                continue
            for p in found:
                key = p.title.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    papers.append(p)
                if len(papers) >= max_papers:
                    break
            if len(papers) >= max_papers:
                break

        if enrich_pdf and papers:
            emit("pdf", f"Downloading and parsing PDFs for up to {min(5, len(papers))} papers")
            papers = self.pdf.enrich(papers, max_pdf=5)
            emit("rag", "Updating local RAG store")
            self.rag.upsert_papers(papers)
            self.mem.remember_papers(papers)
        return papers


def _dedup(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out
