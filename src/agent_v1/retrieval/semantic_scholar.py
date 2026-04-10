from __future__ import annotations

import requests

from ..schemas import PaperCandidate


class SemanticScholarRetriever:
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: str, limit: int = 5) -> list[PaperCandidate]:
        resp = requests.get(
            self.BASE,
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,url,openAccessPdf",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        out: list[PaperCandidate] = []
        for row in data:
            pdf = row.get("openAccessPdf") or {}
            out.append(
                PaperCandidate(
                    title=row.get("title", ""),
                    abstract=row.get("abstract", "") or "",
                    url=row.get("url", "") or "",
                    pdf_url=pdf.get("url", "") or "",
                    year=str(row.get("year", "") or ""),
                    source="semantic_scholar",
                )
            )
        return [x for x in out if x.title]
