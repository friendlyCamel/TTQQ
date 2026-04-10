from __future__ import annotations

import re
from pathlib import Path

import requests

from ..schemas import PaperCandidate


class PDFPipeline:
    def __init__(self, root: str | Path = "data/papers"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def enrich(self, papers: list[PaperCandidate], max_pdf: int = 5) -> list[PaperCandidate]:
        done = 0
        for p in papers:
            if done >= max_pdf:
                break
            url = p.pdf_url or ""
            if not url:
                continue
            try:
                local = self._download(url, p.title)
                text = self._extract_text(local)
                p.local_pdf_path = str(local)
                p.extracted_text = text[:6000]
                done += 1
            except Exception:
                continue
        return papers

    def _download(self, url: str, title: str) -> Path:
        name = _safe_name(title) + ".pdf"
        target = self.root / name
        if not target.exists():
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        return target

    def _extract_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception:
            return ""
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages[:8]:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)


def _safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:100] if s else "paper"
