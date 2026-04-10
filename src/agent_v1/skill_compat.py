from __future__ import annotations

from pathlib import Path


def load_markdown_items(root: str | Path, kind: str) -> list[dict]:
    r = Path(root)
    r.mkdir(parents=True, exist_ok=True)
    files = list(r.glob("*.md")) + list(r.glob("*/SKILL.md")) + list(r.glob("*/skill.md"))

    items: list[dict] = []
    for fp in sorted(set(files)):
        try:
            text = fp.read_text(encoding="utf-8")
        except Exception:
            continue
        name = _name_from_path(fp)
        meta, body = _split_frontmatter(text)
        desc = str(meta.get("description", "")).strip() or _extract_description(body)
        item = {
            "name": name,
            "description": desc,
            "content": body.strip(),
            "source": str(fp),
            "skill_dir": str(fp.parent),
            "defaults": {
                "analogy_specialists": int(meta.get("analogy_specialists", 0) or 0),
                "analogy_rounds": int(meta.get("analogy_rounds", 0) or 0),
            },
            "script": str(meta.get("script", "")).strip(),
            "runner": str(meta.get("runner", "")).strip(),
        }
        if kind == "soul":
            item["persona_prompt"] = body.strip()[:4000]
        else:
            item["context_prompt"] = body.strip()[:4000]
        items.append(item)

    if not items:
        _write_examples(r, kind)
        return load_markdown_items(r, kind)
    return _dedup_items(items)


def find_by_name(items: list[dict], name: str) -> dict | None:
    for x in items:
        if str(x.get("name", "")).strip() == name:
            return x
    return None


def _name_from_path(path: Path) -> str:
    if path.name.lower() in {"skill.md"}:
        return path.parent.name
    return path.stem


def _extract_description(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s[:120]
    return ""


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return {}, text
    meta_raw, body = parts
    meta: dict[str, str] = {}
    for line in meta_raw.splitlines()[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip()
    return meta, body


def _write_examples(root: Path, kind: str) -> None:
    if kind == "skill":
        (root / "deep_research.md").write_text(
            """---
description: aggressive cross-domain exploration
analogy_specialists: 6
analogy_rounds: 3
---
# deep_research
Focus on cross-domain transferable mechanisms. Prefer diverse retrieval queries and explicit transfer-risk analysis.
""",
            encoding="utf-8",
        )
        (root / "fast_scan.md").write_text(
            """---
description: fast low-cost scan
analogy_specialists: 3
analogy_rounds: 1
---
# fast_scan
Prioritize low-cost, high-confidence candidate methods and minimal iteration.
""",
            encoding="utf-8",
        )
    else:
        (root / "skeptical_pi.md").write_text(
            """# skeptical_pi
Act like a skeptical reviewer: challenge assumptions, highlight transfer risks, and demand evidence before recommending adoption.
""",
            encoding="utf-8",
        )
        (root / "creative_engineer.md").write_text(
            """# creative_engineer
Act like a creative engineer: propose unconventional but testable routes with concrete implementation steps.
""",
            encoding="utf-8",
        )


def _dedup_items(items: list[dict]) -> list[dict]:
    ranked: dict[str, tuple[int, dict]] = {}
    for it in items:
        name = str(it.get("name", "")).strip()
        if not name:
            continue
        source = str(it.get("source", ""))
        # Prefer directory SKILL.md over flat .md
        score = 2 if source.endswith("/SKILL.md") or source.endswith("\\SKILL.md") else 1
        prev = ranked.get(name)
        if prev is None or score >= prev[0]:
            ranked[name] = (score, it)
    return [v[1] for _, v in sorted(ranked.items(), key=lambda x: x[0])]
