from __future__ import annotations

import json
from pathlib import Path


DEFAULT_SKILLS = {
    "skills": [
        {
            "name": "deep_research",
            "description": "更激进的跨领域检索与联想",
            "context_prompt": "优先探索跨学科可迁移机制，不局限同领域。",
            "defaults": {"analogy_specialists": 6, "analogy_rounds": 3},
        },
        {
            "name": "fast_scan",
            "description": "低成本快速扫描",
            "context_prompt": "优先低成本、高置信候选，减少重检索。",
            "defaults": {"analogy_specialists": 3, "analogy_rounds": 1},
        },
    ]
}

DEFAULT_SOULS = {
    "souls": [
        {
            "name": "skeptical_pi",
            "description": "悲观严谨，强风险控制",
            "persona_prompt": "采取审稿人视角，优先指出不可迁移风险和隐含假设。",
        },
        {
            "name": "creative_engineer",
            "description": "工程创新，偏探索",
            "persona_prompt": "在保证可行性前提下，优先提出非常规但可验证的方案。",
        },
    ]
}


def load_skills(path: str | Path) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(DEFAULT_SKILLS, ensure_ascii=False, indent=2), encoding="utf-8")
    return DEFAULT_SKILLS


def load_souls(path: str | Path) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(DEFAULT_SOULS, ensure_ascii=False, indent=2), encoding="utf-8")
    return DEFAULT_SOULS


def find_by_name(items: list[dict], name: str) -> dict | None:
    for x in items:
        if str(x.get("name", "")).strip() == name:
            return x
    return None
