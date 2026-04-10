from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .llm import OpenAICompatLLM


@dataclass
class SessionMemory:
    llm: OpenAICompatLLM
    summary: str = ""
    recent_turns: list[dict[str, str]] = field(default_factory=list)
    max_recent: int = 8

    def add_turn(self, role: str, content: str) -> None:
        self.recent_turns.append({"role": role, "content": content.strip()})
        if len(self.recent_turns) > self.max_recent:
            self._compress_old_turns()

    def context_text(self) -> str:
        parts: list[str] = []
        if self.summary:
            parts.append(f"session_summary: {self.summary}")
        if self.recent_turns:
            turns = "\n".join(f"{t['role']}: {t['content']}" for t in self.recent_turns)
            parts.append(f"recent_turns:\n{turns}")
        return "\n\n".join(parts).strip()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"summary": self.summary, "recent_turns": self.recent_turns}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, llm: OpenAICompatLLM, path: str | Path) -> "SessionMemory":
        p = Path(path)
        if not p.exists():
            return cls(llm=llm)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                llm=llm,
                summary=str(data.get("summary", "")),
                recent_turns=list(data.get("recent_turns", []))[-8:],
            )
        except Exception:
            return cls(llm=llm)

    def _compress_old_turns(self) -> None:
        old = self.recent_turns[:-4]
        self.recent_turns = self.recent_turns[-4:]
        if not old:
            return
        old_text = "\n".join(f"{t['role']}: {t['content']}" for t in old)
        prompt = (
            f"existing_summary={self.summary}\n"
            f"new_old_turns={old_text}\n"
            "请把会话记忆压缩为 5-8 行要点，保留：用户目标、约束、已经否定的路线、关键偏好。"
        )
        try:
            self.summary = self.llm.complete_text("你是会话记忆压缩器。", prompt, temperature=0.1).strip()[:1600]
        except Exception:
            merged = (self.summary + "\n" + old_text).strip()
            self.summary = merged[-1600:]
