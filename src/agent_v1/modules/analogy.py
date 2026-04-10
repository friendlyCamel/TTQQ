from __future__ import annotations

import json

from ..llm import OpenAICompatLLM
from ..schemas import AnalogyResult, DialogueTurn, PaperCandidate, ProblemFrame


class AnalogyGenerator:
    """Moderator + specialist mini-agents with free-form dialogue."""

    def __init__(self, llm: OpenAICompatLLM):
        self.llm = llm
        self.specialist_pool = [
            ("problem-structure-agent", "专注问题结构抽象与等价问题识别"),
            ("mechanism-agent", "专注方法机制联想与机制可迁移性"),
            ("constraint-agent", "专注现实约束筛选与落地成本"),
            ("failure-agent", "专注失败模式类比与风险前瞻"),
            ("retrieval-strategy-agent", "专注检索策略设计与query拆解"),
            ("classical-method-agent", "专注经典方法迁移与低算力可行性"),
            ("novelty-agent", "专注创新性、差异性与潜在突破点"),
            ("skeptic-agent", "专注反驳、漏洞、不可迁移风险"),
        ]

    def run(
        self,
        frame: ProblemFrame,
        problem_structures: list[str],
        evidence_papers: list[PaperCandidate] | None = None,
        rounds: int = 2,
        specialist_count: int = 4,
    ) -> AnalogyResult:
        transcript: list[DialogueTurn] = []
        evidence_text = _build_evidence_text(evidence_papers or [])
        specialists = self._pick_specialists(specialist_count)

        for _ in range(rounds):
            for name, role_desc in specialists:
                msg = self._speak(name, role_desc, frame, problem_structures, transcript, evidence_text)
                transcript.append(DialogueTurn(speaker=name, content=msg))

        summary = self._moderate(frame, problem_structures, transcript, evidence_text)
        return AnalogyResult(
            problem_structures=problem_structures,
            mechanism_candidates=_to_str_list(summary.get("mechanism_candidates")),
            constraint_filtered_mechanisms=_to_str_list(summary.get("constraint_filtered_mechanisms")),
            failure_analogies=_to_str_list(summary.get("failure_analogies")),
            query_families=_to_str_list(summary.get("query_families")),
            dialogue_log=transcript,
        )

    def _speak(
        self,
        name: str,
        role_desc: str,
        frame: ProblemFrame,
        problem_structures: list[str],
        transcript: list[DialogueTurn],
        evidence_text: str,
    ) -> str:
        system = f"你是{name}，{role_desc}。与其他agent自由对话，给出精炼观点，不要JSON。"
        history = "\n".join(f"[{t.speaker}] {t.content}" for t in transcript[-8:])
        user = (
            f"problem={json.dumps(frame.__dict__, ensure_ascii=False)}\n"
            f"problem_structures={json.dumps(problem_structures, ensure_ascii=False)}\n"
            f"evidence={evidence_text}\n"
            f"recent_dialogue={history}\n"
            "请提出新观点、反驳点或补充检索需求。"
        )
        return self.llm.complete_text(system, user, temperature=0.6).strip()

    def _moderate(
        self,
        frame: ProblemFrame,
        problem_structures: list[str],
        transcript: list[DialogueTurn],
        evidence_text: str,
    ) -> dict:
        system = (
            "你是analogy主持人。汇总多agent讨论，输出JSON:"
            "{mechanism_candidates:string[],constraint_filtered_mechanisms:string[],"
            "failure_analogies:string[],query_families:string[]}"
        )
        dialogue = "\n".join(f"[{t.speaker}] {t.content}" for t in transcript)
        user = (
            f"problem={json.dumps(frame.__dict__, ensure_ascii=False)}\n"
            f"problem_structures={json.dumps(problem_structures, ensure_ascii=False)}\n"
            f"evidence={evidence_text}\n"
            f"dialogue={dialogue}\n"
            "输出可用于下一步retriever的query_families（英文）。"
        )
        return self.llm.complete_json(system, user)

    def _pick_specialists(self, count: int) -> list[tuple[str, str]]:
        count = max(1, count)
        if count <= len(self.specialist_pool):
            return self.specialist_pool[:count]
        out = list(self.specialist_pool)
        for i in range(count - len(self.specialist_pool)):
            out.append((f"custom-agent-{i+1}", "专注补充跨学科联想与反例验证"))
        return out


def _to_str_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return []


def _build_evidence_text(papers: list[PaperCandidate]) -> str:
    if not papers:
        return ""
    lines: list[str] = []
    for p in papers[:6]:
        lines.append(f"title={p.title}; abstract={(p.abstract or '')[:240]}")
    return "\n".join(lines)
