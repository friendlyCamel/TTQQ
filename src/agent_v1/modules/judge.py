from __future__ import annotations

import json

from ..llm import OpenAICompatLLM
from ..schemas import JudgeAudit, PaperCandidate, ProblemFrame, TransferabilityCard


class TransferabilityJudge:
    def __init__(self, llm: OpenAICompatLLM):
        self.llm = llm

    def run(
        self,
        frame: ProblemFrame,
        papers: list[PaperCandidate],
        project_context: str = "",
        limit: int = 8,
    ) -> list[TransferabilityCard]:
        picked = papers[:limit]
        if not picked:
            return []

        compact = [
            {
                "title": p.title,
                "abstract": (p.abstract or "")[:1200],
                "pdf_excerpt": (p.extracted_text or "")[:1200],
                "url": p.url,
                "year": p.year,
                "source": p.source,
            }
            for p in picked
        ]
        system = (
            "你是论文迁移性评估器。必须输出JSON: {cards:[{title,why_relevant,"
            "transferable_components,transfer_risks,adaptation_effort,fit_for_current_stage,url}]}。"
            "字段必须完整。"
        )
        user = (
            f"problem={{task:{frame.task},goal:{frame.current_goal},failure:{frame.observed_failure},"
            f"constraints:{frame.constraints},target_metric:{frame.target_metric}}}\n"
            f"project_context={project_context}\n"
            f"papers={json.dumps(compact, ensure_ascii=False)}\n"
            "给出严格保守评估。"
        )
        data = self.llm.complete_json(system, user)
        out: list[TransferabilityCard] = []
        for row in data.get("cards", []):
            out.append(
                TransferabilityCard(
                    title=str(row.get("title", "")),
                    why_relevant=str(row.get("why_relevant", "")),
                    transferable_components=_to_str_list(row.get("transferable_components")),
                    transfer_risks=_to_str_list(row.get("transfer_risks")),
                    adaptation_effort=str(row.get("adaptation_effort", "")),
                    fit_for_current_stage=str(row.get("fit_for_current_stage", "")),
                    url=str(row.get("url", "")),
                )
            )
        return [x for x in out if x.title and x.why_relevant]

    def skeptical_audit(
        self,
        frame: ProblemFrame,
        cards: list[TransferabilityCard],
        project_context: str = "",
    ) -> JudgeAudit:
        system = (
            "你是悲观审稿人。输出JSON: {concerns:string[],clarification_questions:string[],blocked:boolean}"
        )
        user = (
            f"problem={json.dumps(frame.__dict__, ensure_ascii=False)}\n"
            f"cards={json.dumps([c.__dict__ for c in cards], ensure_ascii=False)}\n"
            f"project_context={project_context}\n"
            "当信息不足时，必须提出澄清问题；澄清问题最多3个。"
        )
        data = self.llm.complete_json(system, user)
        return JudgeAudit(
            concerns=_to_str_list(data.get("concerns")),
            clarification_questions=_to_str_list(data.get("clarification_questions"))[:3],
            blocked=bool(data.get("blocked", False)),
        )


def _to_str_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return []
