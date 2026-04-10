from __future__ import annotations

import json

from ..llm import OpenAICompatLLM
from ..schemas import ProblemFrame, RouteProposal, TransferabilityCard


class ResearchOutputComposer:
    def __init__(self, llm: OpenAICompatLLM):
        self.llm = llm

    def run(
        self,
        frame: ProblemFrame,
        cards: list[TransferabilityCard],
        judge_concerns: list[str] | None = None,
        clarification_questions: list[str] | None = None,
    ) -> tuple[list[RouteProposal], list[str]]:
        mini_cards = [
            {
                "title": c.title,
                "why": c.why_relevant,
                "transferable_components": c.transferable_components,
                "risks": c.transfer_risks,
                "effort": c.adaptation_effort,
                "fit": c.fit_for_current_stage,
            }
            for c in cards
        ]
        system = (
            "你是科研路线合成器。输出 JSON: {routes:[{route_name,logic,pros,cons,paper_titles,first_step,success_signal}],"
            "stage_recommendation:string[]}"
        )
        user = (
            f"problem={{task:{frame.task},goal:{frame.current_goal},failure:{frame.observed_failure},"
            f"constraints:{frame.constraints},metric:{frame.target_metric}}}\n"
            f"cards={json.dumps(mini_cards, ensure_ascii=False)}\n"
            f"judge_concerns={json.dumps(judge_concerns or [], ensure_ascii=False)}\n"
            f"clarification_questions={json.dumps(clarification_questions or [], ensure_ascii=False)}\n"
            "按 ROI 和落地成本排序，生成 2-4 条路线。每条路线必须给出 pros/cons、可参考论文标题、第一步实验、以及可观察成功信号。"
        )
        data = self.llm.complete_json(system, user)

        routes: list[RouteProposal] = []
        for row in data.get("routes", []):
            routes.append(
                RouteProposal(
                    route_name=str(row.get("route_name", "")),
                    logic=str(row.get("logic", "")),
                    pros=_to_str_list(row.get("pros")),
                    cons=_to_str_list(row.get("cons")),
                    paper_titles=_to_str_list(row.get("paper_titles")),
                    first_step=str(row.get("first_step", "")),
                    success_signal=str(row.get("success_signal", "")),
                )
            )
        rec = _to_str_list(data.get("stage_recommendation"))
        return [x for x in routes if x.route_name], rec


def _to_str_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return []
