from __future__ import annotations

import json

from ..llm import OpenAICompatLLM
from ..schemas import ResearchCase


class SelfReflectionEngine:
    def __init__(self, llm: OpenAICompatLLM):
        self.llm = llm

    def review_case(self, case: ResearchCase) -> dict:
        system = (
            "你是系统自反思模块。输出JSON: "
            "{need_retry:boolean,score:int,weak_points:string[],retry_advice:string}"
        )
        compact = {
            "input": case.user_input,
            "frame": case.problem_frame.__dict__ if case.problem_frame else {},
            "analogy": case.analogy.__dict__ if case.analogy else {},
            "queries": case.retrieval_queries,
            "paper_count": len(case.candidate_papers),
            "card_count": len(case.transferability_cards),
            "route_count": len(case.routes),
            "audit": case.judge_audit.__dict__,
        }
        user = (
            f"case={json.dumps(compact, ensure_ascii=False)}\n"
            "如果结果不可靠或者证据不足，要求重试。"
        )
        return self.llm.complete_json(system, user)
