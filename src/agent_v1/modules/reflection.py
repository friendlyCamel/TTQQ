from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass

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
            "frame": _to_jsonable(case.problem_frame),
            "analogy": _to_jsonable(case.analogy),
            "queries": case.retrieval_queries,
            "paper_count": len(case.candidate_papers),
            "card_count": len(case.transferability_cards),
            "route_count": len(case.routes),
            "audit": _to_jsonable(case.judge_audit),
        }
        user = (
            f"case={json.dumps(compact, ensure_ascii=False)}\n"
            "如果结果不可靠或者证据不足，要求重试。"
        )
        return self.llm.complete_json(system, user)


def _to_jsonable(value):
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    return value
