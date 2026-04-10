from __future__ import annotations

import json

from ..llm import OpenAICompatLLM
from ..schemas import ProblemFrame


class OrchestratorQualityGate:
    def __init__(self, llm: OpenAICompatLLM):
        self.llm = llm

    def review_parser_and_abstraction(self, frame: ProblemFrame, structures: list[str]) -> dict:
        system = (
            "你是主控agent的质量审查器。输出JSON: "
            "{pass:boolean,score:int,issues:string[],improve_instructions:string}"
        )
        user = (
            f"frame={json.dumps(frame.__dict__, ensure_ascii=False)}\n"
            f"problem_structures={json.dumps(structures, ensure_ascii=False)}\n"
            "判断是否足够进入下一步联想和检索。"
        )
        return self.llm.complete_json(system, user)
