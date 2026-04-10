from __future__ import annotations

from ..llm import OpenAICompatLLM
from ..schemas import ProblemFrame


class ProblemAbstractionEngine:
    def __init__(self, llm: OpenAICompatLLM):
        self.llm = llm

    def run(self, frame: ProblemFrame, guidance: str = "") -> list[str]:
        system = "你是问题结构抽象器。输出 JSON: {problem_structures: string[]}"
        user = (
            f"task={frame.task}\n"
            f"goal={frame.current_goal}\n"
            f"failure={frame.observed_failure}\n"
            f"constraints={frame.constraints}\n"
            f"guidance={guidance}\n"
            "给出 4-8 个通用问题结构。"
        )
        data = self.llm.complete_json(system, user)
        arr = data.get("problem_structures", [])
        return [str(x) for x in arr if str(x).strip()]
