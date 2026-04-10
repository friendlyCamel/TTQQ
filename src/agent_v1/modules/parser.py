from __future__ import annotations

from ..llm import OpenAICompatLLM
from ..schemas import ProblemFrame


class ResearchProblemParser:
    def __init__(self, llm: OpenAICompatLLM):
        self.llm = llm

    def run(self, user_input: str, input_type: str, guidance: str = "") -> ProblemFrame:
        system = (
            "你是科研问题解析器。输出严格 JSON。"
            "字段: task,current_goal,observed_failure,target_metric,constraints,known_context,uncertainties。"
        )
        user = (
            f"input_type={input_type}\n"
            f"user_input={user_input}\n"
            f"guidance={guidance}\n"
            "请抽取字段。constraints/known_context/uncertainties 必须是字符串数组。"
        )
        data = self.llm.complete_json(system, user)
        return ProblemFrame(
            input_type=input_type,
            user_input=user_input,
            task=_to_text(data.get("task")),
            current_goal=_to_text(data.get("current_goal")),
            observed_failure=_to_text(data.get("observed_failure")),
            target_metric=_to_text(data.get("target_metric")),
            constraints=_to_str_list(data.get("constraints")),
            known_context=_to_str_list(data.get("known_context")),
            uncertainties=_to_str_list(data.get("uncertainties")),
        )


def _to_str_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return []


def _to_text(v: object) -> str:
    if isinstance(v, list):
        return "；".join(str(x) for x in v if str(x).strip())
    return str(v or "")
