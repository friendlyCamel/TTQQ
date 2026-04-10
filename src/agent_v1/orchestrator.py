from __future__ import annotations

from pathlib import Path

from .config import LLMConfig
from .llm import OpenAICompatLLM
from .memory import LongTermMemory
from .modules.abstraction import ProblemAbstractionEngine
from .modules.analogy import AnalogyGenerator
from .modules.composer import ResearchOutputComposer
from .modules.judge import TransferabilityJudge
from .modules.parser import ResearchProblemParser
from .modules.quality_gate import OrchestratorQualityGate
from .modules.reflection import SelfReflectionEngine
from .modules.retriever import CrossDomainRetriever
from .schemas import ResearchCase


class ResearchOrchestrator:
    def __init__(self, llm_config: LLMConfig, project_root: str | Path = "data/default"):
        root = Path(project_root)
        self.llm = OpenAICompatLLM(llm_config)
        self.parser = ResearchProblemParser(self.llm)
        self.abstraction = ProblemAbstractionEngine(self.llm)
        self.analogy = AnalogyGenerator(self.llm)
        self.retriever = CrossDomainRetriever(self.llm, root)
        self.judge = TransferabilityJudge(self.llm)
        self.composer = ResearchOutputComposer(self.llm)
        self.quality_gate = OrchestratorQualityGate(self.llm)
        self.reflection = SelfReflectionEngine(self.llm)
        self.memory = LongTermMemory(root / "memory")

    def run(
        self,
        user_input: str,
        input_type: str,
        mode: str = "problem_solving",
        project_context: str = "",
        max_parse_reviews: int = 2,
        max_reflection_rounds: int = 2,
        analogy_rounds: int = 2,
        analogy_specialists: int = 4,
        progress_callback=None,
    ) -> ResearchCase:
        def emit(stage: str, detail: str) -> None:
            if progress_callback:
                progress_callback(stage, detail)

        case = ResearchCase(user_input=user_input, input_type=input_type, mode=mode)

        guidance = ""
        frame = None
        structures: list[str] = []
        for _ in range(max_parse_reviews + 1):
            emit("parse", "Parsing research problem")
            frame = self.parser.run(user_input, input_type, guidance=guidance)
            emit("abstract", "Building reusable problem structures")
            structures = self.abstraction.run(frame, guidance=guidance)
            emit("quality_gate", "Reviewing parse quality")
            review = self.quality_gate.review_parser_and_abstraction(frame, structures)
            if bool(review.get("pass", False)):
                break
            guidance = str(review.get("improve_instructions", ""))
            issues = review.get("issues", [])
            case.reflection_notes.append(f"parse_review_issues={issues}")

        if frame is None:
            raise RuntimeError("parser failed")

        case.problem_frame = frame

        emit("memory", "Loading related memory notes")
        memory_hints = self.memory.retrieve_related_notes(user_input, top_k=4)
        case.raw_artifacts["memory_hints"] = memory_hints

        emit("analogy", "Running initial analogy discussion")
        analogy = self.analogy.run(
            frame,
            structures,
            evidence_papers=[],
            rounds=analogy_rounds,
            specialist_count=analogy_specialists,
        )
        case.analogy = analogy
        case.retrieval_queries = self.retriever.build_queries(frame, analogy)

        for r in range(max_reflection_rounds + 1):
            emit("retrieve", f"Retrieving papers (round {r + 1})")
            papers = self.retriever.retrieve(case.retrieval_queries, enrich_pdf=True, progress_callback=progress_callback)
            case.candidate_papers = papers

            emit("analogy", f"Refreshing analogy discussion with evidence (round {r + 1})")
            analogy = self.analogy.run(
                frame,
                structures,
                evidence_papers=papers,
                rounds=analogy_rounds,
                specialist_count=analogy_specialists,
            )
            case.analogy = analogy
            case.retrieval_queries = self.retriever.build_queries(frame, analogy)

            emit("judge", f"Assessing transferability (round {r + 1})")
            cards = self.judge.run(frame, papers, project_context=project_context)
            case.transferability_cards = cards
            case.judge_audit = self.judge.skeptical_audit(frame, cards, project_context=project_context)

            emit("compose", f"Composing routes and recommendations (round {r + 1})")
            routes, rec = self.composer.run(
                frame,
                cards,
                judge_concerns=case.judge_audit.concerns,
                clarification_questions=case.judge_audit.clarification_questions,
            )
            case.routes = routes
            case.stage_recommendation = rec

            emit("reflection", f"Running self-review (round {r + 1})")
            reflection = self.reflection.review_case(case)
            note = f"reflection_round={r},score={reflection.get('score')},weak={reflection.get('weak_points')}"
            case.reflection_notes.append(note)
            need_retry = bool(reflection.get("need_retry", False))
            if not need_retry:
                break

            advice = str(reflection.get("retry_advice", ""))
            if advice:
                case.retrieval_queries = [*case.retrieval_queries, advice]

        emit("memory", "Saving case memory")
        self.memory.remember_case(case)
        self.memory.remember_analogy_dialogue(case)
        return case
