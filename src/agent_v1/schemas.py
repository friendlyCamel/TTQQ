from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemFrame:
    input_type: str
    user_input: str
    task: str = ""
    current_goal: str = ""
    observed_failure: str = ""
    target_metric: str = ""
    constraints: list[str] = field(default_factory=list)
    known_context: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)


@dataclass
class DialogueTurn:
    speaker: str
    content: str


@dataclass
class AnalogyResult:
    problem_structures: list[str] = field(default_factory=list)
    mechanism_candidates: list[str] = field(default_factory=list)
    constraint_filtered_mechanisms: list[str] = field(default_factory=list)
    failure_analogies: list[str] = field(default_factory=list)
    query_families: list[str] = field(default_factory=list)
    dialogue_log: list[DialogueTurn] = field(default_factory=list)


@dataclass
class PaperCandidate:
    title: str
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    year: str = ""
    source: str = ""
    local_pdf_path: str = ""
    extracted_text: str = ""


@dataclass
class TransferabilityCard:
    title: str
    why_relevant: str
    transferable_components: list[str]
    transfer_risks: list[str]
    adaptation_effort: str
    fit_for_current_stage: str
    url: str = ""


@dataclass
class RouteProposal:
    route_name: str
    logic: str
    pros: list[str]
    cons: list[str]
    paper_titles: list[str]


@dataclass
class JudgeAudit:
    concerns: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    blocked: bool = False


@dataclass
class ResearchCase:
    user_input: str
    input_type: str
    mode: str = "problem_solving"
    problem_frame: ProblemFrame | None = None
    analogy: AnalogyResult | None = None
    retrieval_queries: list[str] = field(default_factory=list)
    candidate_papers: list[PaperCandidate] = field(default_factory=list)
    transferability_cards: list[TransferabilityCard] = field(default_factory=list)
    judge_audit: JudgeAudit = field(default_factory=JudgeAudit)
    routes: list[RouteProposal] = field(default_factory=list)
    stage_recommendation: list[str] = field(default_factory=list)
    reflection_notes: list[str] = field(default_factory=list)
    raw_artifacts: dict[str, Any] = field(default_factory=dict)
