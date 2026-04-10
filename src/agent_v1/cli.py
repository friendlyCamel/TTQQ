from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from .config import LLMConfig, convert_csv_to_json, load_config_from_env, load_config_from_json, save_config_to_json
from .llm import OpenAICompatLLM
from .session_memory import SessionMemory
from .skill_compat import find_by_name, load_markdown_items
from .skill_runner import run_skill_script


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="httt - Problem-driven research CLI agent")
    sub = p.add_subparsers(dest="cmd", required=False)

    run = sub.add_parser("run", help="single run")
    _add_shared_run_args(run)
    run.add_argument("--query", required=True, help="research question text")

    session = sub.add_parser("session", help="interactive session mode")
    _add_shared_run_args(session)

    init_cfg = sub.add_parser("init-config", help="create json config from csv")
    init_cfg.add_argument("--from-csv", required=True)
    init_cfg.add_argument("--to", default="config/llm.json")
    init_cfg.add_argument("--model", default="qwen3.5-flash")
    init_cfg.add_argument("--embedding-model", default="text-embedding-v3")

    return p


def _add_shared_run_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--config", default="config/llm.json", help="llm config json path")
    sp.add_argument("--model", default="", help="optional model override")
    sp.add_argument("--input-type", default="failure-driven", choices=["idea-driven", "failure-driven", "metric-driven"])
    sp.add_argument("--mode", default="problem_solving", choices=["problem_solving", "field_tracking"])
    sp.add_argument("--project-context", default="", help="project context text or file path")
    sp.add_argument("--max-parse-reviews", type=int, default=2)
    sp.add_argument("--max-reflection-rounds", type=int, default=2)
    sp.add_argument("--analogy-rounds", type=int, default=2)
    sp.add_argument("--analogy-specialists", type=int, default=4)
    sp.add_argument("--interactive-judge", action="store_true")
    sp.add_argument("--max-judge-turns", type=int, default=2)
    sp.add_argument("--workspace", default="workspace")
    sp.add_argument("--project", default="default")
    sp.add_argument("--skills-dir", default="skills")
    sp.add_argument("--souls-dir", default="souls")
    sp.add_argument("--skill", default="")
    sp.add_argument("--soul", default="")


def _read_context(raw: str) -> str:
    if not raw:
        return ""
    p = Path(raw)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return raw


def _ask_judge_questions(questions: list[str]) -> str:
    questions = questions[:3]
    if not questions:
        return ""
    print("\n## Judge Follow-up")
    answers: list[str] = []
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        ans = input("Your answer: ").strip()
        answers.append(f"{q}\nA{i}: {ans}")
    return "\n".join(answers)


def _validate_llm(cfg: LLMConfig) -> bool:
    try:
        llm = OpenAICompatLLM(cfg)
        _ = llm.complete_text("You are a test assistant.", "reply only: ok", temperature=0)
        return True
    except Exception:
        return False


def _prompt_config(json_path: str) -> LLMConfig:
    print("\nNo usable LLM config found. Please input your API settings:")
    base_url = input("base_url (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1): ").strip()
    api_key = getpass.getpass("api_key: ").strip()
    model = input("model [qwen3.5-flash]: ").strip() or "qwen3.5-flash"
    embedding_model = input("embedding_model [text-embedding-v3]: ").strip() or "text-embedding-v3"
    timeout_raw = input("timeout seconds [90]: ").strip() or "90"
    timeout = int(timeout_raw)
    cfg = LLMConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        embedding_model=embedding_model,
        timeout=timeout,
    )
    save_config_to_json(cfg, json_path)
    return cfg


def _load_or_bootstrap_config(config_path: str, model_override: str = "") -> LLMConfig:
    path = Path(config_path)
    cfg: LLMConfig | None = None
    if path.exists():
        try:
            cfg = load_config_from_json(path)
        except Exception:
            cfg = None
    if cfg is None:
        cfg = load_config_from_env()
    while cfg is None:
        cfg = _prompt_config(str(path))
    if model_override:
        cfg.model = model_override
    return cfg


def _compose_context(base_context: str, session_ctx: str, skill: dict | None, soul: dict | None) -> str:
    parts = [base_context.strip(), session_ctx.strip()]
    if skill and skill.get("context_prompt"):
        parts.append(f"skill_prompt: {skill['context_prompt']}")
    if soul and soul.get("persona_prompt"):
        parts.append(f"soul_prompt: {soul['persona_prompt']}")
    return "\n\n".join(x for x in parts if x).strip()


def _project_root(args) -> Path:
    root = Path(args.workspace) / args.project
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_once(app, args, query: str, input_type: str, project_context: str, skill: dict | None):
    defaults = (skill or {}).get("defaults", {}) if skill else {}
    analogy_rounds = int(defaults.get("analogy_rounds", args.analogy_rounds))
    analogy_specialists = int(defaults.get("analogy_specialists", args.analogy_specialists))

    case = app.run(
        user_input=query,
        input_type=input_type,
        mode=args.mode,
        project_context=project_context,
        max_parse_reviews=args.max_parse_reviews,
        max_reflection_rounds=args.max_reflection_rounds,
        analogy_rounds=analogy_rounds,
        analogy_specialists=analogy_specialists,
        progress_callback=_print_progress,
    )

    if args.interactive_judge:
        for _ in range(max(0, args.max_judge_turns)):
            qs = case.judge_audit.clarification_questions[:3]
            if not qs:
                break
            qa_text = _ask_judge_questions(qs)
            if not qa_text.strip():
                break
            project_context = f"{project_context}\n{qa_text}".strip()
            case = app.run(
                user_input=query,
                input_type=input_type,
                mode=args.mode,
                project_context=project_context,
                max_parse_reviews=args.max_parse_reviews,
                max_reflection_rounds=args.max_reflection_rounds,
                analogy_rounds=analogy_rounds,
                analogy_specialists=analogy_specialists,
                progress_callback=_print_progress,
            )
    return case, project_context


def _case_to_brief(case) -> str:
    routes = "; ".join(r.route_name for r in case.routes[:3])
    acts = "; ".join(case.stage_recommendation[:3])
    return f"routes={routes} | actions={acts}".strip()


def _print_case(case) -> None:
    print("\nAssistant:\n")
    f = case.problem_frame
    if f:
        print(f"[Layer1] task={f.task}")
        print(f"[Layer1] goal={f.current_goal}")

    print("\n[Layer2] Routes")
    for i, r in enumerate(case.routes, 1):
        print(f"{i}. {r.route_name}")
        print(f"   logic: {r.logic}")

    print("\n[Layer3] Transferability")
    for i, c in enumerate(case.transferability_cards[:6], 1):
        print(f"{i}. {c.title}")
        print(f"   why_relevant: {c.why_relevant}")
        print(f"   transferable_components: {', '.join(c.transferable_components)}")
        print(f"   transfer_risks: {', '.join(c.transfer_risks)}")
        print(f"   adaptation_effort: {c.adaptation_effort}")
        print(f"   fit_for_current_stage: {c.fit_for_current_stage}")

    if case.judge_audit.concerns:
        print("\n[Judge Concerns]")
        for x in case.judge_audit.concerns:
            print(f"- {x}")

    print("\n[Layer4] Action")
    for i, x in enumerate(case.stage_recommendation, 1):
        print(f"{i}. {x}")


def _print_progress(stage: str, detail: str) -> None:
    print(f"[{stage}] {detail}", flush=True)


def _session_loop(app, args, llm: OpenAICompatLLM, project_root: Path) -> None:
    _print_welcome(project_root, llm.config.model)

    input_type = args.input_type
    base_context = _read_context(args.project_context)
    skills = load_markdown_items(args.skills_dir, kind="skill")
    souls = load_markdown_items(args.souls_dir, kind="soul")
    if args.skill:
        active_skill = find_by_name(skills, args.skill)
    else:
        active_skill = find_by_name(skills, "pdf_hub")
    if args.soul:
        active_soul = find_by_name(souls, args.soul)
    else:
        active_soul = find_by_name(souls, "research_literature_core")
    mem = SessionMemory.load(llm, project_root / "memory" / "session_state.json")

    while True:
        try:
            msg = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break

        if not msg:
            continue
        if msg in {"/exit", "/quit"}:
            mem.save(project_root / "memory" / "session_state.json")
            print("bye")
            break
        if msg == "/help":
            print("Usage:")
            print("/help")
            print("/type <idea-driven|failure-driven|metric-driven>")
            print("/typ <idea-driven|failure-driven|metric-driven>  # alias")
            print("/model <model-name>                               # switch model and save config")
            print("/context <text>")
            print("/skills")
            print("/skill <name|off>")
            print("/skillrun <args>")
            print("/souls")
            print("/soul <name|off>")
            print("/status")
            print("/exit")
            continue
        if msg.startswith("/skillrun"):
            if not active_skill:
                print("No active skill. Set one via /skill <name> first.")
                continue
            raw_args = msg.replace("/skillrun", "", 1).strip()
            code, out, err, cmd = run_skill_script(active_skill, raw_args)
            print(f"[skillrun] cmd: {cmd}")
            print(f"[skillrun] exit: {code}")
            if out.strip():
                print("[skillrun][stdout]")
                print(out.strip()[:3000])
            if err.strip():
                print("[skillrun][stderr]")
                print(err.strip()[:3000])
            continue
        if msg == "/skills":
            for x in skills:
                print(f"- {x.get('name')}: {x.get('description', '')}")
            continue
        if msg == "/souls":
            for x in souls:
                print(f"- {x.get('name')}: {x.get('description', '')}")
            continue
        if msg.startswith("/skill "):
            name = msg.replace("/skill ", "", 1).strip()
            if name == "off":
                active_skill = None
                print("skill off")
            else:
                sel = find_by_name(skills, name)
                if sel:
                    active_skill = sel
                    print(f"skill set: {name}")
                else:
                    print("unknown skill")
            continue
        if msg.startswith("/soul "):
            name = msg.replace("/soul ", "", 1).strip()
            if name == "off":
                active_soul = None
                print("soul off")
            else:
                sel = find_by_name(souls, name)
                if sel:
                    active_soul = sel
                    print(f"soul set: {name}")
                else:
                    print("unknown soul")
            continue
        if msg == "/status":
            print(f"input_type={input_type}")
            print(f"model={llm.config.model}")
            print(f"skill={active_skill.get('name') if active_skill else 'none'}")
            print(f"soul={active_soul.get('name') if active_soul else 'none'}")
            print(f"session_summary={'yes' if mem.summary else 'no'}")
            continue
        if msg.startswith("/type "):
            t = msg.replace("/type ", "", 1).strip()
            if t in {"idea-driven", "failure-driven", "metric-driven"}:
                input_type = t
                print(f"input_type set to {input_type}")
            else:
                print("invalid input_type")
            continue
        if msg.startswith("/typ "):
            t = msg.replace("/typ ", "", 1).strip()
            if t in {"idea-driven", "failure-driven", "metric-driven"}:
                input_type = t
                print(f"input_type set to {input_type}")
            else:
                print("invalid input_type")
            continue
        if msg.startswith("/model "):
            model = msg.replace("/model ", "", 1).strip()
            if not model:
                print("usage: /model <model-name>")
                continue
            llm.config.model = model
            app.llm.config.model = model
            save_config_to_json(llm.config, args.config)
            print(f"model set to {model} (saved to {args.config})")
            continue
        if msg.startswith("/context "):
            base_context = msg.replace("/context ", "", 1).strip()
            print("project_context updated")
            continue

        merged_context = _compose_context(base_context, mem.context_text(), active_skill, active_soul)
        try:
            case, _ = _run_once(app, args, msg, input_type, merged_context, active_skill)
        except Exception as exc:
            print(f"[error] execution failed: {exc}")
            continue
        _print_case(case)

        mem.add_turn("user", msg)
        mem.add_turn("assistant", _case_to_brief(case))
        mem.save(project_root / "memory" / "session_state.json")


def _print_welcome(project_root: Path, model: str) -> None:
    cup = [
        "      ( (",
        "       ) )",
        "    ........",
        "    |      |]",
        "    \\      /",
        "     `----'",
    ]
    logo = [
        ">>>>>    >>>>>      >  >      >  >          ",
        "  >        >       >    >    >    >         ",
        "  >        >      >      >  >      >        ",
        "  >        >       >   > >   >   > >         ",
        "  >        >        >  > >    >  > >        ",
    ]
    print("\n".join(cup))
    print()
    print("HTTT: Problem-driven, analogy-enabled literature agent")
    print("\n".join(logo))
    print(f"\nproject={project_root} | model={model}")
    print("Type /help for command usage.\n")


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]

    if not argv:
        argv = ["session"]
    elif argv[0] not in {"run", "session", "init-config", "-h", "--help"}:
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    cmd = args.cmd or "session"

    if cmd == "init-config":
        out = convert_csv_to_json(
            args.from_csv,
            args.to,
            model=args.model,
            embedding_model=args.embedding_model,
        )
        print(f"config written: {out}")
        return

    cfg = _load_or_bootstrap_config(args.config, model_override=getattr(args, "model", ""))
    llm = OpenAICompatLLM(cfg)

    from .orchestrator import ResearchOrchestrator

    project_root = _project_root(args)

    app = ResearchOrchestrator(cfg, project_root=project_root)
    print(f"[startup] project={project_root} model={cfg.model}", flush=True)

    if cmd == "session":
        _session_loop(app, args, llm, project_root)
        return

    skills = load_markdown_items(args.skills_dir, kind="skill")
    souls = load_markdown_items(args.souls_dir, kind="soul")
    if args.skill:
        active_skill = find_by_name(skills, args.skill)
    else:
        active_skill = find_by_name(skills, "pdf_hub")
    if args.soul:
        active_soul = find_by_name(souls, args.soul)
    else:
        active_soul = find_by_name(souls, "research_literature_core")

    base_context = _read_context(args.project_context)
    merged_context = _compose_context(base_context, "", active_skill, active_soul)
    try:
        case, _ = _run_once(app, args, args.query, args.input_type, merged_context, active_skill)
    except Exception as exc:
        print(f"[error] execution failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    _print_case(case)


if __name__ == "__main__":
    main()
