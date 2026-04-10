from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run_skill_script(skill: dict, raw_args: str = "") -> tuple[int, str, str, str]:
    skill_dir = Path(skill.get("skill_dir", "")).resolve()
    script = str(skill.get("script", "")).strip()
    runner = str(skill.get("runner", "")).strip()

    script_path = _resolve_script(skill_dir, script)
    if script_path is None:
        return 1, "", "No executable script found for current skill.", ""

    cmd = _build_cmd(script_path, runner, raw_args)
    p = subprocess.run(cmd, cwd=str(skill_dir), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr, " ".join(shlex.quote(x) for x in cmd)


def _resolve_script(skill_dir: Path, script: str) -> Path | None:
    if script:
        p = (skill_dir / script).resolve()
        if p.is_file() and str(p).startswith(str(skill_dir)):
            return p

    candidates = [
        skill_dir / "scripts" / "run.py",
        skill_dir / "scripts" / "main.py",
        skill_dir / "scripts" / "take_screenshot.py",
        skill_dir / "scripts" / "run.sh",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _build_cmd(script_path: Path, runner: str, raw_args: str) -> list[str]:
    args = shlex.split(raw_args) if raw_args.strip() else []
    ext = script_path.suffix.lower()
    if runner:
        return [runner, str(script_path), *args]
    if ext == ".py":
        return ["python3", str(script_path), *args]
    if ext == ".sh":
        return ["bash", str(script_path), *args]
    if ext in {".ps1"}:
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path), *args]
    return [str(script_path), *args]
