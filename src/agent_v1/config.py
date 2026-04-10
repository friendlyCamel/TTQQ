from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    embedding_model: str = "text-embedding-v3"
    timeout: int = 90


def load_config_from_json(json_path: str | Path) -> LLMConfig:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    api_key = _coalesce_env(
        str(data.get("api_key", "")).strip(),
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
    )
    base_url = _coalesce_env(
        str(data.get("base_url", "")).strip().rstrip("/"),
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "DASHSCOPE_BASE_URL",
        "QWEN_BASE_URL",
    )
    model = _coalesce_env(
        str(data.get("model", "qwen3.5-flash")).strip(),
        "OPENAI_MODEL",
        "LLM_MODEL",
    )
    embedding_model = _coalesce_env(
        str(data.get("embedding_model", "text-embedding-v3")).strip(),
        "OPENAI_EMBEDDING_MODEL",
        "EMBEDDING_MODEL",
    )
    timeout = int(_coalesce_env(str(data.get("timeout", 90)).strip(), "OPENAI_TIMEOUT", "LLM_TIMEOUT"))
    if not api_key or not base_url:
        raise ValueError("llm json 缺少 api_key 或 base_url")
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        embedding_model=embedding_model,
        timeout=timeout,
    )


def load_config_from_env() -> LLMConfig | None:
    api_key = _coalesce_env("", "OPENAI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY")
    base_url = _coalesce_env(
        "",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "DASHSCOPE_BASE_URL",
        "QWEN_BASE_URL",
    ).rstrip("/")
    if not api_key or not base_url:
        return None
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=_coalesce_env("qwen3.5-flash", "OPENAI_MODEL", "LLM_MODEL"),
        embedding_model=_coalesce_env("text-embedding-v3", "OPENAI_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
        timeout=int(_coalesce_env("90", "OPENAI_TIMEOUT", "LLM_TIMEOUT")),
    )


def save_config_to_json(config: LLMConfig, json_path: str | Path) -> Path:
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": "qwen-compatible",
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model": config.model,
        "embedding_model": config.embedding_model,
        "timeout": config.timeout,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def convert_csv_to_json(
    csv_path: str | Path,
    json_path: str | Path,
    model: str | None = None,
    embedding_model: str = "text-embedding-v3",
) -> Path:
    raw: dict[str, str] = {}
    for line in Path(csv_path).read_text(encoding="utf-8").splitlines():
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        raw[parts[0].strip()] = parts[1].strip()

    cfg = {
        "provider": "qwen-compatible",
        "base_url": raw.get("base URL") or raw.get("openAiCompatible", ""),
        "api_key": raw.get("apiKey", ""),
        "model": model or "qwen3.5-flash",
        "embedding_model": embedding_model,
        "timeout": 90,
    }
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _coalesce_env(default: str, *env_names: str) -> str:
    for name in env_names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default
