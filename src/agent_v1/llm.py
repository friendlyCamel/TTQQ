from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Any

import requests

from .config import LLMConfig


class OpenAICompatLLM:
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        text = self.complete_text(system_prompt, user_prompt)
        return _extract_json(text)

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3) -> str:
        url = f"{self.config.base_url}/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=self.config.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM request failed for {url}: {exc}") from exc
        resp.raise_for_status()
        payload = resp.json()
        return payload["choices"][0]["message"]["content"]

    def embed(self, text: str, dim: int = 256) -> list[float]:
        url = f"{self.config.base_url}/embeddings"
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.embedding_model,
                    "input": text[:8000],
                },
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            emb = payload["data"][0]["embedding"]
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb]
        except Exception:
            pass
        return _hash_embedding(text, dim=dim)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型返回不是 JSON: {text[:200]}")
    return json.loads(text[start : end + 1])


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(sha256(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
