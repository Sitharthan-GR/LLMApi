"""Provider profiles. OpenAI-compatible hosts share one JSON shape; Anthropic does not."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Profile:
    name: str
    kind: str  # "openai" | "anthropic"
    base_url: str
    key_envs: tuple[str, ...]
    default_model: str
    vision_model: str | None = None
    default_key: str | None = None

    def api_key(self) -> str:
        for env in self.key_envs:
            value = (os.getenv(env) or "").strip()
            if value and value not in {"gsk_...", "sk-...", "sk-ant-..."}:
                return value
        if self.default_key:
            return self.default_key
        needed = " or ".join(self.key_envs)
        raise SystemExit(f"provider {self.name}: missing API key ({needed})")


PROFILES = {
    "groq": Profile(
        name="groq",
        kind="openai",
        base_url="https://api.groq.com/openai/v1",
        key_envs=("GROQ_API_KEY", "LLM_API_KEY"),
        default_model="openai/gpt-oss-20b",
        vision_model="qwen/qwen3.8-27b",
    ),
    "openai": Profile(
        name="openai",
        kind="openai",
        base_url="https://api.openai.com/v1",
        key_envs=("OPENAI_API_KEY", "LLM_API_KEY"),
        default_model="gpt-4o-mini",
        vision_model="gpt-4o-mini",
    ),
    "ollama": Profile(
        name="ollama",
        kind="openai",
        base_url="http://127.0.0.1:11434/v1",
        key_envs=("OLLAMA_API_KEY",),
        default_model="llama3.2",
        vision_model=None,
        default_key="ollama",
    ),
    "anthropic": Profile(
        name="anthropic",
        kind="anthropic",
        base_url="https://api.anthropic.com/v1",
        key_envs=("ANTHROPIC_API_KEY",),
        default_model="claude-3-5-haiku-latest",
        vision_model="claude-3-5-haiku-latest",
    ),
}


def load_env(repo_root) -> None:
    from pathlib import Path

    load_dotenv(Path(repo_root) / ".env")


def get_profile(name: str | None) -> Profile:
    chosen = (name or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if chosen not in PROFILES:
        known = ", ".join(PROFILES)
        raise SystemExit(f"unknown provider {chosen!r}. known: {known}")
    return PROFILES[chosen]
