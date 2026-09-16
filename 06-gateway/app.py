#!/usr/bin/env python3
"""OpenAI-compatible gateway: POST /v1/chat/completions

  uvicorn --app-dir 06-gateway app:app --port 8000
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "05-multi-provider"))
sys.path.insert(0, str(HERE))

from client import ChatClient
from logutil import append_log
from profiles import get_profile, load_env
from retry import RetryableError, with_retries

load_env(ROOT)
LOG = HERE / "gateway.jsonl"
app = FastAPI(title="LLMApi gateway", version="0.1.0")


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


def _complete(body: ChatRequest) -> dict:
    profile = get_profile(None)
    client = ChatClient(profile, timeout=60.0)

    def call() -> dict:
        try:
            return client.complete(
                body.messages,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except RuntimeError as exc:
            text = str(exc)
            if "HTTP 429" in text:
                raise RetryableError(text, status=429) from exc
            if "HTTP 5" in text:
                raise RetryableError(text, status=500) from exc
            raise

    return with_retries(call)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest) -> dict:
    if body.stream:
        raise HTTPException(status_code=400, detail="stream=false only in this teaching gateway")
    t0 = time.perf_counter()
    try:
        data = _complete(body)
    except RuntimeError as exc:
        append_log(
            LOG,
            {"ok": False, "error": str(exc), "model": body.model, "message_count": len(body.messages)},
        )
        text = str(exc)
        status = 502
        if "HTTP 401" in text:
            status = 401
        elif "HTTP 400" in text:
            status = 400
        raise HTTPException(status_code=status, detail=text) from exc

    ms = round((time.perf_counter() - t0) * 1000)
    usage = data.get("usage") or {}
    append_log(
        LOG,
        {
            "ok": True,
            "provider": get_profile(None).name,
            "model": data.get("model") or body.model,
            "latency_ms": ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost_usd": 0.0,
            "message_count": len(body.messages),
        },
    )
    return data
