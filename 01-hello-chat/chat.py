#!/usr/bin/env python3
"""Week 1: one Chat Completions call over raw HTTP. No official SDK.

Default host is Groq's free OpenAI-compatible API:
  POST https://api.groq.com/openai/v1/chat/completions

The request is JSON. The response is JSON. Switching providers later is
mostly a different base URL + model name.

Get a free key: https://console.groq.com/keys

Examples:
  python 01-hello-chat/chat.py "What is a token in an LLM API?"
  python 01-hello-chat/chat.py --list-models
  python 01-hello-chat/chat.py "Write a haiku about APIs" --temperature 1.2
  python 01-hello-chat/chat.py "Write a haiku about APIs" --temperature 0.1
  python 01-hello-chat/chat.py "Explain HTTP" --system "Answer in one sentence."
  python 01-hello-chat/chat.py "Explain HTTP" --max-tokens 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Paid OpenAI list prices only. Groq free-tier calls print $0 here.
PRICE_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-5-nano": (0.05, 0.40),
}


def load_settings() -> tuple[str, str, str]:
    load_dotenv(ROOT / ".env")
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not api_key or api_key in {"gsk_...", "sk-..."}:
        sys.exit(
            "Missing API key.\n"
            "This project defaults to Groq's free OpenAI-compatible API.\n"
            "1. Create a key at https://console.groq.com/keys (no paid OpenAI key needed)\n"
            "2. Copy .env.example to .env\n"
            "3. Paste the key as GROQ_API_KEY"
        )
    return base_url, api_key, model


def estimate_cost(base_url: str, model: str, prompt_tokens: int, completion_tokens: int) -> str:
    if "groq.com" in base_url:
        return "$0.000000  (Groq free tier — rate-limited, not billed here)"
    prices = PRICE_PER_MILLION.get(model)
    if not prices:
        return f"no price table entry for {model!r}"
    input_per_m, output_per_m = prices
    usd = (prompt_tokens / 1_000_000) * input_per_m + (
        completion_tokens / 1_000_000
    ) * output_per_m
    return f"${usd:.6f}  (table: ${input_per_m}/1M in, ${output_per_m}/1M out)"


def build_payload(args: argparse.Namespace, model: str) -> dict:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    # Groq and most OpenAI-compatible hosts use max_tokens.
    # OpenAI's newer name is max_completion_tokens (needed for o-series).
    return {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }


def explain_http_error(response: httpx.Response, base_url: str) -> str:
    status = response.status_code
    body = response.text
    try:
        err = response.json().get("error", {})
        if isinstance(err, str):
            message, code = err, None
        else:
            message = err.get("message", body)
            code = err.get("code") or err.get("type")
    except Exception:
        message, code = body, None

    extra = ""
    if status == 401:
        extra = "\nBad or missing key. Groq keys start with gsk_ — https://console.groq.com/keys"
    elif status == 429:
        extra = "\nFree-tier rate limit. Wait a few seconds and retry."
    elif status == 400 and "decommissioned" in str(message).lower():
        extra = "\nThat model id was retired. Run: python 01-hello-chat/chat.py --list-models"
    elif status == 404:
        extra = f"\nCheck LLM_BASE_URL ({base_url}) and the model id."

    return f"HTTP {status} ({code}): {message}{extra}"


def list_models(base_url: str, api_key: str) -> None:
    url = f"{base_url}/models"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if response.is_error:
        sys.exit(explain_http_error(response, base_url))
    data = response.json()
    models = data.get("data") or []
    print(f"GET {url}")
    for item in sorted(models, key=lambda m: m.get("id", "")):
        print(f"  {item.get('id')}")
    if not models:
        print(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", help="User message to send")
    parser.add_argument(
        "--system",
        default="You are a concise tutor for someone learning LLM APIs.",
        help="System message (instructions the model should follow)",
    )
    parser.add_argument("--model", default=None, help="Overrides LLM_MODEL from .env")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="0 is more deterministic, ~1 is more random. Typical range 0–2",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Cap on generated tokens",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full response JSON instead of the pretty summary",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="GET /models and print ids, then exit",
    )
    args = parser.parse_args()

    base_url, api_key, env_model = load_settings()
    model = args.model or env_model

    if args.list_models:
        list_models(base_url, api_key)
        return

    if not args.prompt:
        parser.error("prompt is required unless you pass --list-models")

    payload = build_payload(args, model)
    url = f"{base_url}/chat/completions"

    request_path = HERE / "last-request.json"
    request_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"→ POST {url}")
    print(json.dumps(payload, indent=2))
    print()

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    response_path = HERE / "last-response.json"
    try:
        data = response.json()
        response_path.write_text(json.dumps(data, indent=2) + "\n")
    except json.JSONDecodeError:
        response_path.write_text(response.text)
        sys.exit(f"Non-JSON response ({response.status_code}):\n{response.text}")

    if response.is_error:
        sys.exit(explain_http_error(response, base_url))

    if args.raw:
        print(json.dumps(data, indent=2))
        return

    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    print("← assistant")
    print(message.get("content") or "(empty content)")
    print()
    print("usage")
    print(f"  prompt_tokens:      {prompt_tokens}")
    print(f"  completion_tokens:  {completion_tokens}")
    print(f"  total_tokens:       {total_tokens}")
    print(f"  finish_reason:      {choice.get('finish_reason')}")
    print(f"  estimated_cost:     {estimate_cost(base_url, model, prompt_tokens, completion_tokens)}")
    print()
    print(f"saved {request_path.name} and {response_path.name}")
    print("try next: same prompt with --temperature 0.1 then 1.5, then --max-tokens 16")


if __name__ == "__main__":
    main()
