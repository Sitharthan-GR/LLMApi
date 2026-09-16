#!/usr/bin/env python3
"""Week 2: streaming multi-turn chat over raw HTTP. No official SDK.

Week 1 sent one messages array and waited for the full JSON.
Week 2 does two new things:

1. stream: true — tokens arrive as SSE lines (`data: {...}`) instead of one blob.
2. History — after each reply we append the assistant message and send the
   whole list next turn. That list is the conversation. The API has no memory.

Commands inside the REPL: /help /reset /system /history /exit
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
DEFAULT_SYSTEM = "You are a concise tutor for someone learning LLM APIs."
DEFAULT_BUDGET = 4000

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
            "Missing API key. Put GROQ_API_KEY in the repo-root .env\n"
            "https://console.groq.com/keys"
        )
    return base_url, api_key, model


def approx_tokens(messages: list[dict]) -> int:
    """Rough local count (chars/4). The API usage field is the real number."""
    text = "".join(str(m.get("content") or "") for m in messages)
    return max(1, len(text) // 4) if text else 0


def estimate_cost(base_url: str, model: str, prompt_tokens: int, completion_tokens: int) -> str:
    if "groq.com" in base_url:
        return "$0.000000  (Groq free tier)"
    prices = PRICE_PER_MILLION.get(model)
    if not prices:
        return f"no price table for {model!r}"
    input_per_m, output_per_m = prices
    usd = (prompt_tokens / 1_000_000) * input_per_m + (
        completion_tokens / 1_000_000
    ) * output_per_m
    return f"${usd:.6f}"


def trim_history(messages: list[dict], budget: int) -> tuple[list[dict], int]:
    """Drop oldest non-system messages until the local estimate fits the budget."""
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    dropped = 0
    while rest and approx_tokens(system + rest) > budget:
        rest.pop(0)
        dropped += 1
        if rest and rest[0]["role"] == "assistant":
            rest.pop(0)
            dropped += 1
    return system + rest, dropped


def iter_sse_json(response: httpx.Response):
    """Parse Server-Sent Events: lines of `data: <json>` ending with `data: [DONE]`."""
    for line in response.iter_lines():
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def stream_chat(
    client: httpx.Client,
    url: str,
    api_key: str,
    payload: dict,
) -> tuple[str, dict, str | None, list[dict]]:
    """POST with stream=true. Print assistant tokens as they arrive. Return full text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    pieces: list[str] = []
    chunks: list[dict] = []
    usage: dict = {}
    finish_reason = None

    with client.stream("POST", url, headers=headers, json=payload, timeout=90.0) as response:
        if response.is_error:
            body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status_code}: {body}")

        sys.stdout.write("assistant> ")
        sys.stdout.flush()

        for chunk in iter_sse_json(response):
            chunks.append(chunk)
            if chunk.get("usage"):
                usage = chunk["usage"]
            x_groq = chunk.get("x_groq") or {}
            if isinstance(x_groq, dict) and x_groq.get("usage"):
                usage = x_groq["usage"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            token = delta.get("content") or ""
            if token:
                pieces.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(pieces), usage, finish_reason, chunks


def print_help() -> None:
    print(
        """
commands
  /help              this list
  /reset             drop user/assistant turns; keep the system prompt
  /system <text>     replace the system prompt (history stays)
  /history           print the messages array we will send next
  /exit              quit (also Ctrl-D / Ctrl-C)
""".strip()
    )


def print_history(messages: list[dict]) -> None:
    print(f"{len(messages)} messages  (~{approx_tokens(messages)} tokens local estimate)")
    for i, msg in enumerate(messages):
        content = (msg.get("content") or "").replace("\n", " ")
        if len(content) > 96:
            content = content[:93] + "..."
        print(f"  [{i}] {msg['role']}: {content}")


def handle_command(line: str, messages: list[dict], system: str) -> tuple[list[dict], str, bool]:
    """Returns (messages, system, should_quit)."""
    if line in {"/exit", "/quit"}:
        return messages, system, True
    if line == "/help":
        print_help()
        return messages, system, False
    if line == "/reset":
        messages = [m for m in messages if m["role"] == "system"]
        print("history cleared (system prompt kept)")
        return messages, system, False
    if line == "/history":
        print_history(messages)
        return messages, system, False
    if line.startswith("/system"):
        new_system = line[len("/system") :].strip()
        if not new_system:
            print("usage: /system You are a terse tutor.")
            return messages, system, False
        system = new_system
        messages = [m for m in messages if m["role"] != "system"]
        messages.insert(0, {"role": "system", "content": system})
        print("system prompt updated")
        return messages, system, False
    print("unknown command. /help")
    return messages, system, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Drop oldest turns when local history estimate exceeds this many tokens",
    )
    parser.add_argument("--once", help="Send one user prompt and exit (no REPL)")
    args = parser.parse_args()

    base_url, api_key, env_model = load_settings()
    model = args.model or env_model
    url = f"{base_url}/chat/completions"
    system = args.system
    messages: list[dict] = [{"role": "system", "content": system}]
    session_tokens = 0
    last_usage: dict = {}

    print(f"model={model}")
    print(f"POST {url}  stream=true")
    print(f"history budget ~{args.budget} tokens (local chars/4 estimate)")
    print("type /help  ·  /exit to quit")
    print()

    client = httpx.Client()
    try:
        while True:
            if args.once:
                user_text = args.once
            else:
                try:
                    user_text = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

            if not user_text:
                if args.once:
                    break
                continue

            if user_text.startswith("/"):
                messages, system, quit_repl = handle_command(user_text, messages, system)
                if quit_repl:
                    break
                continue

            messages.append({"role": "user", "content": user_text})
            messages, dropped = trim_history(messages, args.budget)
            if dropped:
                print(f"(dropped {dropped} old messages to stay under budget)")

            payload = {
                "model": model,
                "messages": messages,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            (HERE / "last-request.json").write_text(json.dumps(payload, indent=2) + "\n")

            try:
                content, usage, finish_reason, chunks = stream_chat(
                    client, url, api_key, payload
                )
            except RuntimeError as exc:
                messages.pop()  # do not keep a user turn the model never answered
                print(exc)
                if args.once:
                    sys.exit(1)
                continue

            (HERE / "last-stream.json").write_text(json.dumps(chunks, indent=2) + "\n")

            messages.append({"role": "assistant", "content": content})
            last_usage = usage or {}
            prompt_tokens = int(last_usage.get("prompt_tokens") or 0)
            completion_tokens = int(last_usage.get("completion_tokens") or 0)
            total_tokens = int(last_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            session_tokens += total_tokens

            print(
                f"turn  prompt={prompt_tokens}  completion={completion_tokens}  "
                f"total={total_tokens}  finish={finish_reason or '?'}  "
                f"cost={estimate_cost(base_url, model, prompt_tokens, completion_tokens)}"
            )
            print(
                f"session billed tokens={session_tokens}  "
                f"(each turn re-sends the full history)  "
                f"messages={len(messages)}  ~{approx_tokens(messages)} local"
            )
            print()

            if args.once:
                break
    finally:
        client.close()


if __name__ == "__main__":
    main()
