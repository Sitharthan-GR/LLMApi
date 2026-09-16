#!/usr/bin/env python3
"""Week 3: structured JSON out (no tools). Same Chat Completions HTTP.

  python 03-tool-agent/extract.py "LangChain wraps LLM APIs. People often start there too early."
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

SCHEMA_KEYS = ("title", "tags", "summary")
SYSTEM = (
    "Extract a card from the user text. Reply with a JSON object only, keys: "
    "title (string), tags (array of 1-5 short strings), summary (one sentence). "
    "No markdown."
)


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
        sys.exit("Missing GROQ_API_KEY in repo-root .env")
    return base_url, api_key, model


def validate_card(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return "root must be an object"
    missing = [k for k in SCHEMA_KEYS if k not in obj]
    if missing:
        return f"missing keys: {missing}"
    if extra := [k for k in obj if k not in SCHEMA_KEYS]:
        return f"unexpected keys: {extra}"
    if not isinstance(obj["title"], str) or not obj["title"].strip():
        return "title must be a non-empty string"
    if not isinstance(obj["summary"], str) or not obj["summary"].strip():
        return "summary must be a non-empty string"
    tags = obj["tags"]
    if not isinstance(tags, list) or not tags or not all(isinstance(t, str) and t.strip() for t in tags):
        return "tags must be a non-empty array of strings"
    if len(tags) > 5:
        return "tags: at most 5"
    return None


def complete(client: httpx.Client, url: str, api_key: str, payload: dict) -> dict:
    response = client.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90.0,
    )
    data = response.json()
    (HERE / "last-extract-request.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "last-extract-response.json").write_text(json.dumps(data, indent=2) + "\n")
    if response.is_error:
        sys.exit(f"HTTP {response.status_code}: {data.get('error', data)}")
    return data


def parse_json_content(text: str) -> object:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    base_url, api_key, env_model = load_settings()
    model = args.model or env_model
    url = f"{base_url}/chat/completions"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.text},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client() as client:
        for attempt in (1, 2):
            data = complete(client, url, api_key, payload)
            raw = data["choices"][0]["message"].get("content") or ""
            print(f"attempt {attempt}  finish={data['choices'][0].get('finish_reason')}")
            try:
                obj = parse_json_content(raw)
            except json.JSONDecodeError as exc:
                err = f"not JSON: {exc}"
                obj = None
            else:
                err = validate_card(obj)

            if not err:
                print(json.dumps(obj, indent=2))
                return

            print(f"rejected: {err}")
            print(f"raw: {raw[:400]}")
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid JSON ({err}). Return only a valid object with title, tags, summary.",
                }
            )
            payload["messages"] = messages

    sys.exit("schema still invalid after retry")


if __name__ == "__main__":
    main()
