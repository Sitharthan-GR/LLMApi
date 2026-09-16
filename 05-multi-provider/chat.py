#!/usr/bin/env python3
"""One chat completion through a named provider profile.

  python 05-multi-provider/chat.py "What is a token?"
  python 05-multi-provider/chat.py --provider groq "Hello"
  python 05-multi-provider/chat.py --provider ollama "Hello"
  python 05-multi-provider/chat.py --provider anthropic "Hello"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from client import ChatClient
from profiles import get_profile, load_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    load_env(ROOT)
    profile = get_profile(args.provider)
    client = ChatClient(profile)
    messages = [
        {"role": "system", "content": "You are a concise tutor for LLM APIs."},
        {"role": "user", "content": args.prompt},
    ]
    print(f"provider={profile.name} kind={profile.kind} base_url={profile.base_url}")
    print(f"model={args.model or profile.default_model}")
    data = client.complete(
        messages,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    (HERE / "last-request-meta.json").write_text(
        json.dumps({"provider": profile.name, "kind": profile.kind, "base_url": profile.base_url}, indent=2)
        + "\n"
    )
    (HERE / "last-response.json").write_text(json.dumps(data, indent=2) + "\n")
    choice = data["choices"][0]
    print("assistant>")
    print(choice["message"].get("content") or "")
    usage = data.get("usage") or {}
    print()
    print(
        f"finish={choice.get('finish_reason')}  "
        f"prompt={usage.get('prompt_tokens')}  "
        f"completion={usage.get('completion_tokens')}"
    )


if __name__ == "__main__":
    main()
