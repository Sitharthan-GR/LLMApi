#!/usr/bin/env python3
"""Same prompt, two models. Prints latency and token usage.

  python 05-multi-provider/compare.py "Explain finish_reason length in one sentence."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from client import ChatClient
from profiles import get_profile, load_env


def run_one(provider: str, model: str | None, prompt: str) -> dict:
    profile = get_profile(provider)
    client = ChatClient(profile)
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    data = client.complete(messages, model=model, temperature=0.2, max_tokens=256)
    ms = (time.perf_counter() - t0) * 1000
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    content = (choice["message"].get("content") or "").replace("\n", " ")
    if len(content) > 140:
        content = content[:137] + "..."
    return {
        "provider": profile.name,
        "model": model or profile.default_model,
        "latency_ms": round(ms),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": choice.get("finish_reason"),
        "preview": content,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument(
        "--left",
        default="groq:openai/gpt-oss-20b",
        help="provider:model",
    )
    parser.add_argument(
        "--right",
        default="groq:qwen/qwen3.8-27b",
        help="provider:model",
    )
    args = parser.parse_args()
    load_env(ROOT)

    rows = []
    for spec in (args.left, args.right):
        if ":" not in spec:
            sys.exit("use provider:model e.g. groq:openai/gpt-oss-20b")
        provider, model = spec.split(":", 1)
        print(f"→ {provider} {model}")
        rows.append(run_one(provider, model, args.prompt))

    (HERE / "last-compare.json").write_text(json.dumps(rows, indent=2) + "\n")
    print()
    print(f"{'provider':<10} {'model':<28} {'ms':>6} {'in':>5} {'out':>5}  preview")
    for row in rows:
        print(
            f"{row['provider']:<10} {row['model']:<28} {row['latency_ms']:>6} "
            f"{str(row['prompt_tokens']):>5} {str(row['completion_tokens']):>5}  {row['preview']}"
        )
    print()
    print("Groq free tier: dollar cost is $0. Latency and tokens still differ by model.")


if __name__ == "__main__":
    main()
