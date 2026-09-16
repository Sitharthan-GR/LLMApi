#!/usr/bin/env python3
"""Retrieve top-k chunks, then generate an answer with citations.

  python 04-rag/ask.py "What is a token in an LLM API?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raglib import HERE, load_index, load_settings, retrieve

SYSTEM = (
    "Answer using only the CONTEXT. Cite source filenames in square brackets, "
    "e.g. [tokens.md]. If the context is not enough, say you do not know. "
    "Do not use the pizza notes unless the question is about pizza."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    index = load_index()
    hits = retrieve(index, args.question, k=args.k)
    print("retrieved")
    for hit in hits:
        preview = hit["text"].replace("\n", " ")[:90]
        print(f"  {hit['score']:.3f}  [{hit['source']}] {preview}")
    print()

    strong = [h for h in hits if h["score"] >= args.min_score]
    if not strong:
        print("assistant>")
        print("I do not know — retrieval scores were too weak to ground an answer.")
        return

    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in strong)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION: {args.question}",
        },
    ]
    base_url, api_key, env_model = load_settings()
    model = args.model or env_model
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
    }
    url = f"{base_url}/chat/completions"
    (HERE / "last-request.json").write_text(json.dumps(payload, indent=2) + "\n")

    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90.0,
    )
    data = response.json()
    (HERE / "last-response.json").write_text(json.dumps(data, indent=2) + "\n")
    if response.is_error:
        sys.exit(f"HTTP {response.status_code}: {data}")

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
