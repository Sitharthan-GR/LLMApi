#!/usr/bin/env python3
"""Score whether the expected source file is in the top-k retrieved chunks.

  python 04-rag/eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raglib import load_index, retrieve

CASES = [
    {"q": "What is a token in an LLM API?", "file": "tokens.md"},
    {"q": "What do prompt_tokens and completion_tokens measure?", "file": "tokens.md"},
    {"q": "What does finish_reason length mean?", "file": "tokens.md"},
    {"q": "What are the system user and assistant roles?", "file": "messages.md"},
    {"q": "Does the Chat Completions API remember past turns by itself?", "file": "messages.md"},
    {"q": "What is delta.content in a streaming response?", "file": "streaming.md"},
    {"q": "Who executes tool functions, the model or our Python?", "file": "tools.md"},
    {"q": "What toppings are on a Neapolitan pizza?", "file": "pizza.md"},
]


def main() -> None:
    index = load_index()
    k = 3
    hits = 0
    print(f"{'hit':<4} {'expected':<14} sources  question")
    for case in CASES:
        ranked = retrieve(index, case["q"], k=k)
        sources = [row["source"] for row in ranked]
        ok = case["file"] in sources
        hits += int(ok)
        mark = "yes" if ok else "NO"
        print(f"{mark:<4} {case['file']:<14} {sources}  {case['q']}")
    print()
    print(f"hit-rate {hits}/{len(CASES)} (expected file in top {k})")


if __name__ == "__main__":
    main()
