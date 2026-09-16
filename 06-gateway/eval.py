#!/usr/bin/env python3
"""Run golden evals against Groq or the local gateway.

  python 06-gateway/eval.py
  python 06-gateway/eval.py --gateway http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "05-multi-provider"))

from client import ChatClient
from profiles import get_profile, load_env


def normalize(text: str) -> str:
    text = (text or "").strip()
    text = text.strip("`\"'")
    first = text.splitlines()[0].strip() if text else ""
    return re.sub(r"\s+", " ", first)


def passes(got: str, expect: dict) -> bool:
    norm = normalize(got)
    if "equals" in expect:
        want = str(expect["equals"]).upper()
        if norm.upper() == want:
            return True
        return re.search(rf"\b{re.escape(want)}\b", (got or "").upper()) is not None
    if "contains" in expect:
        return str(expect["contains"]).lower() in (got or "").lower()
    return False


def complete_direct(prompt: str, model: str | None, max_tokens: int) -> tuple[str, dict]:
    client = ChatClient(get_profile(None))
    data = client.complete(
        [
            {"role": "system", "content": "Follow the user instructions exactly. Do not add extra words unless asked."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    content = data["choices"][0]["message"].get("content") or ""
    return content, data.get("usage") or {}


def complete_gateway(base: str, prompt: str, model: str | None, max_tokens: int) -> tuple[str, dict]:
    url = base.rstrip("/") + "/chat/completions"
    response = httpx.post(
        url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "Follow the user instructions exactly. Do not add extra words unless asked."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=90.0,
    )
    data = response.json()
    if response.is_error:
        raise RuntimeError(f"gateway HTTP {response.status_code}: {data}")
    content = data["choices"][0]["message"].get("content") or ""
    return content, data.get("usage") or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", default=None, help="e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    load_env(ROOT)

    cases = json.loads((HERE / "evals.json").read_text())
    passed = 0
    token_sum = 0
    print(f"{'id':<20} {'result':<6} got")
    for case in cases:
        if args.gateway:
            content, usage = complete_gateway(args.gateway, case["prompt"], args.model, args.max_tokens)
        else:
            content, usage = complete_direct(case["prompt"], args.model, args.max_tokens)
        ok = passes(content, case["expect"])
        passed += int(ok)
        token_sum += int(usage.get("total_tokens") or 0)
        preview = normalize(content)[:60]
        print(f"{case['id']:<20} {'PASS' if ok else 'FAIL':<6} {preview}")

    n = len(cases)
    avg = token_sum / n if n else 0
    print()
    print(f"passed {passed}/{n}  avg total_tokens {avg:.1f}")
    (HERE / "last-eval.json").write_text(
        json.dumps({"passed": passed, "n": n, "avg_total_tokens": avg}, indent=2) + "\n"
    )
    if passed < n:
        sys.exit(1)


if __name__ == "__main__":
    main()
