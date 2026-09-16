#!/usr/bin/env python3
"""Week 3: local tool-calling loop over raw HTTP. No official SDK.

The model cannot do math, clocks, or files by itself. It returns a
tool_calls object; *we* run the function, append a role=tool message,
and POST again. That loop is an agent.

  python 03-tool-agent/agent.py "What is 17*19, and what time is it?"
  python 03-tool-agent/agent.py "Read notes.md and summarize it in one sentence."
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
SANDBOX = (HERE / "sandbox").resolve()

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 8
MAX_FILE_BYTES = 32_768

SYSTEM = (
    "You are a concise tutor. Use tools for arithmetic, the current time, "
    "and reading sandbox files. Never invent file contents or calculation results. "
    "When you have what you need, answer the user in plain text."
)

# JSON Schema the model sees. It chooses a name + arguments; it does not run code.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression. Use for any math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. 17*19 or (3+4)/2",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "now",
            "description": "Return the current local date and time as an ISO-8601 string.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from the sandbox folder. "
                "Pass a relative path such as notes.md. Absolute paths and .. are rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the sandbox, e.g. notes.md",
                    }
                },
                "required": ["path"],
            },
        },
    },
]

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


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


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("only numbers and + - * / // % ** are allowed")


def tool_calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    value = _eval_ast(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return json.dumps({"expression": expression, "result": value})


def tool_now() -> str:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    return json.dumps({"now": stamp})


def tool_read_file(path: str) -> str:
    target = (SANDBOX / path).resolve()
    if not target.is_relative_to(SANDBOX):
        raise ValueError("path is outside the sandbox")
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    data = target.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("file too large")
    return json.dumps({"path": path, "content": data.decode("utf-8")})


HANDLERS = {
    "calculator": lambda args: tool_calculator(args["expression"]),
    "now": lambda args: tool_now(),
    "read_file": lambda args: tool_read_file(args["path"]),
}


def run_tool(name: str, arguments_json: str) -> str:
    if name not in HANDLERS:
        return json.dumps({"error": f"unknown tool {name!r}"})
    try:
        args = json.loads(arguments_json or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        return HANDLERS[name](args)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def chat(client: httpx.Client, url: str, api_key: str, payload: dict) -> dict:
    (HERE / "last-request.json").write_text(json.dumps(payload, indent=2) + "\n")
    response = client.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90.0,
    )
    try:
        data = response.json()
    except json.JSONDecodeError:
        sys.exit(f"Non-JSON HTTP {response.status_code}: {response.text}")
    (HERE / "last-response.json").write_text(json.dumps(data, indent=2) + "\n")
    if response.is_error:
        err = data.get("error", data)
        sys.exit(f"HTTP {response.status_code}: {err}")
    return data


def assistant_message_for_history(message: dict) -> dict:
    """Keep the fields the next request needs, especially tool_calls + id."""
    out: dict = {"role": "assistant"}
    if message.get("content"):
        out["content"] = message["content"]
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="User task for the agent")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    base_url, api_key, env_model = load_settings()
    model = args.model or env_model
    url = f"{base_url}/chat/completions"
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.prompt},
    ]

    print(f"model={model}  POST {url}")
    print("tools: calculator, now, read_file (sandbox only)")
    print()

    trace: list[dict] = []
    with httpx.Client() as client:
        for step in range(1, args.max_steps + 1):
            payload = {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            data = chat(client, url, api_key, payload)
            choice = data["choices"][0]
            message = choice["message"]
            finish = choice.get("finish_reason")
            usage = data.get("usage") or {}
            tool_calls = message.get("tool_calls") or []

            print(
                f"step {step}  finish={finish}  "
                f"prompt={usage.get('prompt_tokens')}  "
                f"completion={usage.get('completion_tokens')}"
            )

            if finish == "tool_calls" or tool_calls:
                messages.append(assistant_message_for_history(message))
                for call in tool_calls:
                    fn = call.get("function") or {}
                    name = fn.get("name") or ""
                    raw_args = fn.get("arguments") or "{}"
                    result = run_tool(name, raw_args)
                    print(f"  → {name}({raw_args})")
                    print(f"  ← {result}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "name": name,
                            "content": result,
                        }
                    )
                    trace.append(
                        {
                            "step": step,
                            "tool": name,
                            "arguments": raw_args,
                            "result": result,
                        }
                    )
                print()
                continue

            content = message.get("content") or ""
            print("assistant>")
            print(content)
            print()
            trace.append({"step": step, "finish": finish, "content": content})
            (HERE / "last-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
            return

    sys.exit(f"stopped after {args.max_steps} steps with no final answer")


if __name__ == "__main__":
    main()
