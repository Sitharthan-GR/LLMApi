# Tools and the agent loop

The model cannot run Python. It returns `tool_calls` with a function name
and a JSON string of arguments. Your code executes the function, then
appends a message with `role: tool` and the matching `tool_call_id`.

Loop:

1. POST messages plus a `tools` catalog (JSON Schema).
2. If `finish_reason` is `tool_calls`, run the functions locally.
3. POST again with the tool results.
4. Repeat until `finish_reason` is `stop` and `content` is the answer.

Jail tools in Python: allowlist file paths, never `eval()` arbitrary code,
never let the model pick a shell command. The prompt is not a security boundary.

`openai/gpt-oss-20b` does not support parallel tool use (one call per step).
