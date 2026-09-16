# Week 3 notes: tools and structured JSON

Brush-up for `03-tool-agent`. Still **httpx**, still `POST /v1/chat/completions`. The model still only returns JSON. What is new: it can ask **us** to run a function, and we can demand **JSON that matches a schema**.

## The leap from week 2

Week 2: `messages` grow with `user` / `assistant` text.

Week 3: the assistant message can contain `tool_calls` instead of a final answer. We execute those calls **in our process**, append `role: "tool"` results, and POST the list again. Repeat until `finish_reason` is `stop`.

The model never runs `calculator`, `now`, or `read_file`. It only names them and fills arguments. If we skipped the Python handlers, nothing would happen.

## Run

From the repo root, `.venv` active:

```bash
python 03-tool-agent/agent.py "What is 17*19, and what time is it?"
python 03-tool-agent/agent.py "Read notes.md and summarize it in one sentence."
python 03-tool-agent/extract.py "LangChain wraps LLM APIs. People often start there too early."
```

Inspect after a run: `last-request.json`, `last-response.json`, `last-trace.json`.

## The agent loop

```text
you → POST messages + tools
model → finish_reason=tool_calls, tool_calls=[{name, arguments, id}]
you → run the function locally
you → append assistant (with tool_calls) + tool (result)
you → POST again
model → either more tool_calls, or finish_reason=stop + content
```

Same `messages` idea as week 2. Two extra roles/fields:

| Piece | Meaning |
| --- | --- |
| `tools` | JSON Schema catalog we advertise on **every** request |
| `tool_choice: auto` | Model may call a tool or answer in text |
| `message.tool_calls` | “Please run this function with these args” |
| `function.arguments` | A **string** of JSON, not an object — `json.loads` it |
| `role: tool` | Our result. `tool_call_id` must match `tool_calls[].id` |
| `finish_reason: tool_calls` | Not done yet |
| `finish_reason: stop` | Final answer in `content` |

`openai/gpt-oss-20b` does **not** do parallel tool use (one call per step). Other models may return several `tool_calls` at once; this script still loops over the array.

## Why we execute locally

- Allowlist: `read_file` can only open files under `sandbox/` (no `../.env`).
- `calculator` walks an AST — numbers and `+ - * / // % **` only. Not `eval()`, not shell.
- `now` is our clock, not the model’s training data.

If the model asks for `read_file` with `../../.env`, we return an error JSON in `role: tool`. We do not open the file.

## The three tools

| Name | What we actually do |
| --- | --- |
| `calculator` | Parse `expression`, compute, return `{"result": ...}` |
| `now` | `datetime.now().astimezone().isoformat()` |
| `read_file` | Read UTF-8 from `03-tool-agent/sandbox/<path>` |

Try a path outside the sandbox and read the `error` in the tool result.

## Structured JSON (`extract.py`)

No tools. `response_format: { "type": "json_object" }` plus a system prompt that names the keys:

```json
{ "title": "...", "tags": ["..."], "summary": "..." }
```

We **validate in Python**. If keys are wrong, we send the bad output back and retry once. The API flag is not enough by itself — always check the object.

`extract.py` uses `temperature` `0.1`; `agent.py` uses `0.2`. Ranges live in the week 1 README.

## Mental model to keep

1. Tools are JSON Schema ads. Execution is your code.
2. `tool_calls` is a request, not a side effect.
3. `role: tool` is how results re-enter `messages`.
4. Jail tools (paths, eval, network) in Python, not in the prompt.
5. Structured output = JSON mode + local schema check.
