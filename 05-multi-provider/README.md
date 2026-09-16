# Week 5 notes: providers and vision

Brush-up for `05-multi-provider`. Still httpx. New idea: **`base_url` + API key + default model` live in a profile**, not scattered `if groq` blocks. Most hosts speak Chat Completions. Anthropic does not.

## Run

```bash
python 05-multi-provider/chat.py "What is a token?"
python 05-multi-provider/chat.py --provider groq "Hello"
python 05-multi-provider/compare.py "Explain finish_reason length in one sentence."
python 05-multi-provider/vision.py
```

`--provider ollama` needs a local Ollama daemon. `--provider anthropic` needs `ANTHROPIC_API_KEY`. `--provider openai` needs `OPENAI_API_KEY`. Default is Groq, same key as week 1.

## Profiles (`profiles.py`)

| name | kind | base_url | default model |
| --- | --- | --- | --- |
| groq | openai | `https://api.groq.com/openai/v1` | `openai/gpt-oss-20b` |
| openai | openai | `https://api.openai.com/v1` | `gpt-4o-mini` |
| ollama | openai | `http://127.0.0.1:11434/v1` | `llama3.2` (key can be `ollama`) |
| anthropic | anthropic | `https://api.anthropic.com/v1` | `claude-3-5-haiku-latest` |

Groq, OpenAI, Ollama, vLLM: **same JSON**. You change host and model id. An official OpenAI SDK would do `OpenAI(base_url=..., api_key=...)`. We still build that POST ourselves.

## Anthropic payload diffs

OpenAI-compatible:

```json
{
  "messages": [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "Hi"}
  ],
  "max_tokens": 256
}
```

Anthropic Messages (`anthropic_adapter.py`):

```json
{
  "system": "Be terse.",
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
  ],
  "max_tokens": 256
}
```

| OpenAI-compat | Anthropic |
| --- | --- |
| `Authorization: Bearer` | `x-api-key` + `anthropic-version` |
| `system` is a message role | `system` is a top-level string |
| `content` is usually a string | `content` is a list of blocks |
| `max_tokens` optional on many models | `max_tokens` required |
| `finish_reason` | `stop_reason` (`end_turn` ≈ `stop`) |
| `usage.prompt_tokens` | `usage.input_tokens` |

`ChatClient` maps Anthropic back to the OpenAI-shaped object so `chat.py` prints the same fields.

## Vision (`vision.py`)

User `content` becomes a **list of parts**, not a string:

```json
[
  {"type": "text", "text": "What color?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]
```

On Groq, `openai/gpt-oss-20b` is text. Vision uses `qwen/qwen3.8-27b`. `sample.png` is a red square generated with the standard library.

## Compare (`compare.py`)

Same prompt, two `provider:model` pairs. Default: Groq `gpt-oss-20b` vs Groq `qwen/qwen3.8-27b`. Watch latency and token counts. On the free tier the dollar cost stays $0.

## Mental model to keep

1. Config object, not vendor if-else sprinkled through the app.
2. OpenAI-compat is a dialect. Anthropic is a different dialect; translate at the edge.
3. Vision is still Chat Completions; only `content` parts change.
4. Model cards matter: tools, vision, context, price. Do not assume every id on a host can do everything.
