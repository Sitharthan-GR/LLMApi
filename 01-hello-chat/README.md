# Week 1 notes: one chat completion

Brush-up for `01-hello-chat`. We did **not** use an official SDK. One HTTP request, JSON in, JSON out.

## What this folder is

A single call to:

```text
POST {LLM_BASE_URL}/chat/completions
Authorization: Bearer {API_KEY}
Content-Type: application/json
```

Default host: Groq’s free OpenAI-compatible API  
`https://api.groq.com/openai/v1/chat/completions`

Same payload shape as OpenAI. Switching providers later is mostly a different **base URL** and **model id**.

## ChatGPT Plus is not the API

- **ChatGPT Plus** = the chat app. No `api.openai.com` access.
- **OpenAI API** = paid, separate key + billing at [platform.openai.com](https://platform.openai.com/api-keys).
- **This project** = Groq free key from [console.groq.com/keys](https://console.groq.com/keys) (`gsk_...`).

## Run

From the repo root, with `.venv` active and `.env` filled in:

```bash
python 01-hello-chat/chat.py "What is a token in an LLM API?"
python 01-hello-chat/chat.py --list-models
./01-hello-chat/curl-example.sh "What is a token in an LLM API?"
```

Useful flags:

```bash
python 01-hello-chat/chat.py "Explain HTTP" --system "Answer in one sentence."
python 01-hello-chat/chat.py "Write a haiku" --temperature 0.2
python 01-hello-chat/chat.py "Write a haiku" --temperature 1.5
python 01-hello-chat/chat.py "Explain HTTP" --max-tokens 20
```

Dumps land in `last-request.json` and `last-response.json` (gitignored). Read those when you want the raw contract.

`.env` (repo root):

- `GROQ_API_KEY` — required
- `LLM_BASE_URL` — default `https://api.groq.com/openai/v1`
- `LLM_MODEL` — default `openai/gpt-oss-20b`

## The request body (`build_payload`)

```json
{
  "model": "openai/gpt-oss-20b",
  "messages": [
    { "role": "system", "content": "You are a concise tutor..." },
    { "role": "user", "content": "What is a token in an LLM API?" }
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

| Field | Role |
| --- | --- |
| `model` | Which hosted model to run. Same endpoint, different weights. |
| `messages` | The conversation so far. |
| `temperature` | How randomly the next token is sampled. |
| `max_tokens` | Cap on **generated** tokens (not prompt tokens). Groq / most OpenAI-compatible hosts use this name. OpenAI’s newer name is `max_completion_tokens` (required for o-series). |

`build_payload` only builds that dict. `httpx` then POSTs it.

## Messages and roles

Each message is `{ "role", "content" }`.

| Role | Meaning |
| --- | --- |
| `system` | Instructions for the model. Optional. We skip it if `--system` is empty. |
| `user` | Your prompt. |
| `assistant` | The model’s reply. Comes back in the **response**. Week 2 is when you send it back as history. |

## Tokens and usage

A **token** is a chunk of text the model actually reads/writes (a word, part of a word, punctuation). APIs count and (on paid hosts) bill by tokens.

Response `usage`:

- `prompt_tokens` — input (system + user, and later history)
- `completion_tokens` — generated output
- `total_tokens` — sum

`openai/gpt-oss-20b` also spends completion tokens on **reasoning**. If `max_tokens` is too small, `content` can be empty even though usage is non-zero. Look at `message.reasoning` in `last-response.json`.

## `finish_reason`

- `stop` — the model finished on its own
- `length` — `max_tokens` cut it off; raise the cap if you needed the rest

## Temperature

Temperature does **not** make the model smarter. It only changes how randomly it picks the next token from the probability list.

- **Low (near 0):** almost always the most likely token. Stable, repeatable. Good for facts/code.
- **Mid (~0.7):** default in this script. Normal chat.
- **High (1.0–1.5):** more variety, more chance of ignoring instructions.

Live example we ran (same prompt, twice each):

*Prompt:* `Write one 8-word sentence about a robot eating pizza.`

| temperature | run 1 | run 2 |
| --- | --- | --- |
| `0.2` | The robot devours pizza with metallic enthusiasm daily. | The robot devours pizza with metallic enthusiasm daily. |
| `1.5` | The robot devours pizza with precise mechanical delight. | The robot devours a steaming slice of pizza. |

Low temp repeated itself. High temp wandered.

Suggested ranges:

| Task                               | Suggested temperature |
| ---------------------------------- | --------------------: |
| Data extraction                    |               `0–0.2` |
| SQL or DAX generation              |               `0–0.3` |
| Business report summary            |             `0.2–0.5` |
| General chatbot                    |             `0.5–0.8` |
| Brainstorming names and ideas      |             `0.8–1.2` |
| Creative stories or marketing copy |             `1.0–1.5` |

Groq turns exact `temperature: 0` into a tiny epsilon (`1e-8`). Use `0.1` if you want “basically greedy.”

## `list_models`

Does **not** generate text. It **GET**s `{base_url}/models` with the same Bearer key and prints every `id` this key can use.

```bash
python 01-hello-chat/chat.py --list-models
```

We needed this because `llama-3.1-8b-instant` returned:

```text
HTTP 404 (model_not_found)
```

That id was gone from Groq’s catalog. Live chat models on this key included `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`. Default is now `openai/gpt-oss-20b`.

Model ids go stale. `--list-models` beats old docs.

## Auth and errors we care about

| Status | Meaning |
| --- | --- |
| `401` | Bad/missing key. Groq keys start with `gsk_`. |
| `404` `model_not_found` | Wrong model id, or no access. List models. |
| `429` | Free-tier rate limit. Wait and retry. |

Never commit `.env`.

## Mental model to keep

1. An LLM API is a next-token generator behind HTTP.
2. You send a **message list**, not a magic prompt box.
3. Inspect `last-request.json` / `last-response.json` until that JSON is obvious.
4. Measure tokens on every call.
5. Frameworks (LangChain, etc.) wrap this same payload. Learn this first.
