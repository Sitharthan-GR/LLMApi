# Week 2 notes: streaming chat with history

Brush-up for `02-chat-cli`. Same Chat Completions JSON as week 1. Two additions: **`stream: true`** and a **growing `messages` list**.

## The leap from week 1

Week 1: one `user` message → wait → one full `assistant` JSON.

Week 2:

1. Tokens print as they are generated (SSE), not after the whole reply.
2. We append that assistant text onto `messages` and send the **entire list** next turn.

The API has no session. If you omit an old message, the model never saw it.

## Run

From the repo root, `.venv` active, `.env` already set from week 1:

```bash
python 02-chat-cli/chat.py
```

One-shot (no loop, useful for testing):

```bash
python 02-chat-cli/chat.py --once "What did we just say about temperature?"
```

REPL commands:

| Command | Effect |
| --- | --- |
| `/help` | list commands |
| `/reset` | drop user/assistant turns; keep system prompt |
| `/system <text>` | replace the system prompt |
| `/history` | print the `messages` array we will send next |
| `/exit` | quit (also Ctrl-D / Ctrl-C) |

## Streaming (`stream: true`)

Request is the week 1 body plus:

```json
{
  "stream": true,
  "stream_options": { "include_usage": true }
}
```

The server replies with **Server-Sent Events**, not one JSON object. Each line looks like:

```text
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" there"}}]}
data: {"choices":[],"usage":{"prompt_tokens":80,"completion_tokens":12,"total_tokens":92}}
data: [DONE]
```

| Piece | Meaning |
| --- | --- |
| `delta.content` | The next slice of the assistant reply. Concatenate these to rebuild the full message. |
| `finish_reason` | Same as week 1 (`stop` / `length`), usually on a late chunk. |
| `usage` | Only on the **last** chunk, and only if `include_usage` is true. Intermediate chunks have `usage: null`. |
| `data: [DONE]` | Stream is over. |

Week 1 `message.content` is just all those `delta.content` pieces joined.

`iter_sse_json` in `chat.py` is the parser: skip blank lines, strip the `data:` prefix, stop on `[DONE]`.

Inspect `last-request.json` and `last-stream.json` after a turn.

## History is the source of truth

After a turn the list looks like:

```json
[
  { "role": "system", "content": "You are a concise tutor..." },
  { "role": "user", "content": "What is a token?" },
  { "role": "assistant", "content": "A token is a chunk of text..." },
  { "role": "user", "content": "Give an example." }
]
```

Turn 2 **re-sends** the system prompt, the first Q&A, and the new question. That is why `prompt_tokens` grows even if your new question is short.

Try it:

```text
you> My favorite number is 17. Just remember that.
you> What is my favorite number?
you> /history
you> /reset
you> What is my favorite number?
```

After `/reset` it should not know `17`. Nothing was stored on the server.

## Why chats get expensive

You are billed (or rate-limited) per **request**, and each request includes the full history.

If turns cost roughly 100 + 200 + 300 tokens, session billed tokens are **600**, not 300. The CLI prints both:

- **this turn** — `prompt_tokens` / `completion_tokens` from the usage chunk
- **session billed tokens** — sum of `total_tokens` across turns

On Groq’s free tier the dollar amount stays `$0`, but the token counters still show the shape of a paid bill.

## History budget (`--budget`)

Default `4000` local tokens (chars/4 estimate — not the official tokenizer).

When the estimate exceeds the budget, we drop the **oldest non-system** messages (a user turn and its assistant reply if present). System prompt stays.

That is a crude version of what ChatGPT does when a thread gets long: the model only sees what you still send.

`--budget 200` makes trimming easy to see. The API `usage.prompt_tokens` is the number that actually matters.

## `openai/gpt-oss-20b` reminder

This model spends some `completion_tokens` on **reasoning** before `delta.content` appears. A tiny `--max-tokens` can yield an empty reply with non-zero usage (same as week 1). Keep `--max-tokens` at least a few hundred.

## Temperature (from week 1)

Still a request field. Suggested ranges:

| Task                               | Suggested temperature |
| ---------------------------------- | --------------------: |
| Data extraction                    |               `0–0.2` |
| SQL or DAX generation              |               `0–0.3` |
| Business report summary            |             `0.2–0.5` |
| General chatbot                    |             `0.5–0.8` |
| Brainstorming names and ideas      |             `0.8–1.2` |
| Creative stories or marketing copy |             `1.0–1.5` |

This CLI defaults to `0.7`.

## Mental model to keep

1. Streaming is the same completion, chunked over SSE.
2. `delta.content` pieces rebuild `message.content`.
3. Conversation = the `messages` array you send. The host does not remember you.
4. Every turn re-sends history, so prompt tokens grow.
5. Trim or `/reset` when the window (or the bill) gets too large.
