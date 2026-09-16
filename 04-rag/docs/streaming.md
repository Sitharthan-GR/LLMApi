# Streaming with Server-Sent Events

Week 1 waits for one JSON object. Week 2 sets `"stream": true`.

The server then sends Server-Sent Events: lines that start with `data:`.
Each JSON chunk has `choices[0].delta.content` — the next slice of the reply.
Join those pieces to rebuild `message.content`.
The stream ends with `data: [DONE]`.

Token usage on a stream is usually missing until the last chunk.
Ask for it with `"stream_options": {"include_usage": true}`.

Every turn re-sends the full history, so `prompt_tokens` grow even if the new
question is short. That is why long chats get expensive.
