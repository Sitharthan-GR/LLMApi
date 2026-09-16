# A token is a chunk of text

In an LLM API, billing and limits are measured in tokens, not words or characters.

A token can be a whole word ("hello"), part of a word, punctuation, or a space.
The model reads tokens and writes tokens. `usage.prompt_tokens` counts input.
`usage.completion_tokens` counts generated output. `total_tokens` is the sum.

`openai/gpt-oss-20b` can also spend completion tokens on hidden reasoning
before any visible `content` appears. A tiny `max_tokens` can yield an empty
reply with non-zero usage.

Finish reason `length` means `max_tokens` cut the reply off. `stop` means the
model finished on its own.
