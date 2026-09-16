# Messages and roles

Chat Completions is a POST of JSON to `/v1/chat/completions`.
The conversation is the `messages` array. The host does not remember you.

Each message has a role:

- system: instructions for the model
- user: what you typed
- assistant: what the model said last time (you send it back on the next turn)
- tool: the result of a function you ran locally (week 3)

`build_payload` in week 1 only builds that JSON. httpx sends it.
Switching providers is mostly a different base URL and model id.

ChatGPT Plus is the chat app. It is not an API key.
