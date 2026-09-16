#!/usr/bin/env bash
# Week 1: the same Chat Completions call with curl. No Python.
# Default: Groq free OpenAI-compatible API.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Copy .env.example to .env and paste your Groq API key." >&2
  echo "Free key: https://console.groq.com/keys" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

API_KEY="${LLM_API_KEY:-${GROQ_API_KEY:-${OPENAI_API_KEY:-}}}"
BASE_URL="${LLM_BASE_URL:-https://api.groq.com/openai/v1}"
BASE_URL="${BASE_URL%/}"
MODEL="${LLM_MODEL:-openai/gpt-oss-20b}"

if [[ -z "$API_KEY" || "$API_KEY" == "gsk_..." || "$API_KEY" == "sk-..." ]]; then
  echo "GROQ_API_KEY is missing in .env — https://console.groq.com/keys" >&2
  exit 1
fi

PROMPT="${1:-What is a token in an LLM API? Answer in two sentences.}"

python3 - "$PROMPT" "$MODEL" <<'PY' > 01-hello-chat/last-request.json
import json, sys
prompt, model = sys.argv[1], sys.argv[2]
json.dump({
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a concise tutor for someone learning LLM APIs."},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.7,
    "max_tokens": 512,
}, sys.stdout, indent=2)
print()
PY

echo "→ POST ${BASE_URL}/chat/completions"
cat 01-hello-chat/last-request.json
echo

curl -sS "${BASE_URL}/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @01-hello-chat/last-request.json \
  | tee 01-hello-chat/last-response.json

echo
echo
echo "saved 01-hello-chat/last-request.json and 01-hello-chat/last-response.json"
