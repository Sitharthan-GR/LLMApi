# Week 4 notes: embeddings and RAG

Brush-up for `04-rag`. Chat is still `POST /v1/chat/completions` with httpx. New idea: **do not stuff the whole corpus into the prompt**. Embed chunks, retrieve the closest ones, then generate.

## The leap from week 3

Week 3 tools fetch one fact (math, clock, one file). RAG answers questions over **many files** by retrieving the few chunks that look relevant.

```text
docs/*.md  →  chunk  →  embed  →  index.json
question   →  embed  →  cosine vs index  →  top-k chunks  →  chat completion
```

The chat model never sees `pizza.md` unless retrieval ranked it.

## Run

```bash
python 04-rag/similarity.py
python 04-rag/build_index.py
python 04-rag/ask.py "What is a token in an LLM API?"
python 04-rag/eval.py
```

## Cosine similarity (`similarity.py`)

An embedding is a list of numbers. Cosine is:

```text
cos(a, b) = dot(a, b) / (|a| |b|)
```

- `1` — same direction (similar)
- `0` — unrelated
- `-1` — opposite

We compared three sentences. The two token lines should beat the pizza line.

## Chunking (`build_index.py`)

Naive split-by-character cuts sentences in half and mixes topics. This folder:

1. Splits on markdown `#` headings first
2. Then uses a sliding window (`size=500`, `overlap=80`)

Overlap means the end of one chunk repeats at the start of the next, so a sentence on the boundary still appears whole somewhere.

## Why this uses local TF-IDF

Embeddings APIs look like chat, different path:

```text
POST /v1/embeddings
{ "model": "...", "input": ["text", "..."] }
```

Response: `{ "data": [ { "index": 0, "embedding": [0.1, -0.2, ...] } ] }`.

This Groq account 404s `nomic-embed-text-v1_5`. So the default backend is **local TF-IDF** (word counts × inverse document frequency). Same cosine, lexical not semantic: it matches overlapping terms, not paraphrases.

To use a real embeddings HTTP API later, set in `.env` and rebuild:

```text
LLM_EMBED_URL=https://api.openai.com/v1/embeddings
LLM_EMBED_MODEL=text-embedding-3-small
LLM_EMBED_API_KEY=sk-...
```

`ask.py` / `eval.py` do not change. Only the vectors change.

## Retrieve then generate (`ask.py`)

1. Embed the question
2. Score every chunk with cosine, keep top-k
3. If scores are tiny, refuse (“I do not know”) instead of hallucinating
4. POST those chunks as CONTEXT plus the question
5. Ask the model to cite `[filename]`

That is RAG. The generation model can still lie; retrieval plus “use only CONTEXT” makes it less likely.

Stuffing the whole `docs/` folder into one prompt also “works” at this size. RAG matters when the corpus is bigger than the context window, or you want citations and less noise.

## Eval (`eval.py`)

Eight questions with an expected source file. A hit means that file appears in the top-3 chunks. This scores **retrieval**, not whether the final sentence is pretty.

## When RAG is the wrong tool

- The doc is one page — just put it in the prompt (week 2 history).
- You need live data or actions — tools (week 3).
- You need a new skill in the model weights — fine-tune, not this week.

## Mental model to keep

1. Embeddings turn text into vectors. Cosine ranks closeness.
2. Chunking quality beats a fancy vector database at this scale. There is no vector DB here — `index.json` is the index.
3. RAG = retrieve k chunks, then generate. The LLM still has no memory of your files.
4. Always cite sources. Weak scores → “I do not know.”
