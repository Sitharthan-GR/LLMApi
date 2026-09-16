#!/usr/bin/env python3
"""Shared RAG helpers: chunking, cosine, local TF-IDF, optional HTTP embeddings."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"
INDEX_PATH = HERE / "index.json"

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1_5"

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "who",
    "with",
    "you",
}


def load_settings() -> tuple[str, str, str]:
    load_dotenv(ROOT / ".env")
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    chat_model = os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
    if not api_key or api_key in {"gsk_...", "sk-..."}:
        raise SystemExit("Missing GROQ_API_KEY in repo-root .env")
    return base_url, api_key, chat_model


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def cosine(a: list[float], b: list[float]) -> float:
    """dot(a,b) / (|a| |b|). 1 = same direction, 0 = orthogonal, -1 = opposite."""
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Split on headings, then sliding windows. Overlap keeps a sentence from being cut in half."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    sections = re.split(r"(?m)(?=^# )", text)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= size:
            chunks.append(section)
            continue
        start = 0
        while start < len(section):
            end = min(len(section), start + size)
            piece = section[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(section):
                break
            start = max(0, end - overlap)
    return chunks


def load_docs() -> list[tuple[str, str]]:
    files = sorted(DOCS.glob("*.md"))
    if not files:
        raise SystemExit(f"no markdown files in {DOCS}")
    return [(path.name, path.read_text()) for path in files]


class TfidfEmbedder:
    """Local stand-in for POST /v1/embeddings. Same cosine math, lexical not semantic."""

    def __init__(self, vocab: list[str], idf: list[float]):
        self.vocab = vocab
        self.idf = idf
        self.index = {term: i for i, term in enumerate(vocab)}

    @classmethod
    def fit(cls, texts: list[str]) -> "TfidfEmbedder":
        docs_tokens = [tokenize(t) for t in texts]
        df: dict[str, int] = {}
        for tokens in docs_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        vocab = sorted(df)
        n = max(len(texts), 1)
        idf = [math.log((1 + n) / (1 + df[term])) + 1.0 for term in vocab]
        return cls(vocab, idf)

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * len(self.vocab)
        if not tokens:
            return vec
        counts: dict[str, int] = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        length = len(tokens)
        for term, count in counts.items():
            i = self.index.get(term)
            if i is None:
                continue
            tf = count / length
            vec[i] = tf * self.idf[i]
        return vec


def http_embed(texts: list[str], url: str, api_key: str, model: str) -> list[list[float]]:
    """OpenAI-compatible embeddings: POST {url} with {model, input}."""
    vectors: list[list[float] | None] = [None] * len(texts)
    batch_size = 16
    with httpx.Client(timeout=90.0) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": batch},
            )
            data = response.json()
            if response.is_error:
                raise RuntimeError(f"HTTP {response.status_code} embeddings: {data}")
            for item in data["data"]:
                vectors[start + item["index"]] = item["embedding"]
    if any(v is None for v in vectors):
        raise RuntimeError("embedding response missing an index")
    return [v for v in vectors if v is not None]


def embed_backend() -> str:
    load_dotenv(ROOT / ".env")
    return (os.getenv("LLM_EMBED_URL") or "").strip()


def embed_many(texts: list[str], embedder: TfidfEmbedder | None = None) -> tuple[str, list[list[float]], TfidfEmbedder | None]:
    url = embed_backend()
    if url:
        load_dotenv(ROOT / ".env")
        api_key = (
            os.getenv("LLM_EMBED_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        model = os.getenv("LLM_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        return "http", http_embed(texts, url.rstrip("/"), api_key, model), None
    if embedder is None:
        embedder = TfidfEmbedder.fit(texts)
    return "tfidf", [embedder.embed(t) for t in texts], embedder


def save_index(payload: dict) -> None:
    INDEX_PATH.write_text(json.dumps(payload) + "\n")


def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise SystemExit("no index.json — run: python 04-rag/build_index.py")
    return json.loads(INDEX_PATH.read_text())


def retrieve(index: dict, query: str, k: int = 4) -> list[dict]:
    backend = index["backend"]
    if backend == "tfidf":
        embedder = TfidfEmbedder(index["vocab"], index["idf"])
        qvec = embedder.embed(query)
    else:
        _, vectors, _ = embed_many([query])
        qvec = vectors[0]
    scored = []
    for chunk in index["chunks"]:
        score = cosine(qvec, chunk["embedding"])
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:k]
