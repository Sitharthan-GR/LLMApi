#!/usr/bin/env python3
"""Chunk docs/, embed each chunk, write index.json.

  python 04-rag/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raglib import chunk_text, embed_many, load_docs, save_index


def main() -> None:
    records: list[dict] = []
    texts: list[str] = []
    for source, body in load_docs():
        for i, piece in enumerate(chunk_text(body)):
            records.append({"source": source, "chunk": i, "text": piece})
            texts.append(piece)

    backend, vectors, embedder = embed_many(texts)
    for record, vector in zip(records, vectors):
        record["embedding"] = vector

    payload: dict = {"backend": backend, "chunks": records}
    if embedder is not None:
        payload["vocab"] = embedder.vocab
        payload["idf"] = embedder.idf

    save_index(payload)
    print(f"backend={backend}  chunks={len(records)}  sources={sorted({r['source'] for r in records})}")
    print("wrote 04-rag/index.json")
    if backend == "tfidf":
        print("using local TF-IDF (Groq chat key has no embeddings model on this account).")
        print("optional HTTP: set LLM_EMBED_URL and LLM_EMBED_MODEL, then rebuild.")


if __name__ == "__main__":
    main()
