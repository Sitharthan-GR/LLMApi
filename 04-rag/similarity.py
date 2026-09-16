#!/usr/bin/env python3
"""Week 4: cosine similarity by hand. No chat model.

  python 04-rag/similarity.py
"""

from __future__ import annotations

from raglib import TfidfEmbedder, cosine

SENTENCES = [
    "A token is a chunk of text the model reads and writes.",
    "LLM APIs bill by token counts, not by words.",
    "Neapolitan pizza is topped with tomato and mozzarella.",
]


def main() -> None:
    embedder = TfidfEmbedder.fit(SENTENCES)
    vectors = [embedder.embed(s) for s in SENTENCES]
    print("query:", SENTENCES[0])
    print()
    for i, sentence in enumerate(SENTENCES):
        score = cosine(vectors[0], vectors[i])
        print(f"  {score:.3f}  {sentence}")
    print()
    print("Closer to 1 = more similar. The pizza line should score much lower.")


if __name__ == "__main__":
    main()
