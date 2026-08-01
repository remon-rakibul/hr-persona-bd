#!/usr/bin/env python3
"""
Retrieval-augmented (RAG) baseline over the Bangladesh Labour Act PDF.

Non-fine-tuned comparison point: chunk the Act, embed chunks with a local
embedding model (nomic-embed-text via Ollama), and at query time retrieve the
top-k chunks and prompt a generator model (base Llama, or the fine-tuned model
for the RAG+fine-tune condition) to answer grounded in the retrieved text.

The index (chunks + embeddings) is cached to disk so it is built once and reused
across all evaluation runs. Retrieval is exact cosine similarity over ~700
chunks (instant), avoiding an external vector-DB dependency.

Usage:
    # Build the index once
    python scripts/rag_baseline.py --build \
        --pdf data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf

    # Try a query
    python scripts/rag_baseline.py --query "How many weeks of maternity leave?" \
        --gen-model llama3.2:3b-instruct-q4_K_M
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = ROOT / "rag_index"
DEFAULT_PDF = str(ROOT / "data/input/Bangladesh-Labour-Act-2006_English-Upto-2018.pdf")

RAG_SYSTEM = (
    "You are an HR assistant answering strictly from the provided excerpts of "
    "the Bangladesh Labour Act 2006. Use ONLY the excerpts below. If the answer "
    "is not in them, say you cannot find it in the provided text. Cite the "
    "relevant section number when present."
)


def _embed(texts):
    import ollama
    vecs = []
    for t in texts:
        e = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(e["embedding"])
    return np.array(vecs, dtype=np.float32)


def chunk_text(text, size=800, overlap=100):
    text = re.sub(r"[ \t]+", " ", text)
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i + size].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        i += size - overlap
    return chunks


def build_index(pdf=DEFAULT_PDF, size=800, overlap=100):
    from pdfminer.high_level import extract_text
    print(f"Extracting {pdf} ...")
    text = extract_text(pdf)
    chunks = chunk_text(text, size, overlap)
    print(f"{len(chunks)} chunks; embedding with {EMBED_MODEL} ...")
    emb = _embed(chunks)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", emb)
    json.dump(chunks, open(INDEX_DIR / "chunks.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({"model": EMBED_MODEL, "size": size, "overlap": overlap,
               "n_chunks": len(chunks)},
              open(INDEX_DIR / "meta.json", "w"))
    print(f"Index saved to {INDEX_DIR}/")


class RagRetriever:
    def __init__(self, index_dir=INDEX_DIR):
        self.emb = np.load(Path(index_dir) / "embeddings.npy")
        self.chunks = json.load(open(Path(index_dir) / "chunks.json", encoding="utf-8"))

    def retrieve(self, query, k=4):
        q = _embed([query])[0]
        q /= (np.linalg.norm(q) + 1e-8)
        sims = self.emb @ q
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]


def rag_answer(query, gen_model, retriever, k=4, temperature=0.0, num_predict=384,
               seed=3407):
    """Answer `query` from retrieved Act excerpts.

    Greedy and seeded by default so evaluation runs are reproducible.
    """
    import ollama
    hits = retriever.retrieve(query, k)
    context = "\n\n".join(f"[Excerpt {i+1}] {c}" for i, (c, _) in enumerate(hits))
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {query}"},
    ]
    r = ollama.chat(model=gen_model, messages=messages,
                    options={"temperature": temperature, "num_predict": num_predict,
                             "seed": seed})
    return r["message"]["content"], [c for c, _ in hits]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--query")
    ap.add_argument("--gen-model", default="llama3.2:3b-instruct-q4_K_M")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    if args.build:
        build_index(args.pdf, args.size, args.overlap)
    if args.query:
        r = RagRetriever()
        ans, ctx = rag_answer(args.query, args.gen_model, r, k=args.k)
        print("\n=== Retrieved (top of each excerpt) ===")
        for i, c in enumerate(ctx):
            print(f"[{i+1}] {c[:120]}...")
        print("\n=== Answer ===\n", ans)


if __name__ == "__main__":
    main()
