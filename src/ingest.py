"""
ingest.py
---------
Loads the FAQ markdown, splits it into overlapping token chunks,
generates embeddings via OpenAI, and persists a FAISS index to disk.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
import tiktoken
from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072          # dimensionality of text-embedding-3-large
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 50

VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.pkl"


# ── Tokeniser helper ───────────────────────────────────────────────────────────
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ── Text loading ───────────────────────────────────────────────────────────────
def load_markdown(path: str | Path) -> str:
    """Read a markdown file and return its text."""
    return Path(path).read_text(encoding="utf-8")


# ── Chunking ───────────────────────────────────────────────────────────────────
def split_into_chunks(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[str]:
    """
    Split *text* into overlapping windows measured in tokens.

    Strategy:
    1. Tokenise the whole document.
    2. Slide a window of `chunk_tokens` tokens forward by
       (chunk_tokens - overlap_tokens) tokens each step.
    3. Decode each window back to a string.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    step = max(1, chunk_tokens - overlap_tokens)
    chunks: List[str] = []

    for start in range(0, len(tokens), step):
        end = start + chunk_tokens
        chunk_tokens_slice = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens_slice).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break

    return chunks


# ── Embeddings ─────────────────────────────────────────────────────────────────
def embed_texts(texts: List[str], client: OpenAI) -> np.ndarray:
    """
    Embed a list of strings using the OpenAI Embeddings API.
    Returns a float32 numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ── FAISS persistence ──────────────────────────────────────────────────────────
def build_and_save_index(
    chunks: List[str],
    embeddings: np.ndarray,
) -> None:
    """Build a flat L2 FAISS index and persist it alongside the raw chunks."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[ingest] Saved {index.ntotal} vectors → {INDEX_PATH}")
    print(f"[ingest] Saved {len(chunks)} chunks   → {CHUNKS_PATH}")


def load_index() -> tuple[faiss.Index, List[str]]:
    """Load a previously built FAISS index and chunk list from disk."""
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run `python src/ingest.py` first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks: List[str] = pickle.load(f)
    return index, chunks


# Main entry point
def run_ingestion(
    faq_path: str | Path = "data/faq.md",
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> None:
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    print(f"[ingest] Loading {faq_path} …")
    text = load_markdown(faq_path)
    print(f"[ingest] Document: {count_tokens(text)} tokens")

    chunks = split_into_chunks(text, chunk_tokens, overlap_tokens)
    print(f"[ingest] Created {len(chunks)} chunks "
          f"(~{chunk_tokens} tok, {overlap_tokens} tok overlap)")

    print("[ingest] Embedding chunks …")
    embeddings = embed_texts(chunks, client)

    build_and_save_index(chunks, embeddings)
    print("[ingest] Done.")


if __name__ == "__main__":
    run_ingestion()
