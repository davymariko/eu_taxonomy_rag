# EU Taxonomy RAG

A minimal, production-quality **Retrieval-Augmented Generation (RAG)** application that answers questions about the EU Taxonomy using only an official FAQ document as its knowledge source.

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐     embed query      ┌──────────────────────┐
│   CLI/API   │ ──────────────────▶  │  OpenAI Embeddings   │
│  (app.py /  │                      │ text-embedding-3-large│
│   api.py)   │                      └──────────┬───────────┘
└─────────────┘                                 │ query vector
                                                ▼
                                    ┌───────────────────────┐
                                    │   FAISS Index (local) │
                                    │   vectorstore/        │
                                    └──────────┬────────────┘
                                               │ top-k chunks
                                               ▼
                              ┌────────────────────────────────┐
                              │         QA Chain               │
                              │  System: anti-hallucination    │
                              │  prompt + context passages     │
                              │  Model: GPT-4o                 │
                              └───────────────┬────────────────┘
                                              │ grounded answer
                                              ▼
                                         User / API
```

### Module breakdown

| File | Responsibility |
|---|---|
| `src/ingest.py` | Load markdown → token-chunk → embed → persist FAISS index |
| `src/retriever.py` | Embed query → FAISS search → return top-k text chunks |
| `src/qa_chain.py` | Build strict prompt → call GPT-4o → return grounded answer |
| `src/app.py` | Interactive CLI |
| `src/api.py` | FastAPI REST endpoint (optional) |

---

## Setup — Local

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone and install

```bash
git clone <repo-url>
cd eu-taxonomy-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create and edit .env and set OPENAI_API_KEY=sk-...

### 3. Ingest the FAQ

```bash
python src/ingest.py
```

This reads `data/faq.md`, splits it into overlapping chunks (~400 tokens, 50-token overlap), embeds them with `text-embedding-3-large`, and saves a FAISS index to `vectorstore/`.

### 4. Ask questions

```bash
python src/app.py

# Options
python src/app.py --top-k 7          # retrieve 7 chunks
python src/app.py --show-context     # print retrieved passages
python src/app.py --ingest           # re-ingest before starting
```

### 5. (Optional) FastAPI

```bash
uvicorn src.api:app --reload
# → http://localhost:8000/docs
```

Example request:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Green Asset Ratio?", "top_k": 5}'
```

---

## Setup — Docker

### Build and ingest

```bash
cp .env.example .env   # set your OPENAI_API_KEY

# Ingest (one-off)
docker compose --profile ingest up ingest

# Start the API server
docker compose up api

# Interactive CLI
docker compose run cli
```

The `vectorstore/` data is persisted in the `vectorstore` Docker volume across restarts.

---

## Design Decisions

### Chunking strategy
Token-aware sliding window (via `tiktoken`) rather than character or sentence splitting. This guarantees chunks stay within the embedding model's context window and avoids cutting mid-sentence arbitrarily. Overlap (default 50 tokens) preserves context across chunk boundaries.

### Embedding model
`text-embedding-3-large` (3072 dimensions) was chosen for maximum retrieval quality. `text-embedding-3-small` is a cheaper alternative if cost is a concern — change `EMBEDDING_MODEL` and `EMBEDDING_DIM` in `ingest.py`.

### Vector store
FAISS `IndexFlatL2` (exact nearest-neighbour, L2 distance). For a document this size (~5k tokens total, <30 chunks) exact search is instant and requires no tuning. For larger corpora, consider `IndexIVFFlat` or `IndexHNSWFlat`.

### Anti-hallucination prompt
The system prompt explicitly forbids the model from using external knowledge and instructs it to return `"I don't know"` if context is insufficient. `temperature=0.0` further reduces creative drift.

---

## Future Improvements

1. **Hybrid search**: Combine dense (FAISS) retrieval with sparse BM25 for better recall on keyword-heavy queries.
2. **Re-ranking**: Add a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM`) to re-score the top-k chunks before passing them to the LLM.
3. **Streaming**: Stream the LLM response in the CLI and API for a better UX.
4. **Metadata filtering**: Tag chunks with section headers (e.g. "Disclosure Requirements") and allow filtered retrieval.
5. **Caching**: Cache embedding calls for repeated queries to reduce cost.
6. **Evaluation dataset**: Expand `GROUND_TRUTH` in `evaluation.py` for more robust experiments.
7. **Chroma alternative**: Swap FAISS for Chroma for built-in persistence and metadata support without custom pickling.
8. **Async API**: Use `async` OpenAI client in `api.py` for higher throughput.
