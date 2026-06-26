from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from ingest import run_ingestion, INDEX_PATH
from retriever import Retriever
from qa_chain import QAChain

app = FastAPI(
    title="EU Taxonomy RAG API",
    description="Ask questions about the EU Taxonomy. Answers are grounded in the official FAQ document only.",
    version="1.0.0",
)

_retriever: Retriever | None = None
_qa: QAChain | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global _retriever, _qa

    if not INDEX_PATH.exists():
        print("[api] Vector store not found — running ingestion …")
        run_ingestion()

    _retriever = Retriever(top_k=int(os.getenv("TOP_K", "5")))
    _qa = QAChain()
    print("[api] Ready.")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Your question about the EU Taxonomy")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    include_context: bool = Field(False, description="Include retrieved passages in the response")


class QuestionResponse(BaseModel):
    question: str
    answer: str
    context_chunks: list[str] | None = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "vector_store_loaded": _retriever is not None}


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest) -> QuestionResponse:
    if _retriever is None or _qa is None:
        raise HTTPException(status_code=503, detail="Service not ready yet.")

    # Override top_k per request if caller specified it
    _retriever.top_k = request.top_k
    chunks = _retriever.retrieve(request.question)
    answer = _qa.answer(request.question, chunks)

    return QuestionResponse(
        question=request.question,
        answer=answer,
        context_chunks=chunks if request.include_context else None,
    )
