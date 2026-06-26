"""
qa_chain.py
-----------
Combines retrieved context chunks with a strict anti-hallucination prompt
and calls the OpenAI Chat Completions API to generate a grounded answer.
"""

from __future__ import annotations

from typing import List

from openai import OpenAI

# Config
CHAT_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a precise assistant that answers questions about the EU Taxonomy.

Rules you MUST follow:
1. Answer ONLY using the provided context passages below. Do NOT use any external knowledge or make any assumptions beyond what is explicitly stated in the context.
2. If the context does not contain enough information to answer the question, respond with exactly: "I don't know"
3. Do not speculate, infer, or extrapolate beyond the context.
4. Keep your answer concise, factual, and directly grounded in the context.
5. If quoting or referencing specific details, make sure they appear verbatim or near-verbatim in the context."""

CONTEXT_TEMPLATE = """Context passages:
---
{context}
---

Question: {question}

Answer:"""


class QAChain:
    """RAG question-answering chain."""

    def __init__(self, model: str = CHAT_MODEL) -> None:
        self.client = OpenAI()
        self.model = model

    def answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate an answer to *question* grounded in *context_chunks*.

        Parameters
        ----------
        question:
            The user's natural-language question.
        context_chunks:
            Ordered list of relevant text passages retrieved from the vector store.

        Returns
        -------
        str
            The model's answer, or "I don't know" if context is insufficient.
        """
        if not context_chunks:
            return "I don't know"

        context = "\n\n".join(
            f"[Passage {i + 1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )
        user_message = CONTEXT_TEMPLATE.format(
            context=context,
            question=question,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,        # deterministic — we want factual answers
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
