from typing import List

import numpy as np
from openai import OpenAI

from ingest import load_index, embed_texts


class Retriever:
    """Semantic retriever backed by a local FAISS index."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.client = OpenAI()
        self.index, self.chunks = load_index()
        print(f"[retriever] Loaded index with {self.index.ntotal} vectors "
              f"(top_k={self.top_k})")

    def retrieve(self, query: str) -> List[str]:
        """
        Embed *query* and return the top-k most similar chunks.

        Returns a list of strings ordered from most to least similar.
        """
        query_vec = embed_texts([query], self.client)          # (1, dim)
        _, indices = self.index.search(query_vec, self.top_k)

        results: List[str] = []
        for idx in indices[0]:
            if idx != -1:                                       # -1 = no result
                results.append(self.chunks[idx])
        return results
