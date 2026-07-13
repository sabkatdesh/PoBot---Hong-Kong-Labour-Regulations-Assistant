"""
retrieval.py — hybrid retrieval, Cohere rerank, and query expansion.
"""
import os
import re
import json
import numpy as np
import cohere
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from indexing import load_indexes, simple_tokenize
from prompts import EXPAND_PROMPT

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

co = cohere.Client(COHERE_API_KEY)
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0.3)

bm25_index, faiss_index, corpus_dicts, embed_model = load_indexes()


def hybrid_retrieve(query: str, top_k: int = 12, alpha: float = 0.5):
    bm25_scores = bm25_index.get_scores(simple_tokenize(query))

    query_vec = embed_model.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        normalize_embeddings=True
    ).astype("float32")
    sem_scores_full, sem_idx_full = faiss_index.search(query_vec, faiss_index.ntotal)
    semantic_scores = np.zeros(len(corpus_dicts))
    for score, idx in zip(sem_scores_full[0], sem_idx_full[0]):
        semantic_scores[idx] = score

    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    bm25_norm = normalize(bm25_scores)
    semantic_norm = normalize(semantic_scores)
    fused_scores = alpha * bm25_norm + (1 - alpha) * semantic_norm

    top_indices = np.argsort(fused_scores)[::-1][:top_k]
    return [{**corpus_dicts[idx], "fused_score": float(fused_scores[idx]),
             "bm25_score": float(bm25_norm[idx]), "semantic_score": float(semantic_norm[idx])}
            for idx in top_indices]


def rerank_chunks(query: str, candidates: list, top_n: int = 4):
    if not candidates:
        return []
    docs = [c["content"] for c in candidates]
    response = co.rerank(model="rerank-english-v3.0", query=query, documents=docs,
                          top_n=min(top_n, len(docs)))
    return [{**candidates[r.index], "rerank_score": r.relevance_score} for r in response.results]


def expand_query_with_history(query: str, history_context: str = "") -> str:
    """Rewrites follow-up questions into standalone queries using conversation context."""
    prompt = EXPAND_PROMPT.format(history_context=history_context or "None", query=query)
    result = llm.invoke(prompt)
    rewritten = result.content.strip()
    return rewritten if rewritten else query