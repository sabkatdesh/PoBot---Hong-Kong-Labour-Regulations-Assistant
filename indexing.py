"""
indexing.py — builds BM25 + FAISS indexes from chunks.jsonl.
Run after cleaning.py. Produces bm25_index.pkl, faiss_index.bin, chunk_lookup.pkl.
"""
import json
import pickle
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def build_indexes(chunks_path="chunks.jsonl"):
    with open(chunks_path, encoding="utf-8") as f:
        corpus_dicts = [json.loads(line) for line in f]
    print(f"Loaded {len(corpus_dicts)} chunks")

    # BM25
    bm25_tokens = [simple_tokenize(c["content"]) for c in corpus_dicts]
    bm25_index = BM25Okapi(bm25_tokens)
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({"index": bm25_index, "tokenized_corpus": bm25_tokens}, f)
    print("BM25 index saved.")

    # FAISS
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    chunk_texts = [c["content"] for c in corpus_dicts]
    embeddings = embed_model.encode(
        chunk_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, "faiss_index.bin")
    print(f"FAISS index saved: {faiss_index.ntotal} vectors, dim={dimension}")

    with open("chunk_lookup.pkl", "wb") as f:
        pickle.dump(corpus_dicts, f)
    print("Chunk lookup saved.")


def load_indexes():
    """Called by retrieval.py to load pre-built indexes."""
    with open("bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    faiss_index = faiss.read_index("faiss_index.bin")
    with open("chunk_lookup.pkl", "rb") as f:
        corpus_dicts = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return bm25_data["index"], faiss_index, corpus_dicts, embed_model


if __name__ == "__main__":
    build_indexes()