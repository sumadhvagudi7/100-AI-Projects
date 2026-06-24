"""Search helpers for the semantic search app."""

from functools import lru_cache
from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "faiss_index.bin"
MAPPING_FILE = BASE_DIR / "doc_mapping.pkl"


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Missing FAISS index: {INDEX_FILE}. Run embeddings.py first."
        )

    return faiss.read_index(str(INDEX_FILE))


@lru_cache(maxsize=1)
def get_metadata():
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Missing document mapping: {MAPPING_FILE}. Run embeddings.py first."
        )

    with open(MAPPING_FILE, "rb") as f:
        return pickle.load(f)


def search_documents(query, top_k=5):
    model = get_model()
    index = get_index()
    metadata = get_metadata()

    query_embedding = model.encode([query])

    distances, indices = index.search(
        query_embedding.astype("float32"), top_k
    )

    results = []

    for idx, score in zip(indices[0], distances[0]):
        results.append(
            {
                "filename": metadata["filenames"][idx],
                "document": metadata["documents"][idx],
                "distance": float(score),
            }
        )

    return results
