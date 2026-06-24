"""Create and persist embeddings for local documents."""

import os
from pathlib import Path
import pickle

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "docs"
INDEX_FILE = BASE_DIR / "faiss_index.bin"
MAPPING_FILE = BASE_DIR / "doc_mapping.pkl"


def load_documents():
    documents = []
    filenames = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            filepath = DOCS_PATH / file

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append(text)
            filenames.append(file)

    return documents, filenames


def create_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents, filenames = load_documents()

    if not documents:
        raise ValueError(f"No .txt documents found in {DOCS_PATH}")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    faiss.write_index(index, str(INDEX_FILE))

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(
            {
                "documents": documents,
                "filenames": filenames
            },
            f
        )

    print(f"Indexed {len(documents)} documents")


if __name__ == "__main__":
    create_embeddings()
