# app.py
# author : Sumadhva Anand Gudi

import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "faiss_index.bin"
MAPPING_FILE = BASE_DIR / "doc_mapping.pkl"

st.set_page_config(
    page_title="Semantic Search Engine",
    layout="wide"
)

from embeddings import create_embeddings
from search import search_documents


# Create embeddings if they don't exist
if not INDEX_FILE.exists() or not MAPPING_FILE.exists():
    create_embeddings()
    st.session_state.embeddings_created = True

st.title("Semantic Search Engine")

query = st.text_input(
    "Enter your search query"
)

if st.button("Search"):

    if query:

        results = search_documents(
            query,
            top_k=5
        )

        st.subheader("Top 5 Matches")

        for i, result in enumerate(results, start=1):

            st.markdown(
                f"### {i}. {result['filename']}"
            )

            st.write(
                result["document"][:500]
            )

            st.write(
                f"Distance Score: {result['distance']:.4f}"
            )

            st.divider()
