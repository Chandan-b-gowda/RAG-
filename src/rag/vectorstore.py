"""
Step 2 of the RAG pipeline: EMBED + STORE + SEARCH.

We convert each text chunk into an embedding vector and store it in Chroma, a
vector database. Later, to answer a question, we embed the question and ask Chroma
for the chunks whose vectors are closest — that's "semantic search". Unlike keyword
search, it finds passages that *mean* the same thing even if they use different words.

This file covers the "generate embeddings; store and search in Chroma" deliverable.
"""

from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHROMA_DIR, DEFAULT_CONFIG, RAGConfig

# Cache the embedding model so we don't reload it (it's a few hundred MB) each call.
_embedding_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(config: RAGConfig = DEFAULT_CONFIG) -> HuggingFaceEmbeddings:
    """
    Return the embedding model. Uses Sentence-Transformers, which runs locally:
    free, private, offline, and reproducible — good properties for research.
    The first call downloads the model (~90 MB) and caches it on disk.
    """
    name = config.embedding_model
    if name not in _embedding_cache:
        _embedding_cache[name] = HuggingFaceEmbeddings(
            model_name=name,
            # Normalising vectors makes cosine similarity behave nicely.
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_cache[name]


def build_vectorstore(
    chunks: list[Document],
    config: RAGConfig = DEFAULT_CONFIG,
    persist: bool = True,
) -> Chroma:
    """
    Embed all chunks and store them in a fresh Chroma collection.

    If persist=True the database is written to disk (chroma_db/) so we don't have
    to re-embed every time the app restarts. We always wipe any existing persisted
    store first, so a rebuild never leaves stale or duplicated chunks behind — this
    keeps experiments isolated: each chunk/retrieval config starts from a clean slate.
    """
    if persist and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = get_embeddings(config)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.collection_name,
        persist_directory=str(CHROMA_DIR) if persist else None,
        # Use cosine similarity to match our normalized embeddings. This makes the
        # relevance scores meaningful (roughly 0..1) instead of Chroma's default
        # L2 distance, which produced out-of-range scores and a warning.
        collection_metadata={"hnsw:space": "cosine"},
    )
    return store


def load_vectorstore(config: RAGConfig = DEFAULT_CONFIG) -> Chroma:
    """Re-open a previously persisted Chroma database from disk."""
    embeddings = get_embeddings(config)
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def similarity_search(
    store: Chroma,
    query: str,
    config: RAGConfig = DEFAULT_CONFIG,
) -> list[tuple[Document, float]]:
    """
    Return the top-k most relevant chunks for a query, each with a similarity score.
    top_k comes from config — this is the knob we vary in Research Question 3.
    """
    return store.similarity_search_with_relevance_scores(query, k=config.top_k)
