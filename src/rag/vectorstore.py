from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHROMA_DIR, DEFAULT_CONFIG, RAGConfig

# Cache the embedding model so we don't reload it (it's a few hundred MB) each call.
_embedding_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(config: RAGConfig = DEFAULT_CONFIG) -> HuggingFaceEmbeddings:

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
    return store.similarity_search_with_relevance_scores(query, k=config.top_k)
