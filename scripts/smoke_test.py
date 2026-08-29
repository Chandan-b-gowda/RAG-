"""
A no-API-key smoke test for the retrieval half of the pipeline.

It creates a tiny in-memory document, chunks it, embeds it, stores it in Chroma,
and runs a semantic search. If this prints relevant passages, then ingestion +
embeddings + vector search all work — WITHOUT needing a Gemini key yet.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.rag.config import RAGConfig
from src.rag.ingest import chunk_documents
from src.rag.vectorstore import build_vectorstore, similarity_search

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) combines a retriever with a language model.
The retriever finds relevant passages from a document collection using semantic
search over embeddings. Those passages are then given to the language model as
context, so the model answers based on real source material rather than only its
internal memory. This reduces hallucination and lets the system cite its sources.

Chunking splits long documents into smaller overlapping pieces. Chunk size and
overlap are important settings: chunks that are too large dilute relevance, while
chunks that are too small can lose context. Embeddings turn each chunk into a
vector, and a vector database such as Chroma stores these vectors for fast search.
"""


def main() -> None:
    print("1) Building a tiny document and chunking it...")
    config = RAGConfig(chunk_size=300, chunk_overlap=50, top_k=2)
    docs = [Document(page_content=SAMPLE_TEXT, metadata={"source": "sample.txt", "page": 1})]
    chunks = chunk_documents(docs, config)
    print(f"   -> {len(chunks)} chunks created")

    print("2) Loading the embedding model (first run downloads ~90 MB)...")
    store = build_vectorstore(chunks, config, persist=False)
    print("   -> embeddings created and stored in Chroma (in-memory)")

    query = "How does chunk size affect retrieval?"
    print(f"3) Semantic search for: {query!r}")
    results = similarity_search(store, query, config)
    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n   Result {i} (relevance {score:.3f}):")
        print("   " + doc.page_content.strip().replace("\n", " ")[:200])

    print("\n[OK] Retrieval pipeline works. Next step: add your Gemini key to answer questions.")


if __name__ == "__main__":
    main()
