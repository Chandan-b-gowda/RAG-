"""
The RAG pipeline, assembled.

This ties ingestion + retrieval + generation into one object with a simple `.ask()`
method. The Streamlit app and the experiment scripts both use this class, so there
is a single, consistent definition of "how the system answers a question".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .config import DEFAULT_CONFIG, RAGConfig
from .generate import generate_answer, generate_plain_answer
from .ingest import ingest_folder, ingest_pdf
from .vectorstore import (
    build_vectorstore,
    load_vectorstore,
    similarity_search,
)


@dataclass
class RAGAnswer:
    """Everything produced by one question — the answer plus evidence and timing."""

    question: str
    answer: str
    sources: list[Document] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    latency_seconds: float = 0.0

    def citations(self) -> list[dict]:
        """A compact, display-friendly list of the sources used."""
        out = []
        for i, (doc, score) in enumerate(zip(self.sources, self.scores), start=1):
            out.append(
                {
                    "marker": f"[{i}]",
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "score": round(float(score), 3),
                    "text": doc.page_content,
                }
            )
        return out


class RAGPipeline:
    """A ready-to-query RAG system over an already-built vector store."""

    def __init__(self, store, config: RAGConfig = DEFAULT_CONFIG):
        self.store = store
        self.config = config

    # ---- Constructors ----
    @classmethod
    def from_existing(cls, config: RAGConfig = DEFAULT_CONFIG) -> "RAGPipeline":
        """Load a previously persisted vector store from disk."""
        return cls(load_vectorstore(config), config)

    @classmethod
    def from_folder(cls, folder=None, config: RAGConfig = DEFAULT_CONFIG) -> "RAGPipeline":
        """
        Ingest all PDFs in a folder and build a fresh IN-MEMORY vector store.

        In-memory (persist=False) keeps each build fully isolated — essential for
        experiments that rebuild the store many times with different settings — and
        avoids Windows file-locking when the store is rebuilt within one process.
        """
        chunks = ingest_folder(folder, config)
        return cls(build_vectorstore(chunks, config, persist=False), config)

    @classmethod
    def from_pdf(cls, pdf_path, config: RAGConfig = DEFAULT_CONFIG) -> "RAGPipeline":
        """Ingest a single PDF and build a fresh in-memory vector store."""
        chunks = ingest_pdf(pdf_path, config)
        return cls(build_vectorstore(chunks, config, persist=False), config)

    # ---- Querying ----
    def ask(self, question: str) -> RAGAnswer:
        """Full RAG: retrieve relevant chunks, then generate a grounded answer."""
        start = time.perf_counter()
        results = similarity_search(self.store, question, self.config)
        docs = [doc for doc, _ in results]
        scores = [score for _, score in results]
        answer = generate_answer(question, docs, self.config)
        latency = time.perf_counter() - start
        return RAGAnswer(
            question=question,
            answer=answer,
            sources=docs,
            scores=scores,
            latency_seconds=latency,
        )

    def ask_plain(self, question: str) -> RAGAnswer:
        """Baseline: answer with no retrieval (for the RAG-vs-plain comparison)."""
        start = time.perf_counter()
        answer = generate_plain_answer(question, self.config)
        latency = time.perf_counter() - start
        return RAGAnswer(question=question, answer=answer, latency_seconds=latency)
