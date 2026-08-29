"""
Step 1 of the RAG pipeline: INGESTION.

Turn raw PDF files into clean, overlapping text chunks that we can later embed and
search. This is the "read and chunk documents" deliverable (Phase 2 in your exposé).

Flow:  PDF file  ->  raw text per page  ->  overlapping chunks (LangChain Documents)
"""

from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import DEFAULT_CONFIG, RAGConfig


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Load a single PDF into a list of LangChain `Document` objects (one per page).

    Each Document has:
      - .page_content : the text of that page
      - .metadata     : {'source': <file path>, 'page': <page number>, ...}

    That metadata is what later lets us cite the exact source of an answer.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Normalise metadata so citations read cleanly:
    #  - 'source' becomes just the file name (not the full path)
    #  - 'page' becomes 1-indexed (PyPDF counts from 0, but humans count from 1)
    for p in pages:
        p.metadata["source"] = pdf_path.name
        if isinstance(p.metadata.get("page"), int):
            p.metadata["page"] = p.metadata["page"] + 1
    return pages


def chunk_documents(
    docs: list[Document],
    config: RAGConfig = DEFAULT_CONFIG,
) -> list[Document]:
    """
    Split page-level Documents into smaller, overlapping chunks.

    We use RecursiveCharacterTextSplitter: it tries to split on paragraph breaks
    first, then sentences, then words — so chunks stay semantically coherent
    instead of being cut mid-word. chunk_size and chunk_overlap come from config,
    which is exactly what we vary in the chunking experiment (Research Question 2).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        # Order matters: try these separators from most to least preferred.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # Tag each chunk with a stable id so we can reference/attribute it later.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    config: RAGConfig = DEFAULT_CONFIG,
) -> list[Document]:
    """Convenience: load one PDF and return its chunks in a single call."""
    pages = load_pdf(pdf_path)
    return chunk_documents(pages, config)


def ingest_folder(
    folder: str | Path = None,
    config: RAGConfig = DEFAULT_CONFIG,
) -> list[Document]:
    """Ingest every PDF in a folder (defaults to data/documents/)."""
    from .config import DOCUMENTS_DIR

    folder = Path(folder) if folder else DOCUMENTS_DIR
    all_chunks: list[Document] = []
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {folder}")
    for pdf in pdfs:
        all_chunks.extend(ingest_pdf(pdf, config))
    return all_chunks


# Run this file directly to see chunking in action:
#   python -m src.rag.ingest path\to\file.pdf
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.rag.ingest <path-to-pdf>")
        raise SystemExit(1)

    chunks = ingest_pdf(sys.argv[1])
    print(f"Produced {len(chunks)} chunks from {sys.argv[1]}\n")
    print("--- First chunk preview ---")
    print("metadata:", chunks[0].metadata)
    print("content :", chunks[0].page_content[:300], "...")
