from __future__ import annotations

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import DEFAULT_CONFIG, RAGConfig


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load a PDF and return a list of page-level Documents."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()


    for p in pages:
        p.metadata["source"] = pdf_path.name 
        if isinstance(p.metadata.get("page"), int): 
            p.metadata["page"] = p.metadata["page"] + 1 # PyPDF counts from 0, but humans count from 1
    return pages


def chunk_documents(
    docs: list[Document],
    config: RAGConfig = DEFAULT_CONFIG,
) -> list[Document]:
    """Split a list of Documents into smaller chunks for retrieval."""
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
