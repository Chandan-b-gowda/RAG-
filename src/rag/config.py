"""
Central configuration for the RAG system.

Every tunable knob lives here in ONE place. This matters for the research part of
the project: when we run experiments (chunk size, top-k, etc.), we change a value
here (or override it in code) instead of hunting through the codebase. That makes
experiments reproducible — a core requirement of good research.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file (API keys) into the environment.
load_dotenv()

# ---- Project paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Research project
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"          # source PDFs live here
EVAL_DIR = DATA_DIR / "eval"                     # evaluation test set
CHROMA_DIR = PROJECT_ROOT / "chroma_db"          # persisted vector database


@dataclass
class RAGConfig:
    """All tunable RAG settings. Change these to run experiments."""

    # --- Chunking (Research Question 2) ---
    # A "chunk" is a small slice of a document. We split documents because an LLM
    # can only read a limited amount of text at once, and because retrieval works
    # better on focused passages than on whole documents.
    chunk_size: int = 1000        # characters per chunk
    chunk_overlap: int = 150      # characters shared between neighbouring chunks
    # Overlap prevents a sentence that straddles a boundary from being cut in half.

    # --- Embeddings ---
    # An embedding turns text into a vector (list of numbers) that captures meaning.
    # Similar meanings -> nearby vectors. This local model is free and runs offline.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval (Research Question 3) ---
    top_k: int = 4                # how many chunks to retrieve per question

    # --- Generation ---
    # Provider decides HOW we reach Gemini:
    #   "vertex"     -> Google Cloud Vertex AI (auth via gcloud login; uses GCP
    #                   credits). This is what this project uses.
    #   "gemini_api" -> Google AI Studio Developer API (auth via a simple API key).
    llm_provider: str = "vertex"
    llm_model: str = "gemini-2.5-flash"   # verified available on Vertex for this project
    # Pinned to a specific version (not "-latest") so experiment results stay
    # reproducible.
    vertex_location: str = "us-central1"  # Vertex region to call the model in
    temperature: float = 0.0     # 0 = deterministic, factual; higher = more creative

    # --- Vector store ---
    collection_name: str = "documents"

    def summary(self) -> str:
        """One-line description — handy for labelling experiment runs."""
        return (
            f"chunk={self.chunk_size}/{self.chunk_overlap} "
            f"top_k={self.top_k} model={self.llm_model}"
        )


# A ready-to-use default config instance.
DEFAULT_CONFIG = RAGConfig()


def get_google_api_key() -> str:
    """Return the Gemini API key, with a friendly error if it's missing.

    Only needed when llm_provider == "gemini_api".
    """
    key = os.getenv("GOOGLE_API_KEY")
    if not key or key == "your-gemini-key-here":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and paste your "
            "Gemini key from https://aistudio.google.com/app/apikey"
        )
    return key


def get_vertex_project() -> str | None:
    """
    Return the Google Cloud project id for Vertex AI.

    Read from the GOOGLE_CLOUD_PROJECT env var if set; otherwise return None and
    let the Vertex SDK fall back to the project configured via `gcloud`.
    """
    return os.getenv("GOOGLE_CLOUD_PROJECT")
