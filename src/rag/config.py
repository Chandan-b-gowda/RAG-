from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Research project
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"          # source PDFs live here
EVAL_DIR = DATA_DIR / "eval"                     # evaluation test set all things here are paths
CHROMA_DIR = PROJECT_ROOT / "chroma_db"          # persisted vector database


@dataclass
class RAGConfig:
    
    chunk_size: int = 1000        # characters per chunk
    chunk_overlap: int = 150      # characters shared between neighbouring chunks
   

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2" # it turns out this is the best free embedding model for our use case

    
    top_k: int = 4                # how many chunks to retrieve per question

    
    llm_provider: str = "vertex"
    llm_model: str = "gemini-2.5-flash"   # verified available on Vertex for this project
    
    vertex_location: str = "us-central1"  # Vertex region to call the model in
    temperature: float = 0.0     # 0 = deterministic, factual; higher = more creative

    
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
    
    key = os.getenv("GOOGLE_API_KEY")
    if not key or key == "your-gemini-key-here":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and paste your "
            "Gemini key from https://aistudio.google.com/app/apikey"
        )
    return key


def get_vertex_project() -> str | None:
    
    return os.getenv("GOOGLE_CLOUD_PROJECT")
