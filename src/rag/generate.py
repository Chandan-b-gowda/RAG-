"""
Step 3 of the RAG pipeline: GROUNDED GENERATION.

We take the chunks retrieved by semantic search, insert them into a carefully
worded prompt, and ask the LLM (Gemini) to answer *using only those chunks*. This
"grounding" is what reduces hallucination and lets us cite sources — the heart of
Research Question 1 and Objective 3.3 in your exposé.
"""

from __future__ import annotations

from langchain_core.documents import Document

from .config import (
    DEFAULT_CONFIG,
    RAGConfig,
    get_google_api_key,
    get_vertex_project,
)
from .llm_utils import invoke_with_backoff

# The prompt is the single most important piece of engineering in a RAG system.
# We instruct the model to (a) use ONLY the provided context, (b) say when it
# doesn't know, and (c) cite the source of each claim by its [n] marker.
GROUNDED_PROMPT = """You are a careful assistant that answers questions using ONLY \
the context provided below. The context consists of numbered passages extracted \
from the user's documents.

Rules:
- Answer using ONLY information found in the context. Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly: \
"I could not find this in the provided documents."
- Cite the passage number(s) you used in square brackets, e.g. [1] or [2][3].
- Be concise and factual.

Context:
{context}

Question: {question}

Answer (with citations):"""


def format_context(chunks: list[Document]) -> str:
    """Number each chunk so the model (and the user) can cite it as [1], [2], ..."""
    blocks = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{i}] (source: {source}, page {page})\n{doc.page_content}")
    return "\n\n".join(blocks)


def get_llm(config: RAGConfig = DEFAULT_CONFIG):
    """
    Create the Gemini chat model for the configured provider.

    - "vertex":     Vertex AI. Authenticates through your gcloud login (Application
                    Default Credentials); no API key. Billed to your GCP project,
                    so it draws on your Google Cloud credits.
    - "gemini_api": AI Studio Developer API. Requires GOOGLE_API_KEY in your .env.
    """
    if config.llm_provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model=config.llm_model,
            temperature=config.temperature,
            project=get_vertex_project(),      # None -> use gcloud's default project
            location=config.vertex_location,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=config.temperature,
        google_api_key=get_google_api_key(),
    )


def generate_answer(
    question: str,
    chunks: list[Document],
    config: RAGConfig = DEFAULT_CONFIG,
) -> str:
    """
    Given a question and the retrieved chunks, produce a grounded, cited answer.
    This is the RAG path: the model sees real passages from the documents.
    """
    llm = get_llm(config)
    prompt = GROUNDED_PROMPT.format(
        context=format_context(chunks),
        question=question,
    )
    return invoke_with_backoff(llm, prompt)


def generate_plain_answer(
    question: str,
    config: RAGConfig = DEFAULT_CONFIG,
) -> str:
    """
    The BASELINE path (no retrieval): ask the LLM the question directly, with no
    document context. We compare this against the RAG answer in the experiments to
    measure how much retrieval actually helps (Research Question 1).
    """
    llm = get_llm(config)
    return invoke_with_backoff(llm, question)
