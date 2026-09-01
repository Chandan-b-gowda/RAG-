

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the `src` package importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.rag.config import RAGConfig  # noqa: E402
from src.rag.ingest import chunk_documents, load_pdf  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.rag.vectorstore import build_vectorstore  # noqa: E402

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")

st.title("📄 Document Question-Answering with RAG")
st.caption(
    "Upload PDFs, ask questions, and get answers grounded in your documents "
    "— with citations to the exact source passages."
)

# ---- Sidebar: settings (these are the same knobs from config.py) ----
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 50)
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 4, 1)
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Your PDF is split into overlapping chunks.\n"
        "2. Each chunk becomes an embedding vector.\n"
        "3. Your question retrieves the closest chunks.\n"
        "4. Gemini answers using only those chunks."
    )

config = RAGConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)

# ---- Document upload + indexing ----
uploaded = st.file_uploader(
    "Upload one or more PDF documents", type="pdf", accept_multiple_files=True
)

if uploaded and st.button("📥 Build knowledge base", type="primary"):
    with st.spinner("Reading, chunking, and embedding your documents..."):
        all_chunks = []
        for f in uploaded:
            # Streamlit gives us bytes; write to a temp file so PyPDFLoader can read it.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            pages = load_pdf(tmp_path)
            for p in pages:
                p.metadata["source"] = f.name  # keep the real filename for citations
            all_chunks.extend(chunk_documents(pages, config))

        store = build_vectorstore(all_chunks, config, persist=False)
        st.session_state.pipeline = RAGPipeline(store, config)
        st.session_state.num_chunks = len(all_chunks)
        st.session_state.num_docs = len(uploaded)
    st.success(
        f"Indexed {st.session_state.num_docs} document(s) into "
        f"{st.session_state.num_chunks} chunks. Ask away below!"
    )

# ---- Ask a question ----
if "pipeline" in st.session_state:
    st.divider()
    question = st.text_input("Ask a question about your documents:")

    col1, col2 = st.columns([1, 1])
    ask_rag = col1.button("🔍 Answer with RAG", type="primary")
    ask_plain = col2.button("🧠 Answer without retrieval (baseline)")

    if question and (ask_rag or ask_plain):
        pipeline = st.session_state.pipeline
        with st.spinner("Thinking..."):
            result = pipeline.ask(question) if ask_rag else pipeline.ask_plain(question)

        st.subheader("Answer")
        st.write(result.answer)
        st.caption(f"⏱️ {result.latency_seconds:.2f}s")

        if ask_rag and result.citations():
            st.subheader("Sources")
            for c in result.citations():
                with st.expander(
                    f"{c['marker']} {c['source']} — page {c['page']} "
                    f"(relevance {c['score']})"
                ):
                    st.write(c["text"])
else:
    st.info("👆 Upload PDF(s) and click **Build knowledge base** to get started.")
