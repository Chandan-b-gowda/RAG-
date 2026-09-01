# Building and Evaluating a RAG System for Accurate Document Question-Answering

A full-stack **Retrieval-Augmented Generation (RAG)** application: upload PDF
documents, ask questions in natural language, and get **source-cited answers**
grounded in the documents. The project also **evaluates** how much RAG helps over a
plain LLM, and how chunking and retrieval settings affect answer quality.

---

## What is RAG (in one paragraph)

A plain LLM answers only from what it memorised during training, so it can produce
confident but wrong answers ("hallucinations") and can't see your private
documents. RAG fixes this by **retrieving** the most relevant passages from your
documents first, then asking the LLM to answer **using only those passages**. The
answer is grounded in real text and can cite exactly where it came from.

```
                      ┌──────────── INGESTION (offline) ────────────┐
   PDF  ──►  extract text  ──►  split into chunks  ──►  embed  ──►  Chroma vector DB
                      └─────────────────────────────────────────────┘

                      ┌──────────── QUERY (online) ─────────────────┐
   question  ──►  embed  ──►  semantic search (top-k)  ──►  grounded prompt  ──►  LLM  ──►  cited answer
                      └─────────────────────────────────────────────┘
```

## Project structure

```
Research project/
├── src/rag/            # the reusable RAG library (the reusable pipeline deliverable)
│   ├── config.py       #   all tunable settings in one place
│   ├── ingest.py       #   load PDFs + split into chunks
│   ├── vectorstore.py  #   embeddings + Chroma store + semantic search
│   ├── generate.py     #   grounded prompt + Gemini answer generation
│   └── pipeline.py     #   ties it together: RAGPipeline.ask()
├── app/streamlit_app.py    # the web application (upload, ask, see sources)
├── experiments/evaluate.py # the research: RAG vs plain LLM, chunking, top-k
├── scripts/smoke_test.py   # verify retrieval works without an API key
├── data/documents/         # put your source PDFs here
├── data/eval/              # your evaluation test set (questions + answers)
└── requirements.txt
```

## Setup

```bash
# 1. Activate the virtual environment (Windows PowerShell)
    .venv\Scripts\Activate.ps1

# 2. Install dependencies (already done during setup)
pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env
# then edit .env and paste your free Gemini key from
# https://aistudio.google.com/app/apikey
```

## Run it

```bash
# A. Verify the retrieval pipeline (no API key needed)
python scripts/smoke_test.py

#for  authentication from the google run this and login with yor pass
gcloud auth application-default login


# B. Launch the web app
streamlit run app/streamlit_app.py

# C. Run the research experiments (needs the Gemini key + PDFs in data/documents/)
python -m experiments.evaluate
```

## Research questions

1. Does RAG improve accuracy / reduce hallucination vs a plain LLM?
2. How do chunk size and overlap affect answer quality?
3. How does the number of retrieved passages (top-k) affect quality?
4. How reliably can the system attribute answers to the correct source?
5. What are the latency/cost trade-offs of adding retrieval?

See [ROADMAP.md](ROADMAP.md) for the week-by-week plan.

## Tech stack

Python · LangChain · Google Gemini · Sentence-Transformers · Chroma · Streamlit ·
pandas/matplotlib · Git/GitHub
