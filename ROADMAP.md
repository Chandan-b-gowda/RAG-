# 3-Week Sprint Plan

Your exposé planned 12 weeks; this compresses it into ~3 weeks while still hitting
every deliverable. Each phase lists what to **build**, what to **learn**, and the
**"done" check**. Tick items as you go.

> How a research project flows (keep this in mind): **motivation → research
> questions → method → build the artefact → run controlled experiments → analyse
> results → write up honestly (including failures) → reflect.** The build serves the
> questions; the questions are answered by the experiments; the report tells that
> story.

---

## Week 1 — Foundations + working RAG pipeline

**Build**
- [x] Environment: Python 3.12 venv, project scaffold, git repo
- [x] Retrieval library: ingest → chunk → embed → Chroma → search (`src/rag/`)
- [ ] Install dependencies and pass `scripts/smoke_test.py`
- [ ] Get a free Gemini API key; put it in `.env`
- [ ] First end-to-end grounded answer (retrieval → Gemini → citation)

**Learn** (concepts, ~1–2 hrs each, as you build)
- Embeddings & semantic search — why vectors capture meaning
- Chunking — why we split, and the size/overlap trade-off
- Grounded prompting — how the prompt forces the model to use only the context

**Done when:** you can ask a question about a PDF and get a cited answer.

---

## Week 2 — Full-stack app + evaluation test set

**Build**
- [ ] Streamlit app: multi-PDF upload, ask, show answer + source passages
- [ ] Polish: latency display, "not found" handling, RAG-vs-baseline toggle
- [ ] Curate the evaluation test set (`data/eval/testset.json`): 15–25 questions
      with known correct answers, drawn from your chosen documents
- [ ] Deploy to Streamlit Cloud (free) — or keep local if time is tight

**Learn**
- How an LLM-as-judge evaluation works, and its limitations
- What makes a fair test set (coverage, difficulty, no leakage)

**Done when:** the app runs and you have a real, documented test set.

---

## Week 3 — Experiments, analysis, report, demo

**Build / Run**
- [ ] RQ1: RAG vs plain LLM (`experiments/evaluate.py`)
- [ ] RQ2: chunking sweep (500/1000/1500)
- [ ] RQ3: top-k sweep (2/4/8)
- [ ] RQ4: manually check source-attribution correctness on a sample
- [ ] Analyse results; inspect failure cases qualitatively
- [ ] Write the final report (see structure below)
- [ ] Record a short demo video

**Report structure (maps to your exposé)**
1. Introduction & motivation
2. Background / related work (use the exposé references)
3. System design & architecture (reuse the README diagram)
4. Implementation (ingestion, retrieval, generation, app)
5. Evaluation method (test set, metrics, judge)
6. Results & discussion (the charts from `experiments/results/`)
7. Limitations & failure analysis (be honest — this earns marks)
8. Conclusion & future work

**Done when:** report + deployed app + demo + code on GitHub are all submitted.

---

## Daily rhythm suggestion
Small commits, every working session. A clean git history is itself evidence of
good engineering process for your report.
