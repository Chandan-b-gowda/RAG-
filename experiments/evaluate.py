"""
The RESEARCH engine of the project (Phase 5 / the ~30% research contribution).

This script runs controlled experiments and produces the numbers and charts that
answer your research questions:

  RQ1: Does RAG improve accuracy vs a plain LLM?          -> experiment_rag_vs_plain
  RQ2: How does chunk size/overlap affect quality?        -> experiment_chunking
  RQ3: How does top-k affect quality?                     -> experiment_topk
  RQ5: What are the latency/cost trade-offs?              -> recorded in every run

Accuracy is measured against a curated test set (data/eval/testset.json) of
questions with known correct answers. We use an automatic "LLM-as-judge" plus a
simple keyword check, and we always keep the raw answers so failures can be
inspected qualitatively too.

Run:  python -m experiments.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import EVAL_DIR, RAGConfig  # noqa: E402
from src.rag.generate import get_llm  # noqa: E402
from src.rag.llm_utils import invoke_with_backoff  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- Result cache ----------
# We cache each (mode, config, question) -> full result on disk. Re-running an
# experiment then reuses previous answers instead of paying for the API again.
# Delete experiments/results/.cache.json to force a fresh run.
_CACHE_PATH = RESULTS_DIR / ".cache.json"


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def load_testset() -> list[dict]:
    """Load the curated evaluation questions with their reference answers."""
    path = EVAL_DIR / "testset.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No test set at {path}. Create it first (see data/eval/testset.example.json)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- Scoring ----------
def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (fast, no API cost)."""
    if not expected_keywords:
        return float("nan")
    answer_low = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_low)
    return hits / len(expected_keywords)


JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Is the system answer factually correct and consistent with the reference answer?
Reply with a single word: CORRECT, PARTIAL, or WRONG."""


def judge_score(question: str, reference: str, answer: str, judge_llm) -> str:
    """Use an LLM as an automatic grader. Returns CORRECT / PARTIAL / WRONG."""
    raw = invoke_with_backoff(
        judge_llm,
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer),
    )
    verdict = raw.strip().upper()
    for label in ("CORRECT", "PARTIAL", "WRONG"):
        if label in verdict:
            return label
    return "WRONG"


def _verdict_to_number(v: str) -> float:
    return {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}.get(v, 0.0)


# ---------- Experiments ----------
def run_over_testset(pipeline: RAGPipeline, testset: list[dict], judge_llm, mode: str):
    """
    Run every test question through the pipeline in `mode` ('rag' or 'plain').

    Uses the on-disk cache: a question already answered under the same config+mode
    is not re-sent to the API. This makes repeat runs free and near-instant.
    """
    cache = _load_cache()
    sig = f"{mode}|{pipeline.config.summary()}"  # identifies this exact setup
    rows = []
    for item in testset:
        q = item["question"]
        ref = item.get("answer", "")
        key = f"{sig}|{q}"

        if key in cache:
            row = dict(cache[key])  # reuse previous answer + verdict
        else:
            result = pipeline.ask(q) if mode == "rag" else pipeline.ask_plain(q)
            verdict = judge_score(q, ref, result.answer, judge_llm)
            row = {
                "question": q,
                "mode": mode,
                "answer": result.answer,
                "verdict": verdict,
                "score": _verdict_to_number(verdict),
                "keyword_score": keyword_score(result.answer, item.get("keywords", [])),
                "latency_s": round(result.latency_seconds, 2),
                "num_sources": len(result.sources),
            }
            cache[key] = row
            _save_cache(cache)  # save after each call so progress survives a crash
        row["config"] = pipeline.config.summary()
        rows.append(row)
    return rows


def experiment_rag_vs_plain(config: RAGConfig):
    """RQ1: compare RAG against the plain-LLM baseline on the same questions."""
    print("\n=== RQ1: RAG vs plain LLM ===")
    testset = load_testset()
    pipeline = RAGPipeline.from_folder(config=config)
    judge = get_llm(config)

    rows = run_over_testset(pipeline, testset, judge, "rag")
    rows += run_over_testset(pipeline, testset, judge, "plain")
    df = pd.DataFrame(rows)

    summary = df.groupby("mode").agg(
        accuracy=("score", "mean"),
        avg_latency_s=("latency_s", "mean"),
    )
    print(summary)
    df.to_csv(RESULTS_DIR / "rq1_rag_vs_plain.csv", index=False)
    _save_bar(summary["accuracy"], "RQ1: Accuracy — RAG vs Plain LLM",
              "accuracy", RESULTS_DIR / "rq1_accuracy.png")
    return df


def experiment_chunking(configs: list[RAGConfig]):
    """RQ2: vary chunk size/overlap and measure RAG accuracy."""
    print("\n=== RQ2: effect of chunking ===")
    testset = load_testset()
    judge = get_llm(configs[0])
    rows = []
    for cfg in configs:
        pipeline = RAGPipeline.from_folder(config=cfg)
        label = f"{cfg.chunk_size}/{cfg.chunk_overlap}"
        for r in run_over_testset(pipeline, testset, judge, "rag"):
            r["config"] = label
            rows.append(r)
    df = pd.DataFrame(rows)
    summary = df.groupby("config").agg(accuracy=("score", "mean"),
                                       avg_latency_s=("latency_s", "mean"))
    print(summary)
    df.to_csv(RESULTS_DIR / "rq2_chunking.csv", index=False)
    _save_bar(summary["accuracy"], "RQ2: Accuracy by chunk size/overlap",
              "accuracy", RESULTS_DIR / "rq2_chunking.png")
    return df


def experiment_topk(top_ks: list[int], base: RAGConfig):
    """RQ3: vary how many passages we retrieve and measure RAG accuracy."""
    print("\n=== RQ3: effect of top-k ===")
    testset = load_testset()
    judge = get_llm(base)
    rows = []
    for k in top_ks:
        cfg = RAGConfig(chunk_size=base.chunk_size, chunk_overlap=base.chunk_overlap,
                        top_k=k)
        pipeline = RAGPipeline.from_folder(config=cfg)
        for r in run_over_testset(pipeline, testset, judge, "rag"):
            r["top_k"] = k
            rows.append(r)
    df = pd.DataFrame(rows)
    summary = df.groupby("top_k").agg(accuracy=("score", "mean"),
                                      avg_latency_s=("latency_s", "mean"))
    print(summary)
    df.to_csv(RESULTS_DIR / "rq3_topk.csv", index=False)
    _save_bar(summary["accuracy"], "RQ3: Accuracy by top-k",
              "accuracy", RESULTS_DIR / "rq3_topk.png")
    return df


# ---------- Plotting ----------
def _save_bar(series, title, ylabel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    series.plot(kind="bar", ax=ax, color="#2E86C1")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved chart -> {path}")


if __name__ == "__main__":
    base = RAGConfig()

    # RQ1 — the headline experiment.
    experiment_rag_vs_plain(base)

    # RQ2 — a few chunking configurations.
    experiment_chunking([
        RAGConfig(chunk_size=500, chunk_overlap=50),
        RAGConfig(chunk_size=1000, chunk_overlap=150),
        RAGConfig(chunk_size=1500, chunk_overlap=200),
    ])

    # RQ3 — a few retrieval depths.
    experiment_topk([2, 4, 8], base)

    print("\nAll experiments complete. See experiments/results/ for CSVs and charts.")
