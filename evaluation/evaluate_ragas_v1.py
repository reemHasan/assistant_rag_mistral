"""
evaluate_ragas.py
=================
NBA Reddit RAG — RAGAS evaluation pipeline
with Pydantic validation + Pydantic Logfire tracing.

Stack
-----
  Generator (RAG)  : Mistral, via langchain-mistralai (v1.x SDK)
  Judge (RAGAS)    : Gemini 2.0 Flash, via LangChain (kept as a second
                      provider so judge != generator, methodologically
                      cleaner for the comparative report)
  Vector store     : FAISS (cosine similarity via IndexFlatIP + L2 norm)
  PDF loading      : PyMuPDF (fitz) + EasyOCR  ← image-based PDFs
  Chunking         : LangChain RecursiveCharacterTextSplitter
  Evaluation       : RAGAS (faithfulness, answer_relevancy,
                            context_recall, context_precision)
  Validation       : Pydantic v2
  Observability    : Pydantic Logfire

"""
from __future__ import annotations
import os
import re
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import logfire
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from utils.models import PipelineConfig, TestRow, EvalResult
from rag.retriever import CosineRetriever
from mistralai import Mistral as MistralClient
# The RAG generation path uses MistralClient (raw SDK) directly.
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
 
# ── Judge: Gemini via LangChain (separate provider from the generator) ─────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rag.generator import build_rag_answer
#LangchainLLMWrapper / LangchainEmbeddingsWrapper are deprecated in v0.4
# (still functional, but emit warnings) in favor of llm_factory(), which
# wraps a native async client directly. TestsetGenerator still documents
# LangchainLLMWrapper as of this writing, so it's kept there; the judge
# path below uses the wrapper too since langchain_google_genai doesn't
# expose a llm_factory-compatible raw async client out of the box.
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
# AnswerCorrectness is not yet migrated to ragas.metrics.collections as of
# 0.4.3 — it is computed manually below via answer/ground_truth similarity
# using the same judge LLM, to avoid depending on the legacy metrics module.
# NOTE: datasets.Dataset is no longer needed — the old evaluate(dataset=...)
# batch call is replaced by per-row async .ascore() in the collections API.
 
# ───────────────────────────────────────────────────────────────────────────
# 0. Bootstrap
# ───────────────────────────────────────────────────────────────────────────

load_dotenv()
console = Console()
logging.basicConfig(level=logging.WARNING)

# Logfire — instrument everything
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_pydantic()

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]          # generator (raw SDK)
CORPUS_DIR      = Path(os.getenv("CORPUS_DIR", "./corpus"))     # PDFs go here
OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR", "./eval_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K            = int(os.getenv("TOP_K", "4"))


@logfire.instrument("run_rag_over_testset")
def run_rag_over_testset(
    rows:        list[TestRow],
    retriever:   CosineRetriever,
    mistral_llm: MistralClient,
    config:      PipelineConfig,
) -> list[TestRow]:
    """Fill answer + contexts for every TestRow using normalized cosine search."""
    filled: list[TestRow] = []
    for i, row in enumerate(rows):
        with logfire.span("rag_single_question", idx=i, category=row.category):
            try:
                answer, contexts = build_rag_answer(
                    row.question, retriever, mistral_llm, config.mistral_model, config.top_k
                )
                updated = row.model_copy(update={
                    "answer":   answer,
                    "contexts": contexts,
                })
                # Re-validate with answer + contexts filled
                updated = TestRow(**updated.model_dump())
                filled.append(updated)
                logfire.info(
                    "RAG answer generated",
                    question_snippet = row.question[:60],
                    answer_snippet   = answer[:80],
                    n_contexts       = len(contexts),
                )
            except Exception as exc:
                logfire.error("RAG step failed", question=row.question, error=str(exc))
                filled.append(row)   # keep original (empty answer)
    return filled


# ───────────────────────────────────────────────────────────────────────────
# 7. RAGAS evaluation
# ───────────────────────────────────────────────────────────────────────────
 
@logfire.instrument("run_ragas_evaluation")
async def _score_one_row(
    row:           TestRow,
    faithfulness:  Faithfulness,
    answer_rel:    AnswerRelevancy,
    context_prec:  ContextPrecision,
    context_rec:   ContextRecall,
    judge_llm:     LangchainLLMWrapper,
) -> EvalResult:
    """
    Score a single TestRow against all four ragas.metrics.collections
    metrics, plus a manually-computed answer_correctness proxy.
 
    Each collections metric's .ascore() takes direct keyword arguments
    (user_input, response, retrieved_contexts, reference) and returns a
    MetricResult with .value (float) and .reason (str explanation).
    """
    kwargs = dict(
        user_input         = row.question,
        response           = row.answer,
        retrieved_contexts = row.contexts,
    )
 
    async def _safe_ascore(metric, **extra) -> Optional[float]:
        try:
            result = await metric.ascore(**kwargs, **extra)
            return _safe_float(result.value)
        except Exception as exc:
            logfire.warn(
                "Metric scoring failed",
                metric=metric.__class__.__name__,
                question=row.question[:60],
                reason=str(exc),
            )
            return None
 
    faith_score = await _safe_ascore(faithfulness)
    rel_score   = await _safe_ascore(answer_rel)
    prec_score  = await _safe_ascore(context_prec, reference=row.ground_truth)
    rec_score   = await _safe_ascore(context_rec,  reference=row.ground_truth)
 
    # answer_correctness: AnswerCorrectness isn't in ragas.metrics.collections
    # yet (v0.4.3) — approximate it by asking the judge LLM to score semantic
    # agreement between the generated answer and the ground truth directly.
    correctness_score = await _judge_answer_correctness(
        question     = row.question,
        answer       = row.answer,
        ground_truth = row.ground_truth,
        judge_llm    = judge_llm,
    )
 
    return EvalResult(
        question           = row.question,
        category           = row.category,
        faithfulness       = faith_score,
        answer_relevancy   = rel_score,
        context_recall     = rec_score,
        context_precision  = prec_score,
        answer_correctness = correctness_score,
    )
 
 
async def _judge_answer_correctness(
    question:     str,
    answer:       str,
    ground_truth: str,
    judge_llm:    LangchainLLMWrapper,
) -> Optional[float]:
    """
    Manual answer_correctness proxy via the judge LLM, since
    AnswerCorrectness is not yet available in ragas.metrics.collections
    as of ragas==0.4.3. Asks the judge to rate semantic agreement on a
    0-1 scale, mirroring RAGAS's own correctness definition.
    """
    prompt = (
        "You are evaluating whether a generated answer correctly matches "
        "a reference ground truth answer for a question.\n\n"
        f"Question: {question}\n\n"
        f"Generated answer: {answer}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        "Rate semantic correctness from 0.0 (completely wrong or "
        "contradicts the ground truth) to 1.0 (fully correct and "
        "equivalent in meaning, even if worded differently). "
        "Respond with ONLY a single number between 0.0 and 1.0, "
        "no explanation."
    )
    try:
        response = await judge_llm.agenerate_text(prompt)
        text = response.generations[0][0].text.strip()
        score = float(text)
        return round(score, 4) if 0 <= score <= 1 else None
    except Exception as exc:
        logfire.warn("answer_correctness judge call failed", reason=str(exc))
        return None
 
 
@logfire.instrument("run_ragas_evaluation")
def run_ragas_evaluation(
    rows:          list[TestRow],
    llm_wrapper:   LangchainLLMWrapper,
    embed_wrapper: LangchainEmbeddingsWrapper,
) -> list[EvalResult]:
    """
    Run RAGAS metrics (collections API, ragas==0.4.3) and validate
    results with Pydantic.
 
    Architecture note
    ──────────────────
    ragas 0.4's collections metrics are async-only and scored per-sample,
    not in a single batched evaluate(dataset, metrics=[...]) call like
    older ragas versions. This function instantiates each metric once
    with the judge LLM/embeddings, then loops over rows calling .ascore()
    via asyncio.run() for each — matching the new architecture while
    keeping a synchronous public interface for the rest of the pipeline.
    """
    import asyncio
 
    with logfire.span("ragas_evaluate", n_rows=len(rows)):
        faithfulness_metric  = Faithfulness(llm=llm_wrapper)
        answer_rel_metric    = AnswerRelevancy(llm=llm_wrapper, embeddings=embed_wrapper)
        context_prec_metric  = ContextPrecision(llm=llm_wrapper)
        context_rec_metric   = ContextRecall(llm=llm_wrapper)
 
        async def _score_all() -> list[EvalResult]:
            results = []
            for i, row in enumerate(tqdm(rows, desc="RAGAS scoring")):
                er = await _score_one_row(
                    row,
                    faithfulness_metric,
                    answer_rel_metric,
                    context_prec_metric,
                    context_rec_metric,
                    llm_wrapper,
                )
                results.append(er)
                logfire.info(
                    "Row scored",
                    idx=i,
                    category=row.category,
                    mean_score=er.mean_score,
                )
            return results
 
        eval_results = asyncio.run(_score_all())
 
    logfire.info("RAGAS evaluation complete", rows=len(eval_results))
    return eval_results
 
 
def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return round(f, 4) if 0 <= f <= 1 else None
    except (TypeError, ValueError):
        return None
 
 
# ───────────────────────────────────────────────────────────────────────────
# 8. Reporting
# ───────────────────────────────────────────────────────────────────────────
 
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "answer_correctness",
]
 
CATEGORY_ORDER = ["simple", "comparative", "multihop", "noisy", "out_of_scope"]
 
def _fmt(v: float | None) -> str:
    if v is None: return "—"
    bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
    color = "green" if v >= 0.7 else ("yellow" if v >= 0.4 else "red")
    return f"[{color}]{v:.2f}[/{color}] {bar}"
 
 
def print_summary_table(results: list[EvalResult]) -> None:
    """Rich table: per-category mean scores."""
    table = Table(title="RAGAS scores by category", show_lines=True)
    table.add_column("Category",   style="bold")
    table.add_column("N")
    for m in METRICS:
        table.add_column(m.replace("_", "\n"), justify="right")
    table.add_column("Mean", justify="right", style="bold")
 
    by_cat: dict[str, list[EvalResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
 
    for cat in CATEGORY_ORDER:
        rows = by_cat.get(cat, [])
        if not rows: continue
 
        def mean_metric(m: str) -> float | None:
            vals = [getattr(r, m) for r in rows if getattr(r, m) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
 
        means = [mean_metric(m) for m in METRICS]
        overall = round(sum(v for v in means if v is not None) / max(1, sum(1 for v in means if v is not None)), 4)
        table.add_row(
            cat, str(len(rows)),
            *[_fmt(v) for v in means],
            _fmt(overall),
        )
 
    console.print(table)
 
 
def save_outputs(results: list[EvalResult], rows: list[TestRow]) -> None:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # CSV — detailed
    df_rows = []
    for r, row in zip(results, rows):
        df_rows.append({
            "id":                  f"{row.category[:3].upper()}{rows.index(row)+1:02d}",
            "category":            r.category,
            "question":            r.question,
            "ground_truth":        row.ground_truth,
            "answer":              row.answer,
            "faithfulness":        r.faithfulness,
            "answer_relevancy":    r.answer_relevancy,
            "context_recall":      r.context_recall,
            "context_precision":   r.context_precision,
            "answer_correctness":  r.answer_correctness,
            "mean_score":          r.mean_score,
        })
    df = pd.DataFrame(df_rows)
    csv_path = OUTPUT_DIR / f"ragas_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]CSV saved:[/green] {csv_path}")
 
    # JSON — full detail
    json_path = OUTPUT_DIR / f"ragas_results_{ts}.json"
    json_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False)
    )
    console.print(f"[green]JSON saved:[/green] {json_path}")
 
    # Summary stats log to Logfire
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.mean_score)
    summary = {cat: round(sum(v)/len(v), 4) for cat, v in by_cat.items()}
    logfire.info("Evaluation summary", **summary)
 
 
# ───────────────────────────────────────────────────────────────────────────
# 9. Main orchestration
# ───────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    t0 = time.perf_counter()
    console.rule("[bold]NBA Reddit RAG — RAGAS Evaluation Pipeline[/bold]")
 
    # ── Validate config ──────────────────────────────────────────────────
    with logfire.span("validate_config"):
        config = PipelineConfig()
        console.print(f"[cyan]Config:[/cyan] {config.model_dump()}")
 
    # ── Generator models ──────────────────────────────────────────────────
    # mistral_llm       : raw Mistral SDK v1.x — used in build_rag_answer
    # mistral_embeddings: LangChain wrapper — used in build_faiss_index
    #                     and CosineRetriever (embed_documents / embed_query)
    #
    # Both use mistralai==1.9.10 (v1.x). Key API changes vs v0.x:
    #   MistralClient(api_key=...)  → Mistral(api_key=...)  [aliased above]
    #   client.chat(model, messages) → client.chat.complete(model, messages)
    #   ChatMessage(role=, content=) → {"role": ..., "content": ...}  (plain dict)
    with logfire.span("init_generator_models"):
        mistral_llm = MistralClient(api_key=MISTRAL_API_KEY)
        mistral_embeddings = MistralAIEmbeddings(
            model   = config.embed_model,
            api_key = MISTRAL_API_KEY,
        )
        console.print(
            f"[cyan]Generator:[/cyan] {config.mistral_model} "
            f"| embeddings: {config.embed_model}"
        )
 
    # ── Judge models (Gemini via LangChain) ──────────────────────────────
    # RAGAS metrics (Faithfulness, AnswerRelevancy, etc.) call the judge
    # LLM internally via LangchainLLMWrapper. Gemini is used here to keep
    # the judge fully decoupled from the mistralai SDK version.
    with logfire.span("init_judge_models"):
        judge_llm = ChatGoogleGenerativeAI(
            model          = config.judge_model,
            google_api_key = GOOGLE_API_KEY,
            temperature    = 0,
        )
        judge_embeddings = GoogleGenerativeAIEmbeddings(
            model          = config.judge_embedding_model,
            google_api_key = GOOGLE_API_KEY,
        )
        llm_wrapper   = LangchainLLMWrapper(judge_llm)
        embed_wrapper = LangchainEmbeddingsWrapper(judge_embeddings)
        console.print(f"[cyan]Judge:[/cyan] {config.judge_model} (Gemini, via LangChain)")
 
    # ── Load corpus ───────────────────────────────────────────────────────
    console.print("\n[bold]Step 1/6[/bold] Loading corpus …")
    raw_docs = load_corpus(CORPUS_DIR)
 
    # ── Chunk & validate ──────────────────────────────────────────────────
    console.print("[bold]Step 2/6[/bold] Chunking & validating …")
    valid_chunks, lc_docs = chunk_and_validate(raw_docs, config)
    console.print(f"  → {len(valid_chunks)} valid chunks")
 
    # ── Build FAISS index (or reload from cache) ──────────────────────────
    # Built using Mistral embeddings — must match the embeddings used at
    # query time in CosineRetriever (same model family, same vector space).
    console.print("[bold]Step 3/6[/bold] Building FAISS index …")
    index, valid_chunks = build_faiss_index(
        docs          = lc_docs,
        valid_chunks  = valid_chunks,
        embeddings    = mistral_embeddings,
        corpus_dir    = CORPUS_DIR,
        batch_size    = 32,
        force_rebuild = os.getenv("FORCE_REBUILD", "").lower() in ("1", "true"),
    )
 
    # ── Generate testset (Gemini judge synthesizes Q&A from corpus) ────────
    console.print("[bold]Step 4/6[/bold] Generating testset …")
    test_rows = generate_testset(lc_docs, config, llm_wrapper, embed_wrapper)
    console.print(f"  → {len(test_rows)} test rows")
 
    # ── RAG inference (Mistral generator answers using retrieved context) ──
    console.print("[bold]Step 5/6[/bold] Running RAG over testset …")
    retriever = CosineRetriever(index=index, embeddings=mistral_embeddings)
    test_rows = run_rag_over_testset(test_rows, retriever, mistral_llm, config)
 
    # ── Evaluate (Gemini judge scores Mistral's answers) ────────────────────
    console.print("[bold]Step 6/6[/bold] Running RAGAS evaluation …")
    eval_results = run_ragas_evaluation(test_rows, llm_wrapper, embed_wrapper)
 
    # ── Report ────────────────────────────────────────────────────────────
    console.rule("Results")
    print_summary_table(eval_results)
    save_outputs(eval_results, test_rows)
 
    elapsed = time.perf_counter() - t0
    logfire.info("Pipeline complete", elapsed_s=round(elapsed, 2))
    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")
 
 
if __name__ == "__main__":
    main()