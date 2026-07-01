"""
evaluate_ragas.py
=================
NBA Reddit RAG tool — RAGAS evaluation pipeline
with Pydantic validation + Pydantic Logfire tracing.

Stack
-----
  Generator (RAG tool called by agent_react)  : Mistral, via langchain-mistralai (v1.x SDK)
  Judge (RAGAS)    : Gemini 3.0 Flash, via LangChain (kept as a second
                      provider so judge != generator, methodologically
                      cleaner for the comparative report)
  Evaluation       : RAGAS (faithfulness, answer_relevancy,
                            context_recall, context_precision)
  Validation       : Pydantic v2
  Observability    : Pydantic Logfire

"""
from __future__ import annotations
import argparse
import json
import time
import logging
import re
from pathlib import Path
from typing import Any
import logfire
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from datasets import Dataset
from utils.models import PipelineConfig, TestRow, EvalResult
from agent_react import build_agent
from langchain.agents import AgentExecutor
from utils.config import DATABASE_URL, MISTRAL_API_KEY, GEMINI_API_KEY
# ── Judge: Gemini via LangChain (separate provider from the generator) ─────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision,
)
# define metric, and strict generate 1 question for answer relevancy to not supercharge llm
answer_relevancy = AnswerRelevancy(strictness=1)
metrics=[
        Faithfulness(),
        ContextRecall(),
        ContextPrecision(),
        answer_relevancy
    ]
run_config = RunConfig(
                max_workers=1,
                timeout=300,
            )
# ───────────────────────────────────────────────────────────────────────────
# 0. Bootstrap
# ───────────────────────────────────────────────────────────────────────────

load_dotenv()
console = Console()
logging.basicConfig(level=logging.WARNING)
# ───────────────────────────────────────────────────────────────────────────
# 1. Rag pipline
# ───────────────────────────────────────────────────────────────────────────
_CHUNK_HEADER_RE = re.compile(
    r"^\[Chunk\s+\d+\s*\|[^\]]*\]\s*\n?",
    re.MULTILINE,
)
def _strip_chunk_headers(raw_obs: str) -> list[str]:
    """
    Split the RAG tool observation into individual chunks and remove
    the metadata header injected by NBAKnowledgeTool._run().
 
    NBAKnowledgeTool joins chunks with '\n\n---\n\n' and prefixes each
    chunk with:
        [Chunk 1 | source: Reddit_2 | topic: reddit_discussion | score: 0.7985]
 
    This header is useful for logging and Logfire tracing but harmful
    for RAGAS evaluation — the judge LLM treats it as content, which
    lowers faithfulness and context_recall scores spuriously.
 
    Returns a list of clean content strings with headers stripped.
    """
    raw_chunks = [c.strip() for c in raw_obs.split("---") if c.strip()]
    clean = []
    for chunk in raw_chunks:
        cleaned = _CHUNK_HEADER_RE.sub("", chunk).strip()
        if cleaned:
            clean.append(cleaned)
    return clean if clean else [raw_obs]


def load_eval_dataset(
        dataset_file: str= "eval_dataset.json",
) -> list[TestRow]:
    """ Load & convert the evaluation dataset as json file into list[TestRow]"""
    # Load rag evaluation dataset
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rows: list[TestRow] = []
    # Loop over questions and insert them into TestRow
    for sample in dataset:
        try:
            rows.append(TestRow(**sample))
        except Exception as exc:
            logfire.warn("TestRow validation failed (rag evaluation dataset)", reason=str(exc))
    logfire.info(f"Loaded {len(rows)} TestRow from evaluation dataset")
    return rows

def build_rag_answer(
    agent: AgentExecutor,
    question: str,
) -> tuple[str, list[str]]:
    """ Retrive answer + context for one TestRow throught rag tool """
    try:
        result = agent.invoke({"input": question})
    except Exception:
        logfire.exception("Agent invocation failed")
        return "", []
    answer = result.get("output", "")
    steps = result.get("intermediate_steps", [])
    contexts = []
    for action, observation in steps:
        if action.tool == "nba_knowledge_rag":
            raw = str(observation)
            contexts = _strip_chunk_headers(raw)
    if not contexts:
        logfire.info("No RAG context available for this question.")
    return answer, contexts


#@logfire.instrument("run_rag_over_testset")
def run_rag_over_testset(
    rows:  list[TestRow],
    agent: AgentExecutor,
    dataset_output_file: str ="eval_agent_rag_dataset_filled.json"
) -> list[TestRow]:
    
    """Fill answer + contexts for every TestRow using ReAct-agent with Rag tool."""
    filled: list[TestRow] = []
    q_without_context = 0
    for i, row in enumerate(rows):      
        with logfire.span("rag_single_question", idx=i, category=row.category):
            try:
                answer, contexts = build_rag_answer(agent=agent, question=row.question)
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
                # count the number of questions without contexts
                if contexts == []:
                    q_without_context +=1
                time.sleep(30)
            except Exception as exc:
                logfire.error("RAG step failed", question=row.question[:60], error=str(exc))
                filled.append(row)   # keep original (empty answer)
    print("{q_without_context} question without context")
    # save filled dataset
    with open(dataset_output_file, "w", encoding="utf-8") as f:
        json.dump([row.model_dump() for row in filled], f, indent=2, ensure_ascii=False)
    logfire.info("Filled evaluation dataset had been dumped into json file with success")
    return filled

# ───────────────────────────────────────────────────────────────────────────
#  2. RAGAS evaluation
# ───────────────────────────────────────────────────────────────────────────
#@logfire.instrument("run_ragas_evaluation")
def run_ragas_evaluation(
    rows:          list[TestRow],
    llm_wrapper:   LangchainLLMWrapper,
    embed_wrapper: LangchainEmbeddingsWrapper,
) -> list[EvalResult]:
    """
     Run RAGAS metrics (ragas==0.4.3)AnswerRelevancy, Faithfulness, ContextRecall, ContextPrecision on the whole evaluation dataset
     Validate results with Pydantic.
    """

    eval_results = []
    hf_dataset = Dataset.from_list([
    {
        "question": r.question,
        "answer": r.answer,
        "contexts": r.contexts,
        "ground_truth": r.ground_truth,
    }
    for r in rows
    ])
    try:
        result = evaluate(
            dataset    = hf_dataset,
            metrics    = metrics,
            llm        = llm_wrapper,
            embeddings = embed_wrapper,
            run_config = run_config
        )

        df = result.to_pandas()

        eval_results.extend(
            EvalResult(
                question=row.question,
                category=row.category,
                faithfulness=df_row["faithfulness"],
                answer_relevancy=df_row["answer_relevancy"],
                context_precision=df_row["context_precision"],
                context_recall=df_row["context_recall"],
            )
            for row, (_, df_row) in zip(rows, df.iterrows())
        )
        logfire.info("RAGAS evaluation complete")
        return eval_results
    
    except Exception as exc:
        logfire.exception(f"RAGAS evaluation error : {exc}")
        raise

# ───────────────────────────────────────────────────────────────────────────
# 3. Reporting
# ───────────────────────────────────────────────────────────────────────────
 
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    #"answer_correctness",
]
 
CATEGORY_ORDER = ["simple", "comparative", "multihop", "noisy"]
 
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
        if not rows:
            continue
 
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
 
 
def save_outputs(results: list[EvalResult], rows: list[TestRow], OUTPUT_DIR) -> None:
 
    # CSV — detailed
    df_rows = []
    for r, row in zip(results, rows):
        df_rows.append({
            "id":                  f"{row.category[:3].upper()}{rows.index(row)+1:02d}",
            "category":            r.category,
            "user_input":            r.question,
            "retrieved_contexts":    row.contexts,
            "reference":        row.ground_truth,
            "response":              row.answer,
            "faithfulness":        r.faithfulness,
            "answer_relevancy":    r.answer_relevancy,
            "context_recall":      r.context_recall,
            "context_precision":   r.context_precision,
            #"answer_correctness":  r.answer_correctness,
            "mean_score":          r.mean_score,
        })
    df = pd.DataFrame(df_rows)
    csv_path = OUTPUT_DIR / "ragas_agent_rag_gemini_results.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]CSV saved:[/green] {csv_path}")
 
    # JSON — full detail
    json_path = OUTPUT_DIR / "ragas_agent_rag_gemini_results.json"
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
#  4. Main orchestration
# ───────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    # __ Get evalaution dataset path _______________________
    BASE_DIR = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Create reponse&context for testset using Agent Rag")
    parser.add_argument(
        "--testset_name",
        type=str,
        default="eval_dataset.json",
        help="The name of testset to be used in evaluation of rag tool by ReAct Agent"
    )
    parser.add_argument(
        "--testset_filled",
        type=str,
        default="eval_agent_rag_dataset_filled.json",
        help="The name of testset after filled with reponse&context from rag tool by ReAct Agent"
    )
    args = parser.parse_args()
    input_eval_dataset = BASE_DIR/f"evaluation/eval_artifacts/{args.testset_name}"
    output_eval_dataset = BASE_DIR/f"evaluation/eval_artifacts/{args.testset_filled}"
    
    t0 = time.perf_counter()
    console.rule("[bold]NBA Reddit Agent (RAG)  — RAGAS Evaluation Pipeline[/bold]")
 
    # ── Validate config ──────────────────────────────────────────────────
    config = PipelineConfig()
    console.print(f"[cyan]Config:[/cyan] {config.model_dump()}")

    # ── Judge models (Gemini via LangChain) ──────────────────────────────
    # RAGAS metrics (Faithfulness, AnswerRelevancy, etc.) call the judge
    # LLM internally via LangchainLLMWrapper. Gemini is used here to keep
    # the judge fully decoupled from the mistralai SDK version.
    judge_llm = ChatGoogleGenerativeAI(
        model          = config.judge_model,
        google_api_key = GEMINI_API_KEY,
        temperature    = 0,
    )
    judge_embeddings = GoogleGenerativeAIEmbeddings(
        model          = config.judge_embedding_model,
        google_api_key = GEMINI_API_KEY,
    )
    llm_wrapper   = LangchainLLMWrapper(judge_llm)
    embed_wrapper = LangchainEmbeddingsWrapper(judge_embeddings)
    console.print(f"[cyan]Judge:[/cyan] {config.judge_model} (Gemini, via LangChain)")
 
    # ── Initialize agent ──────────────────────────────────────────────────
    agent  = build_agent(db_uri=DATABASE_URL, mistral_api_key=MISTRAL_API_KEY,agent_model=config.mistral_model, 
                         sql_model=config.sql_generator_model, embed_model=config.embed_model, top_k_rag=config.top_k, verbose=True)
    
    # ── Load Evaluation dataset ───────────────────────────────────────────────────────
    console.print("\n[bold]Step 1/3[/bold] Loading Evaluation dataset …")
    eval_rows = load_eval_dataset(input_eval_dataset)
    #print(eval_rows[0])
    
    # ── RAG tool inference (Mistral generator answers using retrieved context inside Agent) ──
    console.print("[bold]Step 2/3[/bold] Running RAG over testset …")
    #test_rows = run_rag_over_testset(rows=eval_rows[:3], agent=agent, dataset_output_file=output_eval_dataset)
    test_rows = run_rag_over_testset(rows=eval_rows, agent=agent, dataset_output_file=output_eval_dataset)

    # ── Evaluate (Gemini judge scores Mistral's answers) ────────────────────
    console.print("[bold]Step 3/3[/bold] Running RAGAS evaluation …")
    eval_results = run_ragas_evaluation(test_rows, llm_wrapper, embed_wrapper)
 
    # ── Report ────────────────────────────────────────────────────────────
    console.rule("Results")
    print_summary_table(eval_results)
    save_outputs(eval_results, test_rows, BASE_DIR/"evaluation/eval_artifacts")

    elapsed = time.perf_counter() - t0
    logfire.info("Pipeline complete", elapsed_s=round(elapsed, 2))
    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")
 
 
if __name__ == "__main__":
    main()