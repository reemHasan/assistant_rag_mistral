"""
evaluate_agent.py
=================
Route-aware evaluation of the two-tool NBA agent (RAG + SQL).

Problem
-------
RAGAS context_recall and context_precision assume retrieved text chunks
exist. The SQL tool returns structured table results, not text chunks —
so these two metrics are undefined for SQL-routed questions.

Solution: split evaluation by the tool the agent actually used, and
apply the right metric set to each subset:

  RAG-routed questions
    • faithfulness          (answer grounded in retrieved chunks?)
    • answer_relevancy      (answer on-topic?)
    • context_recall        (ground truth covered by chunks?)
    • context_precision     (chunks actually useful?)

  SQL-routed questions
    • faithfulness          (answer grounded in SQL result table?)
    • answer_relevancy      (answer on-topic?)
    • sql_execution_success (did the query run without error?)
    • result_non_empty      (did the query return rows?)
    — context_recall    → N/A  (no text chunks, marked None)
    — context_precision → N/A  (no text chunks, marked None)

The final comparative table includes all six columns.
SQL-only rows have None for the two context metrics, which are
excluded from per-category averages — not treated as 0.

Usage
-----
    python evaluate_agent.py \
        --testset  eval_dataset.json \
        --output   eval_agent_results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision,
    #AnswerCorrectness
)
from ragas.run_config import RunConfig
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from utils.models import PipelineConfig
from agent_react import build_agent
from utils.config import DATABASE_URL, MISTRAL_API_KEY, GEMINI_API_KEY


log     = logging.getLogger(__name__)
console = Console()

# define metric, and strict generate 1 question for answer relevancy to not supercharge llm
answer_relevancy = AnswerRelevancy(strictness=1)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentEvalRow:
    """One row of the agent evaluation — filled progressively."""
    question:     str
    ground_truth: str
    category:     str

    # Expected tool from the testset 
    # Possible values: "nba_stats_sql" | "nba_knowledge_rag" | "both" | None
    expected_tool: Optional[str] = None

    # Filled after agent invocation
    answer:         str       = ""
    tools_used:     list[str] = field(default_factory=list)
    # Every distinct tool name called during this turn, in call order.
    # e.g. ["nba_stats_sql", "nba_knowledge_rag"] for a hybrid question
    # that correctly used both. Empty list means no tool was called.

    contexts:       list[str] = field(default_factory=list)
    sql_query:      str       = ""
    raw_sql_result: str       = ""

    # Routing accuracy — filled after invocation when expected_tool is known
    routing_correct:    Optional[bool] = None  # True/False/None(no expectation)
    routing_error_type: str            = ""
    # Possible routing_error_type values:
    #   ""                    — correct or no expectation
    #   "used_sql_not_rag"    — used SQL only when RAG was expected
    #   "used_rag_not_sql"    — used RAG only when SQL was expected
    #   "used_none"           — agent answered directly, a tool was expected
    #   "missed_sql"          — expected "both", only RAG was called
    #   "missed_rag"          — expected "both", only SQL was called
    #   "missed_both"         — expected "both", no tool was called
    #   "used_wrong_tool"     — catchall for other mismatches

    # SQL-specific binary metrics (None for RAG-only rows)
    sql_execution_success: Optional[bool] = None
    result_non_empty:      Optional[bool] = None

    # RAGAS scores
    faithfulness:       Optional[float] = None
    answer_relevancy:   Optional[float] = None
    context_recall:     Optional[float] = None   # None for SQL-only rows
    context_precision:  Optional[float] = None   # None for SQL-only rows
    #answer_correctness: Optional[float] = None

    @property
    def tool_used(self) -> str:
        """
        Backward-compatible single-tool view: the LAST tool called,
        or "none" if no tool was used. Used by reporting code that
        groups rows by a single tool (RAG table vs SQL table sections).
        """
        return self.tools_used[-1] if self.tools_used else "none"

    @property
    def is_sql_routed(self) -> bool:
        return "nba_stats_sql" in self.tools_used

    @property
    def is_rag_routed(self) -> bool:
        return "nba_knowledge_rag" in self.tools_used

    @property
    def is_hybrid_routed(self) -> bool:
        return self.is_sql_routed and self.is_rag_routed

    @property
    def mean_score(self) -> float:
        scores = [
            s for s in [
                self.faithfulness, self.answer_relevancy,
                self.context_recall, self.context_precision,
                #self.answer_correctness,
            ]
            if s is not None
        ]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def compute_routing_accuracy(self) -> None:
        """
        Compare expected_tool vs tools_used and populate
        routing_correct + routing_error_type.

        Called after invoke_agent() fills tools_used.
        Rows without expected_tool (RAG-only testset rows)
        are left as None — not treated as errors.

        Handles three expected_tool cases:
          "nba_stats_sql"     → correct iff SQL was called (RAG may or may
                                 not also be called — extra RAG use is not
                                 penalised, since it doesn't hurt the answer)
          "nba_knowledge_rag" → correct iff RAG was called (same logic)
          "both"               → correct iff BOTH tools were called
        """
        if self.expected_tool is None:
            self.routing_correct    = None
            self.routing_error_type = ""
            return
        # verify which tool is called in tool_used list
        used_sql = self.is_sql_routed
        used_rag = self.is_rag_routed
        used_none = len(self.tools_used) == 0

        # hybired question rag+sql
        if self.expected_tool == "both":
            if used_sql and used_rag:
                self.routing_correct    = True
                self.routing_error_type = ""
            elif used_none:
                self.routing_correct    = False
                self.routing_error_type = "missed_both"
            elif used_sql and not used_rag:
                self.routing_correct    = False
                self.routing_error_type = "missed_rag"
            elif used_rag and not used_sql:
                self.routing_correct    = False
                self.routing_error_type = "missed_sql"
            else:
                self.routing_correct    = False
                self.routing_error_type = "used_wrong_tool"
            return
        
        # questions designed to be called by sql tool
        if self.expected_tool == "nba_stats_sql":
            if used_sql:
                self.routing_correct    = True
                self.routing_error_type = ""
            elif used_none: # agent did not call any tool
                self.routing_correct    = False
                self.routing_error_type = "used_none"
            else:
                self.routing_correct    = False
                self.routing_error_type = "used_rag_not_sql"
            return
        
        # questions designed to be called by rag tool
        if self.expected_tool == "nba_knowledge_rag":
            if used_rag:
                self.routing_correct    = True
                self.routing_error_type = ""
            elif used_none: # agent did not call any tool
                self.routing_correct    = False
                self.routing_error_type = "used_none"
            else:
                self.routing_correct    = False
                self.routing_error_type = "used_sql_not_rag"
            return

        # Unknown expected_tool value — defensive fallback
        self.routing_correct    = False
        self.routing_error_type = "used_wrong_tool"
        log.warning(
            "Unrecognised expected_tool value: %r — treated as mismatch",
            self.expected_tool,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Agent invocation & tool-path extraction
# ─────────────────────────────────────────────────────────────────────────────

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


def _extract_sql_from_observation(observation: str) -> tuple[str, str]:
    """
    Parse the tool observation returned by NBAStatsSQLTool._run().
    Format: "SQL:\n<query>\n\nRESULTS:\n<table>"
    Returns (sql_query, raw_result).
    """
    sql, result = "", ""
    if "SQL:" in observation and "RESULTS:" in observation:
        parts  = observation.split("RESULTS:", 1)
        result = parts[1].strip() if len(parts) > 1 else ""
        sql    = parts[0].replace("SQL:", "").strip()
    return sql, result


def invoke_agent(agent, question: str, expected_tool: Optional[str] = None) -> AgentEvalRow:
    """
    Run the agent on one question and extract everything needed for evaluation.
    After agent invoking:
    - Iterates ALL intermediate_steps (not just the last one) so that hybrid
    questions — where the agent correctly calls both nba_stats_sql and
    nba_knowledge_rag in the same turn — are captured fully in tools_used.
    - Then computes routing accuracy once tools_used is fully populated.
    """
    row = AgentEvalRow(
        question      = question,
        ground_truth  = "",
        category      = "",
        expected_tool = expected_tool,
    )

    try:
        result = agent.invoke({"input": question})
    except Exception as exc:
        log.error("Agent invocation failed: %s", exc)
        row.answer = f"[AGENT ERROR: {exc}]"
        row.compute_routing_accuracy()
        return row
    # get answer & steps
    row.answer = result.get("output", "")
    steps = result.get("intermediate_steps", [])

    if not steps:
        # Agent answered directly without any tool call
        row.compute_routing_accuracy()
        return row

    # ── Iterate every step — accumulate tools_used and merge observations ──
    rag_contexts:  list[str] = []
    sql_query:     str       = ""
    sql_result:    str       = ""
    sql_success:   Optional[bool] = None
    sql_nonempty:  Optional[bool] = None
    
    for action, observation in steps:
        # 1. Extract tool_used list
        tool_name = action.tool
        if tool_name not in row.tools_used:
            row.tools_used.append(tool_name)

        raw_obs = str(observation)
        # 2. Extract rag context
        if tool_name == "nba_knowledge_rag":
            rag_contexts.extend(_strip_chunk_headers(raw_obs))
        # 3. Extract sql query & sql result
        elif tool_name == "nba_stats_sql":
            q, r = _extract_sql_from_observation(raw_obs)
            # Keep the LAST SQL call's query/result for reporting —
            # if the agent retried after an error, the final attempt
            # is the one that matters for sql_query/raw_sql_result.
            sql_query    = q or sql_query
            sql_result   = r or sql_result
            sql_success  = "SQL execution error" not in raw_obs
            sql_nonempty = (
                sql_success
                and "No results found" not in raw_obs
                and r.strip() != ""
            )

    # ── Populate row from accumulated data ─────────────────────────────────
    row.sql_query              = sql_query
    row.raw_sql_result         = sql_result
    row.sql_execution_success  = sql_success
    row.result_non_empty       = sql_nonempty

    # contexts used for RAGAS faithfulness/context_recall/context_precision:
    #   RAG-only rows  → the retrieved chunks
    #   SQL-only rows  → the raw SQL result table (single "chunk")
    #   Hybrid rows    → both, concatenated — faithfulness checks the
    #                    answer against the union of everything retrieved
    if rag_contexts and sql_result: # both tools called
        row.contexts = rag_contexts + [sql_result]
    elif rag_contexts: # just rag tool called 
        row.contexts = rag_contexts
    elif sql_result: # just sql tool called
        row.contexts = [sql_result]
    else: # agent did not call any tool
        row.contexts = []
    # filling routing_correct, routing_error_type by filled tool_used list
    row.compute_routing_accuracy()
    return row


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RAGAS scoring — synchronous batch evaluate() with ragas.metrics
# ─────────────────────────────────────────────────────────────────────────────
def _safe_float(v) -> Optional[float]:
    """Return float in [0,1] or None — rejects out-of-range judge outputs."""
    try:
        f = float(v)
        return round(f, 4) if 0.0 <= f <= 1.0 else None
    except (TypeError, ValueError):
        return None

def _make_judge_models(
    judge_model:       str,
    judge_embed_model: str,
    google_api_key:    str,
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """
    Instantiate fresh Gemini judge LLM + embeddings.

    Called ONCE per evaluate() batch rather than once per script run,
    because the underlying grpc.aio channel enters a half-closed state
    after the first evaluate() finishes. Reusing the same LangChain
    wrapper objects for a second evaluate() call triggers:
        AttributeError: 'InterceptedUnaryUnaryCall' has no attribute
                        '_interceptors_task'
    Constructing new wrapper objects forces a fresh gRPC channel for
    each evaluate() batch, sidestepping the channel reuse problem.
    """
    llm_raw = ChatGoogleGenerativeAI(
        model          = judge_model,
        google_api_key = google_api_key,
        temperature    = 0,
    )
    embed_raw = GoogleGenerativeAIEmbeddings(
        model          = judge_embed_model,
        google_api_key = google_api_key,
    )
    return LangchainLLMWrapper(llm_raw), LangchainEmbeddingsWrapper(embed_raw)


def _run_evaluate_batch(
    dataset:        Dataset,
    metrics:        list,
    judge_model:    str,
    judge_embed_model: str,
    google_api_key: str,
    label:          str = "",
) -> Optional[pd.DataFrame]:
    """
    Run one ragas evaluate() call with a freshly constructed judge.

    A new judge is built for every call to avoid the grpc.aio channel
    corruption that occurs when the same wrapper is reused across two
    sequential evaluate() calls.
    """
    if len(dataset) == 0:
        return None

    import asyncio

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    # Fresh judge objects → fresh gRPC channel → no interceptor-task error
    judge_llm, judge_embed = _make_judge_models(
        judge_model, judge_embed_model, google_api_key
    )
    run_cfg = RunConfig(max_workers=1, timeout=300)

    try:
        console.print(f"  [dim]evaluate() → {label} ({len(dataset)} rows)[/dim]")
        result = evaluate(
            dataset    = dataset,
            metrics    = metrics,
            llm        = judge_llm,
            embeddings = judge_embed,
            run_config = run_cfg,
        )
        return result.to_pandas()
    except Exception as exc:
        log.error("ragas evaluate() failed [%s]: %s", label, exc)
        return None


def score_all_rows(
    rows:             list[AgentEvalRow],
    judge_model:      str,
    judge_embed_model: str,
    google_api_key:   str,
) -> list[AgentEvalRow]:
    """
    Score all AgentEvalRows using the ragas.metrics batch evaluate().

    Row categorisation
    ──────────────────
    Three non-overlapping groups are built from tools_used:

      rag_only  — only nba_knowledge_rag was called
                  Metrics: faithfulness, answer_relevancy,
                           context_recall, context_precision,

      both      — BOTH tools were called (hybrid question)
                  contexts = RAG chunks + SQL result table (concatenated
                  by invoke_agent, so faithfulness checks the answer
                  against both sources)
                  Metrics: same as rag_only (context metrics apply
                  because real text chunks are present alongside the
                  SQL result)

      sql_only  — only nba_stats_sql was called
                  contexts = [raw_sql_result] — faithfulness checks
                  whether the answer is grounded in the SQL table
                  context_recall = None   (no text chunks)
                  context_precision = None

    Each group calls _run_evaluate_batch() with a FRESH judge instance
    to avoid the grpc.aio channel corruption that occurs when the same
    LangChain wrapper is reused across two sequential evaluate() calls.

    RunConfig(max_workers=1, timeout=300)
    ──────────────────────────────────────
    max_workers=1 — sequential, avoids concurrent Gemini 429s
    timeout=300   — 5 min per metric call for slow free-tier responses
    """

    # ── Categorise rows ───────────────────────────────────────────────────
    rag_only_idx = [
        i for i, r in enumerate(rows)
        if r.is_rag_routed and not r.is_sql_routed and r.contexts
    ]
    both_idx = [
        i for i, r in enumerate(rows)
        if r.is_hybrid_routed and r.contexts
    ]
    sql_only_idx = [
        i for i, r in enumerate(rows)
        if r.is_sql_routed and not r.is_rag_routed and r.contexts
    ]
    skipped_idx = [
        i for i, r in enumerate(rows)
        if not r.contexts
    ]

    console.print(
        f"  Scoring split: "
        f"[cyan]RAG-only {len(rag_only_idx)}[/cyan]  "
        f"[magenta]hybrid {len(both_idx)}[/magenta]  "
        f"[yellow]SQL-only {len(sql_only_idx)}[/yellow]  "
        f"[dim]skipped (no context) {len(skipped_idx)}[/dim]"
    )

    # ── Helper: build HF Dataset from row indices ─────────────────────────
    def _make_dataset(indices: list[int]) -> Dataset:
        return Dataset.from_dict({
            "question":     [rows[i].question     for i in indices],
            "answer":       [rows[i].answer        for i in indices],
            "contexts":     [rows[i].contexts      for i in indices],
            "ground_truth": [rows[i].ground_truth  for i in indices],
        })

    # ── Helper: write scores back to rows ─────────────────────────────────
    def _write_scores(
        df:      pd.DataFrame,
        indices: list[int],
        has_context_metrics: bool,
    ) -> None:
        for pos, row_idx in enumerate(indices):
            r = rows[row_idx]
            r.faithfulness      = _safe_float(df.at[pos, "faithfulness"])
            r.answer_relevancy  = _safe_float(df.at[pos, "answer_relevancy"])
            #r.answer_correctness= _safe_float(df.at[pos, "answer_correctness"])
            if has_context_metrics:
                r.context_recall    = _safe_float(df.at[pos, "context_recall"])
                r.context_precision = _safe_float(df.at[pos, "context_precision"])
            else:
                r.context_recall    = None   # N/A — not 0
                r.context_precision = None
            log.info("Scored row %d — mean=%.3f", row_idx, r.mean_score)

    ALL_METRICS = [
        Faithfulness(), answer_relevancy,ContextRecall(),
        ContextPrecision(),
    ]
    SQL_METRICS = [
        Faithfulness(), answer_relevancy,
    ]

    # ── Batch 1: RAG-only rows (full metric set) ──────────────────────────
    if rag_only_idx:
        console.print("\n[bold]Batch 1/3:[/bold] RAG-only rows")
        df = _run_evaluate_batch(
            dataset        = _make_dataset(rag_only_idx),
            metrics        = ALL_METRICS,
            judge_model    = judge_model,
            judge_embed_model = judge_embed_model,
            google_api_key = google_api_key,
            label          = "RAG-only",
        )
        if df is not None:
            _write_scores(df, rag_only_idx, has_context_metrics=True)

    # ── Batch 2: Hybrid rows (full metric set, fresh gRPC channel) ────────
    if both_idx:
        console.print("\n[bold]Batch 2/3:[/bold] Hybrid rows (RAG + SQL contexts)")
        df = _run_evaluate_batch(
            dataset        = _make_dataset(both_idx),
            metrics        = ALL_METRICS,
            judge_model    = judge_model,
            judge_embed_model = judge_embed_model,
            google_api_key = google_api_key,
            label          = "hybrid",
        )
        if df is not None:
            _write_scores(df, both_idx, has_context_metrics=True)

    # ── Batch 3: SQL-only rows (no context metrics, fresh gRPC channel) ───
    if sql_only_idx:
        console.print("\n[bold]Batch 3/3:[/bold] SQL-only rows")
        df = _run_evaluate_batch(
            dataset        = _make_dataset(sql_only_idx),
            metrics        = SQL_METRICS,
            judge_model    = judge_model,
            judge_embed_model = judge_embed_model,
            google_api_key = google_api_key,
            label          = "SQL-only",
        )
        if df is not None:
            _write_scores(df, sql_only_idx, has_context_metrics=False)

    return rows

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Reporting
# ─────────────────────────────────────────────────────────────────────────────

METRICS = [
    ("faithfulness",       True,  True),    # (name, rag, sql)
    ("answer_relevancy",   True,  True),
    ("context_recall",     True,  False),   # RAG only
    ("context_precision",  True,  False),   # RAG only
    #("answer_correctness", True,  True),
]

SQL_METRICS = [
    ("sql_execution_success", "SQL ran without error"),
    ("result_non_empty",      "Query returned rows"),
]


def print_routing_accuracy(rows: list[AgentEvalRow]) -> None:
    """
    Print a dedicated routing accuracy report.

    Only rows with expected_tool defined are counted.
    Testset rows (expected_tool=None) are excluded.

    Metrics shown:
      Overall accuracy      — correct / total rows with expectation
      Per-tool accuracy     — accuracy split by expected_tool value
      Error breakdown       — how many misroutes of each type
      Per-category accuracy — which question categories trip the router
    """
    # Only evaluate rows that have an expected_tool defined
    rows_with_expectation = [r for r in rows if r.expected_tool is not None]
    rows_no_expectation   = [r for r in rows if r.expected_tool is None]

    if not rows_with_expectation:
        console.print(
            "\n[dim]No rows with expected_tool defined — "
            "routing accuracy not computed.[/dim]"
        )
        return

    correct   = [r for r in rows_with_expectation if r.routing_correct is True]
    incorrect = [r for r in rows_with_expectation if r.routing_correct is False]
    accuracy  = len(correct) / len(rows_with_expectation)

    color = "green" if accuracy >= 0.85 else ("yellow" if accuracy >= 0.65 else "red")

    console.print("\n[bold]Routing Accuracy[/bold]")
    console.print(
        f"  Evaluated: {len(rows_with_expectation)} rows with expectation  "
        f"| Skipped: {len(rows_no_expectation)} rows without expectation"
    )
    console.print(
        f"  Overall accuracy: [{color}]{accuracy:.1%}[/{color}]  "
        f"({len(correct)}/{len(rows_with_expectation)} correct)"
    )

    # ── Per-tool accuracy ──────────────────────────────────────────────────
    tool_table = Table(title="Routing accuracy by expected tool", show_lines=True)
    tool_table.add_column("Expected tool")
    tool_table.add_column("Total", justify="right")
    tool_table.add_column("Correct", justify="right")
    tool_table.add_column("Accuracy", justify="right")
    tool_table.add_column("Most common error")

    for expected in ["nba_stats_sql", "nba_knowledge_rag", "both"]:
        # get just question with the three defined expected tools
        subset  = [r for r in rows_with_expectation if r.expected_tool == expected]
        if not subset:
            continue
        ok      = [r for r in subset if r.routing_correct]
        acc     = len(ok) / len(subset)
        # Most common error type among wrong rows
        errors  = [r.routing_error_type for r in subset if not r.routing_correct]
        top_err = max(set(errors), key=errors.count) if errors else "—"
        c       = "green" if acc >= 0.85 else ("yellow" if acc >= 0.65 else "red")
        tool_table.add_row(
            expected,
            str(len(subset)),
            str(len(ok)),
            f"[{c}]{acc:.1%}[/{c}]",
            top_err,
        )

    console.print(tool_table)

    # ── Error breakdown ────────────────────────────────────────────────────
    if incorrect:
        console.print("\n[bold]Routing errors breakdown:[/bold]")
        error_counts: dict[str, int] = {}
        for r in incorrect:
            error_counts[r.routing_error_type] = \
                error_counts.get(r.routing_error_type, 0) + 1

        err_descriptions = {
            "used_rag_not_sql": "Asked for stats → used RAG (missed SQL)",
            "used_sql_not_rag": "Asked for narrative → used SQL (missed RAG)",
            "used_none":        "No tool used when one was expected",
            "missed_sql":       "Hybrid question → only RAG called (missed SQL)",
            "missed_rag":       "Hybrid question → only SQL called (missed RAG)",
            "missed_both":      "Hybrid question → no tool called at all",
            "used_wrong_tool":  "Other tool mismatch",
        }
        for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            desc = err_descriptions.get(err_type, err_type)
            pct  = count / len(rows_with_expectation) * 100
            console.print(f"  [red]{desc}[/red]: {count} ({pct:.0f}%)")

        # Print misrouted questions for qualitative analysis
        console.print("\n[bold]Misrouted questions (for critical analysis):[/bold]")
        err_table = Table(show_lines=True, show_header=True)
        err_table.add_column("Category",      style="dim", max_width=25)
        err_table.add_column("Expected",      max_width=12)
        err_table.add_column("Tools used",    max_width=22)
        err_table.add_column("Error type",    max_width=18)
        err_table.add_column("Question",      max_width=50)

        for r in incorrect:
            tools_str = ", ".join(r.tools_used) if r.tools_used else "none"
            err_table.add_row(
                r.category,
                r.expected_tool or "—",
                tools_str,
                r.routing_error_type,
                r.question[:50] + ("…" if len(r.question) > 50 else ""),
            )
        console.print(err_table)

    # ── Per-category accuracy ──────────────────────────────────────────────
    cat_table = Table(title="Routing accuracy by category", show_lines=True)
    cat_table.add_column("Category")
    cat_table.add_column("N",        justify="right")
    cat_table.add_column("Expected", justify="right")
    cat_table.add_column("Accuracy", justify="right")

    by_cat: dict[str, list[AgentEvalRow]] = {}
    for r in rows_with_expectation:
        by_cat.setdefault(r.category, []).append(r)

    for cat in sorted(by_cat):
        subset   = by_cat[cat]
        ok       = [r for r in subset if r.routing_correct]
        acc      = len(ok) / len(subset)
        expected = subset[0].expected_tool or "—"
        c        = "green" if acc >= 0.85 else ("yellow" if acc >= 0.65 else "red")
        cat_table.add_row(cat, str(len(subset)), expected, f"[{c}]{acc:.1%}[/{c}]")

    console.print(cat_table)
    
def _fmt(v: float | None) -> str:
    if v is None: return "—"
    bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
    color = "green" if v >= 0.7 else ("yellow" if v >= 0.4 else "red")
    return f"[{color}]{v:.2f}[/{color}] {bar}"


def print_summary(rows: list[AgentEvalRow]) -> None:
    # Note: a hybrid row (both tools called) appears in BOTH rag_rows and
    # sql_rows below — this is intentional, since RAGAS metrics for that
    # row reflect contributions from both context sources. hybrid_rows is
    # reported separately so the count is not silently double-hidden.
    rag_rows    = [r for r in rows if r.is_rag_routed]
    sql_rows    = [r for r in rows if r.is_sql_routed]
    hybrid_rows = [r for r in rows if r.is_hybrid_routed]
    none_rows   = [r for r in rows if not r.tools_used]

    console.print(f"\n[bold]Total rows:[/bold] {len(rows)}")
    console.print(
        f"  RAG-routed (incl. hybrid): [cyan]{len(rag_rows)}[/cyan]  "
        f"SQL-routed (incl. hybrid): [yellow]{len(sql_rows)}[/yellow]  "
        f"Hybrid (both): [magenta]{len(hybrid_rows)}[/magenta]  "
        f"Direct: [dim]{len(none_rows)}[/dim]"
    )

    # ── Routing accuracy (new section) ─────────────────────────────────────
    print_routing_accuracy(rows)

    # ── Per-tool RAGAS summary ─────────────────────────────────────────────
    ragas_table = Table(title="RAGAS scores by tool path", show_lines=True)
    ragas_table.add_column("Metric")
    ragas_table.add_column("RAG tool",  justify="right")
    ragas_table.add_column("SQL tool",  justify="right")
    ragas_table.add_column("Combined",  justify="right")

    def mean(rows_: list, attr: str) -> Optional[float]:
        vals = [getattr(r, attr) for r in rows_ if getattr(r, attr) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    for metric_name, applies_rag, applies_sql in METRICS:
        rag_score = mean(rag_rows, metric_name) if applies_rag  else None
        sql_score = mean(sql_rows, metric_name) if applies_sql  else None
        all_score = mean(rows,     metric_name)
        ragas_table.add_row(
            metric_name.replace("_", " ").title(),
            _fmt(rag_score),
            _fmt(sql_score),
            _fmt(all_score),
        )

    console.print(ragas_table)

    # ── SQL health metrics ─────────────────────────────────────────────────
    if sql_rows:
        console.print("\n[bold]SQL tool health:[/bold]")
        for attr, label in SQL_METRICS:
            vals  = [getattr(r, attr) for r in sql_rows if getattr(r, attr) is not None]
            pct   = sum(vals) / len(vals) * 100 if vals else 0
            color = "green" if pct >= 90 else ("yellow" if pct >= 70 else "red")
            console.print(
                f"  {label}: [{color}]{pct:.0f}%[/{color}] ({sum(vals)}/{len(vals)})"
            )

    # ── Per-category RAGAS breakdown ───────────────────────────────────────
    cat_table = Table(title="RAGAS mean score by category", show_lines=True)
    cat_table.add_column("Category")
    cat_table.add_column("N",          justify="right")
    cat_table.add_column("Tool used")
    cat_table.add_column("Mean score", justify="right")

    by_cat: dict[str, list[AgentEvalRow]] = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)

    for cat in sorted(by_cat):
        cat_rows   = by_cat[cat]
        tools_used = sorted(set(r.tool_used for r in cat_rows))
        cat_table.add_row(
            cat,
            str(len(cat_rows)),
            ", ".join(tools_used),
            _fmt(mean(cat_rows, "mean_score")),
        )

    console.print(cat_table)


def save_results(rows: list[AgentEvalRow], output_path: Path) -> None:
    records = []
    for r in rows:
        records.append({
            "category":              r.category,
            "expected_tool":         r.expected_tool,
            "tools_used":            "+".join(r.tools_used) if r.tools_used else "none",
            # e.g. "nba_stats_sql+nba_knowledge_rag" for a correctly-handled
            # hybrid row — preserves full information
            "is_hybrid":             r.is_hybrid_routed,
            "routing_correct":       r.routing_correct,
            "routing_error_type":    r.routing_error_type,
            "question":              r.question,
            "ground_truth":          r.ground_truth,
            "answer":                r.answer,
            "sql_query":             r.sql_query,
            "sql_execution_success": r.sql_execution_success,
            "result_non_empty":      r.result_non_empty,
            "faithfulness":          r.faithfulness,
            "answer_relevancy":      r.answer_relevancy,
            "context_recall":        r.context_recall,
            "context_precision":     r.context_precision,
            #"answer_correctness":    r.answer_correctness,
            "mean_score":            r.mean_score,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    console.print(f"\n[green]CSV saved:[/green] {output_path}")

    # Excel with five sheets for the mission report
    excel_path = output_path.with_suffix(".xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="All", index=False)

        # contains() matching on the "+"-joined tools_used string —
        # correctly includes hybrid rows in BOTH the RAG and SQL sheets,
        # since a hybrid row genuinely used both tools.
        df[df.tools_used.str.contains("nba_knowledge_rag", na=False)].to_excel(
            writer, sheet_name="RAG_rows", index=False
        )
        df[df.tools_used.str.contains("nba_stats_sql", na=False)].to_excel(
            writer, sheet_name="SQL_rows", index=False
        )
        # Hybrid rows isolated on their own sheet — these are exactly the
        # rows the original "both" bug silently mis-evaluated.
        df[df.is_hybrid == True].to_excel(
            writer, sheet_name="Hybrid_rows", index=False
        )

        # Routing errors sheet — most useful for critical analysis
        routing_rows = df[
            df["routing_correct"].notna() & (df["routing_correct"] == False)
        ]
        routing_rows.to_excel(writer, sheet_name="Routing_errors", index=False)

    console.print(f"[green]Excel saved:[/green] {excel_path} (5 sheets)")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the two-tool NBA agent.")
    parser.add_argument("--testset", default="eval_agent_sql-rag_dataset.json", help="Path to eval_dataset.json")
    parser.add_argument("--output", default="eval_agent_results.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level   = logging.INFO if args.verbose else logging.WARNING,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── Validate config ──────────────────────────────────────────────────
    config = PipelineConfig()
    console.print(f"[cyan]Config:[/cyan] {config.model_dump()}")

    # ── Load testset ───────────────────────────────────────────────────────
    BASE_DIR = Path(__file__).parent.parent
    agent_eval_result_path = BASE_DIR/f"evaluation/eval_artifacts/{args.output}"
    testset_path = BASE_DIR/f"evaluation/eval_artifacts/{args.testset}"
    with open(testset_path, encoding="utf-8") as f:
        raw_rows = json.load(f)

    eval_rows: list[AgentEvalRow] = [
        AgentEvalRow(
            question      = r["question"],
            ground_truth  = r["ground_truth"],
            category      = r.get("category", "general"),
            expected_tool = r.get("expected_tool", None),
        )
        for r in raw_rows
    ]

    n_with_exp = sum(1 for r in eval_rows if r.expected_tool is not None)
    no_expct_tool = sum(1 for r in eval_rows if r.expected_tool is None)
    console.print(
        f"[cyan]Testset:[/cyan] {len(eval_rows)} questions  "
        f"| with expected_tool: [yellow]{n_with_exp}[/yellow]  "
        f"| No expectation: [dim]{no_expct_tool}[/dim]"
    )

    # ── Build agent (generator = Mistral, embeddings = Mistral) ───────────
    console.rule("Step 1 — Building agent")
    # ── Initialize agent ──────────────────────────────────────────────────
    agent  = build_agent(db_uri=DATABASE_URL, mistral_api_key=MISTRAL_API_KEY,agent_model=config.mistral_model, 
                         sql_model=config.sql_generator_model, embed_model=config.embed_model,
                         top_k_rag=config.top_k, verbose=True)

    # ── Run agent on every question ────────────────────────────────────────
    console.rule("Step 2 — Agent invocation")
    for i, row in enumerate(tqdm(eval_rows, desc="Agent inference")):
        populated = invoke_agent(agent, row.question, expected_tool=row.expected_tool)
        row.answer                = populated.answer
        row.tools_used            = populated.tools_used   # list, not the old single string
        row.contexts              = populated.contexts
        row.sql_query             = populated.sql_query
        row.raw_sql_result        = populated.raw_sql_result
        row.sql_execution_success = populated.sql_execution_success
        row.result_non_empty      = populated.result_non_empty
        row.routing_correct       = populated.routing_correct
        row.routing_error_type    = populated.routing_error_type
        time.sleep(30)
    # Tool distribution — counts hybrid rows under a distinct "both" bucket
    tool_dist: dict[str, int] = {}
    for r in eval_rows:
        if r.is_hybrid_routed:
            key = "both"
        elif r.tools_used:
            key = r.tools_used[0]
        else:
            key = "none"
        tool_dist[key] = tool_dist.get(key, 0) + 1
    console.print(f"Tool distribution: {tool_dist}")

    # ── Ragas Evaluation on the whole dataset ───────────────────
    console.rule("Step 3 — RAGAS scoring")
    eval_rows = score_all_rows(rows=eval_rows, judge_model=config.judge_model,
                                judge_embed_model=config.judge_embedding_model, google_api_key=GEMINI_API_KEY)

    # ── Report & save ──────────────────────────────────────────────────────
    console.rule("Results")
    print_summary(eval_rows)
    save_results(eval_rows, agent_eval_result_path)


if __name__ == "__main__":
    main()