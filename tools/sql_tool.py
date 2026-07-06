"""
sql_tool.py
===========
Custom LangChain SQL tool that uses SQLDatabase for the connection
(schema introspection + safe query execution via .run()) while injecting
our own few-shot prompt for SQL generation.

Why not SQLDatabaseToolkit?
────────────────────────────
SQLDatabaseToolkit gives QuerySQLDataBaseTool, InfoSQLDatabaseTool,
ListSQLDatabaseTool, QuerySQLCheckerTool — but its prompt is fixed and
cannot be extended with few-shot examples. Our approach:
  • SQLDatabase.from_uri()     → connection, dialect, schema introspection
  • SQLDatabase.get_table_info → live schema string (no hardcoded constant)
  • SQLDatabase.run(sql)       → safe execution with result truncation
  • Our PromptTemplate         → few-shot examples injected at call time
  • BaseTool subclass          → full routing control for the ReAct agent

Architecture note — why SQL generation is inside the tool, not before
──────────────────────────────────────────────────────────────────────
The ReAct agent loop is:
  Thought → Action (tool name) → Action Input (NL question) → tool._run()

The agent LLM decides *which* tool to call (routing decision).
SQL generation only happens inside _run() because that is where the
schema and few-shot examples become available.  If generation happened
in the agent prompt, the full schema + few-shots would be injected on
every turn — even for RAG questions — wasting context budget.

  Agent LLM  →  "stats question → call nba_stats_sql"   (routing)
  Tool LLM   →  "schema + few-shots → generate SQL"     (generation)

Two separate LLM calls, two separate prompts.  The agent stays lean.

Context window strategy
──────────────────────
db.get_table_info() is called ONCE at tool construction and cached on
the instance as self._schema_info.  Every subsequent _run() call reads
from the cache — zero redundant API calls, always in sync with the live
schema, and about 600 tokens are NOT re-sent on every invocation.
"""

from __future__ import annotations
import hashlib
import logging
import logfire
import re
import time
from typing import Any, Optional, Type
from langchain_community.tools import BaseTool
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from tools.sql_tool_config import FEW_SHOT_EXAMPLES, SQL_PROMPT
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Retry helper — exponential backoff for Mistral 429 rate limit errors
# ─────────────────────────────────────────────────────────────────────────────
def _invoke_with_retry(
    llm:         Any,
    prompt:      str,
    max_retries: int   = 4,
    base_delay:  float = 10.0 #5.0,   # seconds — start conservative for free tier
) -> Any:
    """
    Call llm.invoke(prompt) with exponential backoff on 429 errors.

    Why this is needed
    ──────────────────
    When the agent calls the SQL tool twice in one turn (e.g. for a hybrid
    question), two sequential Mistral API calls fire within milliseconds of
    each other. The free-tier rate limit (roughly 2 req/s or ~60 req/min)
    triggers a 429 on the second call.

    Backoff schedule (base_delay=5s):
      attempt 1 → immediate
      attempt 2 → wait 5s
      attempt 3 → wait 10s
      attempt 4 → wait 20s
    Total max wait: ~35s — acceptable for an evaluation pipeline.

    For a production system, use a token-bucket rate limiter instead.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

            if not is_rate_limit:
                raise   # non-rate-limit errors should surface immediately

            wait = base_delay * (2 ** attempt)
            log.warning(
                "Rate limit hit (attempt %d/%d) — waiting %.0fs before retry. Error: %s",
                attempt + 1, max_retries, wait, err_str[:120],
            )
            time.sleep(wait)
            last_exc = exc

    raise RuntimeError(
        f"Mistral API rate limit exceeded after {max_retries} retries. "
        f"Last error: {last_exc}"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  1. build Few-shot examples 
# ─────────────────────────────────────────────────────────────────────────────

def _build_few_shot_block() -> str:
    """Format few-shot examples for prompt injection (built once, reused)."""
    lines = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES[:4], 1):
        lines.append(f"Example {i}:\nQ: {ex['q']}\nSQL:\n{ex['sql']}")
    return "\n\n".join(lines)

# Build the few-shot block once at import time — it never changes
_FEW_SHOT_BLOCK: str = _build_few_shot_block()

# ─────────────────────────────────────────────────────────────────────────────
#  NBAStatsSQLTool
# ─────────────────────────────────────────────────────────────────────────────
class _SQLInput(BaseModel):
    question: str = Field(description="Natural language question about NBA stats.")


class NBAStatsSQLTool(BaseTool):
    """
    LangChain tool that answers quantitative NBA questions.

    Rate-limit strategy (three complementary fixes)
    ─────────────────────────────────────────────────
    1. _invoke_with_retry()   — exponential backoff on 429 errors inside
                                the SQL generation LLM call.

    2. _sql_cache (dict)      — if the same question is asked twice in one
                                evaluation run, the generated SQL and its
                                result are returned from cache without any
                                API call. Keyed by SHA-256 of the question.

    3. min_call_interval      — a configurable minimum seconds between
                                consecutive LLM calls. Default 1.5s ensures
                                at most ~40 calls/min, safely under the limit.
                                Set to 0 to disable.

    Context window optimisation
    ────────────────────────────
    _schema_info is populated ONCE in build_sql_tool() via get_table_info().
    _few_shots_block is built once at import time.
    Neither is re-fetched or re-computed on subsequent _run() calls.
    
    Schema-awareness guard (fixes multi-retry on missing columns)
    ──────────────────────────────────────────────────────────────
    Some questions ask for data that structurally cannot exist in the schema
    (home/away splits, game logs, per-quarter stats). Without a guard, the
    agent tries multiple SQL queries, burns rate-limit quota, and still
    fails. _check_schema_feasibility() detects these patterns before any
    API call and returns an immediate informative response.
    """
   

    name:        str = "nba_stats_sql"
    description: str = (
        "Use for quantitative NBA questions: rankings, comparisons, "
        "shooting percentages, efficiency metrics (TS%, EFG%, PIE, net rating), "
        "per-game averages, multi-criteria filters (age, team, games played), "
        "aggregations by team. "
        "Input: a natural language question. "
        "Do NOT use for opinions, history, tactics, or qualitative discussion."
    )
    args_schema:       Type[BaseModel] = _SQLInput

    db:                Any   = Field(description="LangChain SQLDatabase instance")
    llm:               Any   = Field(description="LangChain LLM for SQL generation")
    min_call_interval: float = Field(default=1.5, description="Min seconds between LLM calls")

    # Private cache and timing — not Pydantic fields
    _schema_info:     str   = ""
    _few_shots_block: str   = ""
    _sql_cache:       dict  = {}      # question_hash → result string
    _last_call_time:  float = 0.0     # timestamp of last LLM call

    # Patterns that indicate the question asks for data not in the schema.
    # Checked BEFORE any API call — saves rate-limit quota and avoids
    # the agent looping on unanswerable SQL questions.
    _UNSUPPORTED_PATTERNS: list[tuple[str, str]] = [
        (
            r"home\s+and\s+away|home\s+vs\.?\s+away|away\s+game|home\s+game",
            "home/away split. The database contains only season-aggregate "
            "statistics — no home/away breakdown is stored in player_stats.",
        ),
        (
            r"last\s+\d+\s+game|past\s+\d+\s+game|game\s+log|game.by.game",
            "game-by-game log. The database stores season totals only, "
            "not individual game records.",
        ),
        (
            r"quarter|half.time|overtime|clutch.time|4th.quarter",
            "period/situation split. No per-quarter or clutch-time data "
            "exists in the database.",
        ),
        (
            r"last\s+season|previous\s+season|year.over.year|compared.to.last",
            "prior-season comparison. The database contains only the "
            "2024-25 season.",
        ),
        (
            r"playoff|post.season|finals|championship",
            "playoff data. The database contains regular-season statistics "
            "only — no playoff records.",
        ),
    ]

    model_config = {"arbitrary_types_allowed": True}

    def _check_schema_feasibility(self, question: str) -> Optional[str]:
        """
        Return an informative 'not available' string if the question asks
        for data that cannot exist in the schema — before calling the API.

        Returns None if the question looks answerable from the schema.
        """
        q_lower = question.lower()
        for pattern, explanation in self._UNSUPPORTED_PATTERNS:
            if re.search(pattern, q_lower):
                return (
                    f"This question cannot be answered from the database.\n"
                    f"Reason: it requires {explanation}\n"
                    f"Suggestion: rephrase to ask about season-aggregate "
                    f"statistics (totals, averages, rankings, percentages)."
                )
        return None

    def _rate_limit_pause(self) -> None:
        """
        Enforce minimum interval between consecutive sql_LLM calls.
        Prevents bursting two SQL generation calls back-to-back
        which reliably triggers 429 on the Mistral free tier.
        """
        if self.min_call_interval <= 0:
            return
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_call_interval:
            pause = self.min_call_interval - elapsed
            log.debug("Rate-limit pause: %.2fs", pause)
            time.sleep(pause)
    @logfire.instrument("trace_sql_tool")
    def _run(self, question: str) -> str:
        # ── Guard: detect structurally unanswerable questions ─────────────
        not_available = self._check_schema_feasibility(question)
        if not_available:
            logfire.info("Schema feasibility check blocked query: %s", question[:60])
            return not_available

        # ── Cache lookup ──────────────────────────────────────────────────
        # If the exact same question was already answered this session,
        # return the cached result without any API call.
        cache_key = hashlib.sha256(question.encode()).hexdigest()
        if cache_key in self._sql_cache:
            logfire.info("SQL cache hit for: %s", question[:60])
            return self._sql_cache[cache_key]

        # ── Step 1: build prompt using cached schema ──────────────────────
        prompt_str = SQL_PROMPT.format(
            schema    = self._schema_info,
            few_shots = self._few_shots_block,
            question  = question,
        )

        # ── Step 2: enforce rate limit then generate SQL ──────────────────
        self._rate_limit_pause()
        try:
            response = _invoke_with_retry(self.llm, prompt_str)
            self._last_call_time = time.time()
            sql = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logfire.error(f"SQL generation failed: {exc}")
            return f"SQL generation failed: {exc}"

        # Strip accidental markdown fences
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*",   "", sql)
        sql = sql.strip()

        logfire.info("Generated SQL:",generated_sql=sql)

        # Safety: block non-SELECT statements
        first_word = sql.split()[0].upper() if sql.strip() else ""
        if first_word not in ("SELECT", "WITH"):
            return f"Blocked non-SELECT query: {sql[:120]}"

        # ── Step 3: execute via SQLDatabase.run() ────────────────────────
        try:
            result = self.db.run(sql)
        except Exception as exc:
            logfire.warning("SQL execution error: %s\nQuery: %s", exc, sql)
            return f"SQL execution error: {exc}\nGenerated query:\n{sql}"

        if not result or result.strip() == "":
            logfire.warning(f"No results found.\nQuery:\n{sql}")
            return f"No results found.\nQuery:\n{sql}"

        output = f"SQL:\n{sql}\n\nRESULTS:\n{result}"
        logfire.info("Sql tool found results for the following question ",query_snippet = question[:60], returned= output)
        # ── Cache the result ──────────────────────────────────────────────
        self._sql_cache[cache_key] = output
        return output

    async def _arun(self, question: str) -> str:
        return self._run(question)


# ─────────────────────────────────────────────────────────────────────────────
#   Factory — the only place that calls db.get_table_info()
# ─────────────────────────────────────────────────────────────────────────────

def build_sql_tool(
    db_uri:              str,
    sql_llm:                 Any,
    include_tables:      Optional[list[str]] = None,
    sample_rows:         int                 = 2,
    min_call_interval:   float               = 3.0 #3s between calls → ~20 calls/min #old value 1.5,
) -> NBAStatsSQLTool:
    """
    Construct an NBAStatsSQLTool with rate-limit protection.

    Parameters
    ----------
    db_uri              : SQLAlchemy URI e.g. "sqlite:///nba.db"
    sql_llm             : LLM used ONLY for SQL generation.
                          Use a small/fast model (mistral-small-latest)
                          to keep this off the agent's rate-limit bucket.
                          The agent's main LLM (mistral-large-latest) handles
                          reasoning; only SQL generation uses sql_llm.
    include_tables      : Restrict schema introspection (default: all 4 tables)
    sample_rows         : Rows shown per table in schema info (default 2)
    min_call_interval   : Min seconds between sql_llm calls (default 2.0s)

    Rate-limit protection applied
    ──────────────────────────────
    Three layers work together:
      1. min_call_interval  — prevents back-to-back bursts
      2. _invoke_with_retry — recovers from 429 with exponential backoff
      3. _sql_cache         — eliminates duplicate API calls entirely

    Example
    -------
    >>> from langchain_mistralai import ChatMistralAI
    >>> sql_llm  = ChatMistralAI(model="mistral-large-latest", temperature=0)
    >>> tool = build_sql_tool("sqlite:///nba.db", sql_llm)
    >>> tool._run("Top 5 scorers per game?")
    """
    tables = include_tables or ["teams", "players", "player_stats"]

    # SQLDatabase.from_uri:
    #   • Connects via SQLAlchemy (SQLite, PostgreSQL, MySQL…)
    #   • Introspects schema automatically
    #   • Provides .run(sql) and .get_table_info()
    db = SQLDatabase.from_uri(
        db_uri,
        include_tables            = tables,
        sample_rows_in_table_info = sample_rows,
    )

    # ── Fetch schema info ONCE — cache it on the tool instance ────────────
    # get_table_info() returns CREATE TABLE statements + sample rows.
    # Calling it here (not in _run) means the DB is queried once at startup,
    # not on every user question.
    schema_info = db.get_table_info()
    log.info(
        "SQLDatabase connected: %s | schema: %d chars",
        db_uri, len(schema_info),
    )

    tool = NBAStatsSQLTool(db=db, llm=sql_llm, min_call_interval=min_call_interval)
    tool._schema_info     = schema_info
    tool._few_shots_block = _FEW_SHOT_BLOCK
    tool._sql_cache       = {}
    # indicate the time of last llm api call
    tool._last_call_time  = 0.0

    return tool
