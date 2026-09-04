"""
agent_react.py
========
Assembles the NBA analytics agent with two tools:
  • NBAStatsSQLTool   — quantitative questions → SQLDatabase.run()
  • NBAKnowledgeTool  — qualitative questions → CosineRetriever (FAISS)

The agent uses ReAct (Reason + Act) — at each step it decides:
  Thought:  is this question quantitative or qualitative?
  Action:   nba_stats_sql  OR  nba_knowledge_rag
  Observation: tool result
  Final Answer: synthesised natural language response

Model split strategy (avoids 429 rate-limit errors)
────────────────────────────────────────────────────
  agent_llm  (mistral-small-latest) — reasoning + final synthesis
  sql_llm    (mistral-large-latest) — SQL generation inside tool only

These hit separate Mistral rate-limit buckets, so a double-tool-call
turn no longer exhausts the quota of a single model.

Iteration limit fix
────────────────────
  max_iterations = 8:
    1. Thought + Action (tool 1)
    2. Observation
    3. Thought + Action (tool 2)
    4. Observation
    5. Thought + Final Answer
  Each Thought→Action counts as one iteration in LangChain's counter.
  4 was too tight for any multi-tool question.

  early_stopping_method = "generate" — when the iteration limit is
  reached the agent is asked to produce a Final Answer from whatever
  it has so far, instead of returning the unhelpful
  "Agent stopped due to iteration limit" string.

Usage
-----
    from agent_react import build_agent
    agent = build_agent(
        db_uri         = "sqlite:///nba.db",
        faiss_index_dir= "./faiss_store",
        mistral_api_key= "...",
    )
    result = agent.invoke({"input": "Who leads the league in assists per game?"})
    print(result["output"])
"""

from __future__ import annotations
import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from utils.models import PipelineConfig
from utils.config import  MISTRAL_API_KEY, DATABASE_URL
from tools.rag_tool import build_rag_tool
from tools.sql_tool import build_sql_tool

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ReAct prompt — instructs the LLM when to use which tool
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert NBA analytics assistant for coaches and scouts.
Your goal is to answer the user's question accurately by using the available tools whenever necessary.
========================
AVAILABLE TOOLS
========================
{tools}
========================
WHEN TO USE EACH TOOL
========================
1. nba_stats_sql
Use this tool whenever the question requires structured or numerical information, including:
- player statistics
- team statistics
- rankings
- comparisons
- averages
- totals
- percentages
- shooting efficiency
- rebounds
- assists
- wins/losses
- filters
- leaderboards
- any information stored in the SQL database

2. nba_knowledge_rag
Use this tool whenever the question requires textual knowledge, including:
- scouting reports
- tactical analysis
- play style
- Reddit discussions
- fan opinions
- narratives
- historical context
- qualitative explanations

========================
DECISION RULES
========================

- Numerical questions → use nba_stats_sql.
- Qualitative questions → use nba_knowledge_rag.
- Never invent statistics. Never invent facts.
- Use only information returned by the tools.
- If the tool returns "SQL execution error" or "SQL generation failed",
  you MAY call nba_stats_sql once more with a corrected or simpler version
  of the question. Do not retry more than once.
- If the tool returns "not available", "no home/away split", "no game log",
  or any message explaining the data does not exist in the database,
  do NOT retry. Report the limitation clearly in your Final Answer.

If both tools are required:
1. Call the first tool and wait for the Observation.
2. Decide whether another tool is needed.                                         
3. If needed, call the second tool and wait for the Observation.
4. Produce the Final Answer using both results.

Never call more than one tool in the same assistant message.

========================
OUTPUT FORMAT — follow EXACTLY
========================
When you need a tool:

Thought: <one sentence explaining what you need>
Action: <one of [{tool_names}]>
Action Input: <the question or input for the tool>

After receiving the Observation, either call another tool or finish:

Thought: I now have enough information to answer.
Final Answer: <your answer, citing the data from the tools>

STRICT RULES:
- Every Thought MUST be followed immediately by Action or Final Answer.
- Never write Thought without Action or Final Answer after it.
- Never invent an Action name — use only [{tool_names}].
- Never produce a Final Answer before using a required tool.
- If the tool says data is not available, say so — do not retry.
- Every Thought MUST be followed immediately by Action or Final Answer — never both.
- NEVER write "Final Answer:" in the same message as "Action:".
- If you are not 100% certain you have enough information, choose Action — not Final Answer.
- Once you write "Final Answer:", stop generating immediately. Do not add anything after it.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Create either single invoke or Chat prompt
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(chat=False):

    if chat:
        return PromptTemplate.from_template(
            SYSTEM_PROMPT + """
            Conversation so far:
            {chat_history}
            Current question:
            {input}
            {agent_scratchpad}
            """)
    return PromptTemplate.from_template(
        SYSTEM_PROMPT
        + """
        Question: {input}
        {agent_scratchpad}
        """)
# ─────────────────────────────────────────────────────────────────────────────
# CosineRetriever import — lazy to avoid circular imports
# ─────────────────────────────────────────────────────────────────────────────

def _load_cosine_retriever(embeddings: MistralAIEmbeddings):
    """
    Import CosineRetriever from the rag pipeline and load the
    persisted FAISS index from disk.

    CosineRetriever.__init__ handles the load automatically when no
    index object is passed — it calls _load_index() internally and
    raises RuntimeError with a clear message if files are missing.
    """
    try:
        from rag.retriever import CosineRetriever
    except ImportError:
        raise ImportError(
            "evaluate_ragas.py not found. Make sure it is in the Python path. "
            "The CosineRetriever and FAISS index logic lives there."
        )
    return CosineRetriever(embeddings=embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def build_agent(
    db_uri:          str,
    mistral_api_key: str,
    agent_model:     str   = "mistral-small-latest",
    sql_model:       str   = "mistral-large-latest",
    embed_model:     str = "mistral-embed",
    top_k_rag:       int = 4,
    max_iterations:  int   = 8,
    chat:            bool = False,
    verbose:         bool = True,
) -> AgentExecutor:
    """
    Build the two-tool NBA agent with a split LLM strategy.

    Parameters
    ──────────────────────
    agent_model  (mistral-small-latest)
        • Agent reasoning: reads the question, decides which tool to call
        • Final answer synthesis: reads tool outputs, writes the response
        • Rate-limit bucket: shared with agent reasoning calls only

    sql_model    (mistral-large-latest)
        • SQL generation inside NBAStatsSQLTool._run() only
        • Structured & deterministic task (schema + few-shots → SQL)
        • Rate-limit bucket: SEPARATE from agent_model
          → double-tool-call turns now consume from two different buckets
          → 429 errors on back-to-back SQL calls are eliminated


    Parameters
    ----------
    db_uri            : SQLAlchemy URI for the NBA database
                        e.g. "sqlite:///nba.db"
    mistral_api_key   : Mistral API key (used for generator + embeddings)
    agent_model     : Main reasoning model (default: mistral-small-latest)
    sql_model       : SQL generation model (default: mistral-large-latest)
    embed_model       : Mistral embedding model (default: mistral-embed)
    top_k_rag         : Number of FAISS chunks retrieved per query (default 4)
    max_iterations  : Max ReAct iterations before forced final answer (default 8)
                      Breakdown for a worst-case hybrid question:
                        iter 1 — Thought + Action (sql tool call 1)
                        iter 2 — Thought + Action (rag tool call)
                        iter 3 — Thought + Final Answer
    chat      :  boolean value determine the type of agent either "False" to invoke with one user input
                       or "True" to pass user input with list of messages to the agent
    verbose           : Print agent reasoning steps (default True)

    Returns
    -------
    AgentExecutor — call with .invoke({"input": "your question"})

    Notes
    -----
    - The FAISS index must already be built and saved to ./vector_db/
      First run indexing.py first to chunck & embed corpus and build Faiss index.
    - SQLDatabase connects to the URI on construction — ensure the DB
      file exists (run load_excel_to_db.py to load excel file into Sqlit DB).
    """

    # ── Agent reasoning LLM (large — quality matters for routing) ─────────
    agent_llm = ChatMistralAI(
        model       = agent_model,
        api_key     = mistral_api_key,
        temperature = 0,
        streaming = False,   # ← disables chunk-by-chunk generation
    )

    # ── SQL generation LLM (small — separate rate-limit bucket) ───────────
    sql_llm = ChatMistralAI(
        model       = sql_model,
        api_key     = mistral_api_key,
        temperature = 0,   # deterministic SQL generation
        streaming = False,   # ← disables chunk-by-chunk generation
    )
    print("sql model is :", sql_model)
    # ── Embeddings (for FAISS cosine retrieval) ────────────────────────────
    embeddings = MistralAIEmbeddings(
        model   = embed_model,
        api_key = mistral_api_key,
    )

    # ── Tool 1: SQL ────────────────────────────────────────────────────────
    # build_sql_tool creates SQLDatabase.from_uri() internally
    # SQLDatabase handles:
    #   • Connection pooling via SQLAlchemy
    #   • Schema introspection (get_table_info)
    #   • Safe query execution (run method)
    #   • Result string truncation (max_string_length)
    sql_tool = build_sql_tool(
        db_uri = db_uri,
        sql_llm    = sql_llm,
        min_call_interval = 3.0,
    )
    log.info("SQL tool ready: %s", db_uri)

    # ── Tool 2: RAG ────────────────────────────────────────────────────────
    # CosineRetriever loads FAISS index from disk in __init__
    # Raises RuntimeError with clear message if ./faiss_store/ is missing
    retriever = _load_cosine_retriever(embeddings)
    rag_tool  = build_rag_tool(retriever, top_k=top_k_rag)
    log.info("RAG tool ready: %d vectors in index", retriever.index.index.ntotal)

    # ── Agent assembly ─────────────────────────────────────────────────────
    tools = [sql_tool, rag_tool]
    prompt = build_prompt(chat)
    agent = create_react_agent(llm=agent_llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent                = agent,
        tools                = tools,
        verbose              = verbose,
        max_iterations       = max_iterations,
        #handle_parsing_errors= True,    # recover from malformed LLM output
        handle_parsing_errors="Check your output and make sure it conforms to the format — use either Action or Final Answer, never both in the same response.",
        return_intermediate_steps=True, # expose tool calls in result dict
        early_stopping_method="generate",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    db_uri = sys.argv[1] if len(sys.argv) > 1 else DATABASE_URL
    #q      = sys.argv[2] if len(sys.argv) > 2 else "Who has the best net rating this season?"
    key    = MISTRAL_API_KEY

    if not key:
        print("Set MISTRAL_API_KEY environment variable.")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = PipelineConfig()
    agent  = build_agent(db_uri=db_uri, mistral_api_key=key,agent_model=config.mistral_model, 
                         sql_model=config.sql_generator_model, embed_model=config.embed_model, top_k_rag=config.top_k, verbose=True)
    """
    result = agent.invoke({"input": q})
    print("\n" + "═" * 60)
    print("QUESTION:", q)
    print("─" * 60)
    print("ANSWER:", result["output"])
    print("─" * 60)
    if result.get("intermediate_steps"):
        for action, obs in result["intermediate_steps"]:
            print(f"TOOL USED: {action.tool}")
            print(f"TOOL INPUT: {action.tool_input}")
            print(f"OBSERVATION (truncated): {str(obs)[:200]}")
            print()
            """
    print("\n" + "="*70)
    print("🤖 NBA Reddit RAG & Sql AGENT")
    print("="*70)
    print("💬 Ask me about NBA teams or matches using natural language!")
    print("\nExample queries:")
    print("  • 'Which player has the highest 3-point percentage?'")
    print("  • 'Which team has the best offensive rating?")
    print("  • 'Which team has the most rebounds?'")
    print("  • 'Which players shoot above 40% from three?'")
    print("\nCommands:")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)
        
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("🤖 Bot: Please ask me any question about NBA !!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🤖 Bot: Thank you for using the RAG Reddit RAG AGENT!")
                print("      Hope you found some interesting information! 👋")
                break
            else:
                # Process the user query with RAG Agent
                result = agent.invoke({"input": user_input})
                print("ANSWER:", result["output"])

        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Hope you find what you looking for! 👋")
            break
        except Exception as e:
            print(f"❌ Bot: Sorry, I encountered an error: {e}")
