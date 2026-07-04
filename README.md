# NBA Analytics Assistant — RAG + SQL Agent

> An intelligent assistant for NBA coaches and scouts combining Retrieval-Augmented Generation (RAG) over Reddit discussion corpora with a structured SQL query tool over season statistics — powered by Mistral and evaluated with RAGAS.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [RAG Pipeline Methodology](#3-rag-pipeline-methodology)
4. [SQL Tool Methodology](#4-sql-tool-methodology)
5. [Agent Design](#5-agent-design)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Project Structure](#7-project-structure)
8. [Setup & Installation](#8-setup--installation)
9. [Usage](#9-usage)
10. [Known Limitations](#10-known-limitations)

---

## 1. Project Overview

This project addresses a real problem for NBA coaching staff: **qualitative insights and statistical data live in completely separate places**. Fan discussions, analyst commentary, and tactical narratives are scattered across Reddit threads; structured performance metrics live in spreadsheets. Neither source alone answers a question like *"Is Randle's 45% three-point shooting reflected in how fans perceive his improvement?"*

The assistant solves this by routing each question to the appropriate knowledge source through a two-tool ReAct agent:

| Question type | Source | Tool |
|---|---|---|
| Opinions, tactics, fan sentiment, historical context | Reddit corpus (FAISS index) | `nba_knowledge_rag` |
| Rankings, percentages, averages, multi-criteria filters | NBA season statistics (SQLite) | `nba_stats_sql` |
| Hybrid questions | Both sources | Both tools sequentially |

---

## 2. Project Structure

```
assistant_rag_mistral/
│
├── agent_react.py          # Main entry point: ReAct agent (RAG + SQL tools)
├── MistralChat.py          # Streamlit application
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Locked dependency versions
├── README.md
├── .env                    # API keys (MISTRAL_API_KEY, GOOGLE_API_KEY)
│
├── evaluation/             # RAGAS evaluation scripts and reports
│   ├── evaluate_ragas.py   # Agent RAG-only evaluation
│   ├── evaluate_agent.py   # Two-tool agent evaluation with routing accuracy
|   └── eval_artifact/
│      ├── eval_agent_sql-rag_dataset.json  # Sql + Rag 21-question testset
│      └── evaluation_dataset_rag.json      # RAG-only 26-question subset
│
├── inputs/
│   ├── inputs_rag_tool/    # PDFs and documents to be indexed by RAG pipeline
│   └── inputs_sql_tool/    # Excel files to be imported into SQLite DB
│
├── rag/                    # Retrieval-Augmented Generation pipeline
│   ├── __init__.py
│   ├── indexer.py          # Document loading, chunking, FAISS index build
│   ├── retriever.py        # CosineRetriever (normalized IndexFlatIP)
│   └── generator.py        # Answer generation
│
├── tools/                  # LangChain tools used by the agent
│   ├── __init__.py
│   ├── rag_tool.py         # NBAKnowledgeTool (BaseTool wrapping CosineRetriever)
│   ├── sql_tool.py         # NBAStatsSQLTool (BaseTool wrapping SQLDatabase)
│   └── sql_tool_config.py  # SQL prompt, few-shot examples
│
├── sqlite_db/              # SQLite database
│   ├── nba.db
│   └── schema.sql          # DDL written on first ingestion run
│
├── utils/
│   ├── __init__.py
│   ├── config.py           # Global configuration (paths, model names, chuck size ..)
│   ├── db.py               # SQLite schema DDL constants
│   ├── models.py           # Pydantic v2 validation models
│   ├── load_data.py        # Multi-format document loader (PDF, TXT, CSV, XLSX, DOCX)
│   └── load_excel_to_db.py # Excel → SQLite ingestion with Pydantic validation
│
└── vector_db/              # FAISS vector index (persisted)
    ├── index.faiss         # Binary FAISS index (IndexFlatIP, L2-normalized)
    ├── index.pkl           # LangChain docstore (id → Document mapping)
    ├── chunks.json         # ValidatedChunk metadata (full Pydantic schema)
    └── corpus.hash         # SHA-256 fingerprint of corpus files (cache invalidation)
```

---
## 3. Architecture

```
User question
      │
      ▼
┌─────────────────────────────────┐
│   ReAct Agent  (agent_react.py) │
│   Mistral Small — reasoning     │
└──────────┬──────────────────────┘
           │ routes to 
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│RAG Tool │  │ SQL Tool │
│rag_tool │  │sql_tool  │
└────┬────┘  └────┬─────┘
     │            │
     ▼            ▼
┌─────────┐  ┌──────────────────────┐
│  FAISS  │  │ Few-shot SQL prompt  │
│  index  │  │ (Mistral Large)      │
│vector_db│  └────────┬─────────────┘
└────┬────┘           │
     │                ▼
     │         ┌─────────────┐
     │         │  SQLite DB  │
     │         │  sqlite_db/ │
     │         └─────────────┘
     │                      │
     │                      |
     ▼                      |        
Cosine retrieval            |
(L2-normalized IndexFlatIP) |   
      │                     ▼
      └─────────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Mistral Small    │
               │ Final synthesis  │
               └──────────────────┘
                         │
                         ▼
                   Final answer
```

**Key design decision — split LLM strategy:**
- `mistral-small-latest` handles agent reasoning and final answer synthesis
- `mistral-large-latest` handles SQL generation inside the tool only

This separates the two models onto different rate-limit buckets, eliminating 429 errors when the agent calls the SQL tool multiple times in a single turn.

---

## 4. RAG Pipeline Methodology

### 4.1 Corpus

The RAG knowledge base consists of Reddit `r/nba` discussion threads saved as PDF exports. Four threads are indexed at project launch:

| File | Topic | Pages |
|---|---|---|
| `Reddit_1.pdf` | Playoff teams impressions | 15 |
| `Reddit_2.pdf` | Media bias / Finals ratings debate | 23 |
| `Reddit_3.pdf` | Reggie Miller playoff efficiency analysis | 36 |
| `Reddit_4.pdf` | Reddit discussion | 10 |

Additional documents (`.txt`, `.csv`, `.xlsx`, `.docx`) are supported by the generic loader in `utils/load_data.py`.

### 4.2 Document Loading & Cleaning

**Location:** `utils/load_data.py`, `rag/indexer.py`

Each PDF goes through a two-stage loading strategy:

```
PDF file
  │
  ├─► Stage 1: PyPDF2 text extraction (fast)
  │     └─► if extracted chars < 100 → fall back to Stage 2
  │
  └─► Stage 2: PyMuPDF + EasyOCR (image-based PDFs)
        └─► 2x resolution render → readtext() → confidence filter ≥ 0.3
```

After extraction, Reddit-specific noise is removed by `clean_reddit_text()`:

- French navigation labels (`Accéder au contenu principal`, `Se connecter`)
- Upvote/downvote arrows and counts (`↑ 186 ↓`)
- Sponsored content blocks
- Reddit UI strings (`Répondre`, `Partager`, `Comm. du top 1%`)
- Deleted/removed placeholders (`[supprimé]`, `[removed]`)
- Footer URLs and page number artifacts

### 4.3 Chunking

**Location:** `rag/indexer.py`

`RecursiveCharacterTextSplitter` from LangChian splits cleaned documents at paragraph and sentence boundaries:

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 500 tokens | Large enough to preserve comment context; small enough for precision |
| `chunk_overlap` | 80 tokens | Prevents answers being split across chunk boundaries |
| `separators` | `["\n\n", "\n", ". ", " "]` | Prioritizes semantic boundaries over arbitrary character counts |
| Min chunk length | 80 chars | Rejects near-empty pages (ads, nav bars) |

Each chunk is validated by the `ValidatedChunk` Pydantic model before entering the index, ensuring:
- Content minimum length of 50 characters post-cleaning
- Valid source metadata (non-empty source name, page ≥ 0)
- Topic tag inferred from filename (`playoffs`, `player_stats`, `media_bias`, etc.)

### 4.4 Embedding & Indexing

**Location:** `rag/indexer.py`, `vector_db/`

Embeddings are generated via `mistral-embed` (1024-dimensional dense vectors) in batches of 32 to respect API rate limits.

**Critical implementation detail — cosine similarity correctness:**

LangChain's default `FAISS.from_documents()` builds an `IndexFlatL2` (Euclidean distance). Mistral embeddings are optimized for cosine similarity. The pipeline uses a manual construction to guarantee correctness:

```python
# 1. Embed all chunks
matrix = np.array(all_embeddings, dtype="float32")

# 2. Normalize to unit length — makes inner product = cosine
faiss.normalize_L2(matrix)

# 3. Build index with inner product (= cosine on unit vectors)
index = faiss.IndexFlatIP(dimension)
index.add(matrix)
```

The index and chunk metadata are persisted to `vector_db/` and loaded via a SHA-256 corpus fingerprint that invalidates the cache whenever corpus files change (any supported file type, not just PDFs).

### 4.5 Retrieval — CosineRetriever

**Location:** `rag/retriever.py`

`CosineRetriever` wraps the FAISS index with query-time normalization:

```python
# LangChain's similarity_search() does NOT normalize the query vector.
# Raw embed_query() vector + unit stored vectors → NOT cosine similarity.
# CosineRetriever fixes this:

raw_vec = embeddings.embed_query(question)
vec = np.array([raw_vec], dtype="float32")
faiss.normalize_L2(vec)                     # normalize query to unit length
scores, indices = index.index.search(vec, k) # now: inner product = cosine ✓
```

Each retrieved document has its `cosine_score` attached to `metadata` for traceability in Logfire and in the evaluation pipeline.

`CosineRetriever.__init__` loads the index from disk automatically when no index object is passed — raising a descriptive `RuntimeError` if `vector_db/` is missing rather than failing silently later.

### 4.6 Pydantic Validation Boundaries

The RAG pipeline applies Pydantic v2 validation at three distinct boundaries:

| Boundary | Model | What it catches |
|---|---|---|
| Chunks entering FAISS | `ValidatedChunk` | Chunks < 50 chars, empty source field |
| Chunk metadata | `ChunkMetadata` | Invalid page number, empty topic tag |
| RAGAS scores exiting evaluation | `EvalResult` | Scores outside [0, 1], missing fields |

Cross-field validation is applied where applicable (e.g. `chunk_overlap < chunk_size`).

---

## 5. SQL Tool Methodology

### 5.1 Database Schema

**Location:** `utils/db.py`, `sqlite_db/nba.db`

Three normalized tables store the 2024-25 NBA regular season data:

```sql
teams        (team_id PK, code UNIQUE, full_name)
players      (player_id PK, name UNIQUE, team_id FK, age)
player_stats (stat_id PK, player_id FK, season,
              gp, w, l, min_per_game,
              pts, fgm, fga, fg_pct,
              min_after_15, three_pa,
              ftm, fta, ft_pct, three_p_pct,
              oreb, dreb, reb, ast, tov, stl, blk, pf,
              fp, dd2, td3, plus_minus,
              off_rtg, def_rtg, net_rtg,
              ast_pct, ast_to, ast_ratio,
              oreb_pct, dreb_pct, reb_pct,
              to_ratio, efg_pct, ts_pct, usg_pct,
              pace, pie, poss)

```
<!--reports      (report_id PK, player_id FK,generated_at, content, report_type)-->
### 5.2 Ingestion Pipeline

**Location:** `utils/load_excel_to_db.py`

Excel → SQLite ingestion applies `PlayerStatsRow` Pydantic validation on every row:

- Age bounds: 18 ≤ age ≤ 55
- Games played bounds: 0 ≤ gp ≤ 82
- Cross-field validator: W + L ≤ GP (physically impossible otherwise)
- Team code normalization to uppercase 3-letter codes

### 5.3 SQL Generation — Few-Shot Prompt

**Location:** `tools/sql_tool.py`

SQL queries are generated by `mistral-large-latest` using a few-shot prompt injected with:

1. **Live schema** from `db.get_table_info()` — called once at tool construction and cached on the instance. Never re-fetched per query (saves ~600 tokens per invocation).
2. **4 few-shot examples** — built once at module import time and cached as a module-level constant. 
<!--Cover: simple ranking, per-game averages, multi-criteria filters, team aggregation, ratio metrics, player lookup, defensive metrics, advanced efficiency.-->

### 5.4 Schema Feasibility Guard

Before any API call, `_check_schema_feasibility()` detects questions that ask for data structurally absent from the schema and returns an immediate informative response:

| Pattern | Explanation returned |
|---|---|
| `home.*away` / `away.*game` | No home/away split in `player_stats` |
| `last N games` / `game log` | Season totals only, no game-by-game records |
| `quarter` / `overtime` / `clutch` | No period or situation splits |
| `last season` / `year-over-year` | Only 2024-25 data available |
| `playoff` / `post-season` | Regular season only |

This prevents the agent from looping on unanswerable SQL questions and burning rate-limit quota.

### 5.5 Rate-Limit Protection

Three complementary layers:

| Layer | Mechanism | Benefit |
|---|---|---|
| Model split | `sql_llm` = `mistral-large-latest` (separate bucket) | Doubles effective rate limit for double-tool turns |
| Proactive throttle | `min_call_interval = 3.0s` between SQL LLM calls | Prevents burst triggering |
| Exponential backoff | `_invoke_with_retry()` — waits 10s, 20s, 40s on 429 | Recovers from transient limit hits |
| Result cache | SHA-256 keyed in-memory dict | Eliminates duplicate API calls within a session |

---

## 6. Agent Design

**Location:** `agent_react.py`

### 6.1 ReAct Framework

The agent uses LangChain's `create_react_agent` with a strict ReAct prompt. `create_tool_calling_agent` was considered but causes `"Duplicate tool call id"` errors (Mistral API error 3230) due to a conflict between LangChain's tool-calling implementation and the Mistral SDK version in use. The ReAct string-parsing approach is stable with the current stack.

### 6.2 Routing Rules (from prompt)

```
Numerical / structured → nba_stats_sql
Qualitative / textual  → nba_knowledge_rag
Hybrid                 → call tool 1, observe, call tool 2, synthesise

Retry rules:
  "SQL execution error" → may retry once with simplified query
  "not available in the database" → do NOT retry, report limitation
```

### 6.3 Iteration Budget

```python
AgentExecutor(
    max_iterations        = 8,   # covers worst-case hybrid multi-tool turn
    early_stopping_method = "generate",  # forces Final Answer at limit
    handle_parsing_errors = True,        # recovers from format violations
)
```

`early_stopping_method="generate"` prevents the agent from returning the unhelpful `"Agent stopped due to iteration limit"` string — instead it synthesises a partial answer from whatever it collected.

---

## 7. Evaluation Framework

**Location:** `evaluation/`

### 7.1 Testset

Evaluation questions are divided into two distinct groups:

**RAG-targeted (26 questions)** :

| Category | N | Tests |
|---|---|---|
| `simple` | 8 | Single-chunk factual retrieval |
| `multihop` | 6 | Multi-document synthesis |
| `comparative` | 6 | Opinion comparison across sources |
| `noisy` | 6 | Slang, French, typos |

**SQL-targeted (21 questions)** — all have `expected_tool` field:

| Category | N | Tests |
|---|---|---|
| `sql_simple_ranking` | 4 | `ORDER BY … LIMIT` generation |
| `sql_advanced` | 3 | `ast_to`, `net_rtg`, `efg_pct`, `pie` |
| `sql_team_aggregation` | 2 | `GROUP BY team` |
| `sql_edge_case` | 4 | Schema boundary hallucination |
| `sql_noisy` | 2 | French + informal English routing |
| `hybrid_text_numeric` | 4 | Questions requiring both textual and structured numerical information  |
| `sql_nl_mapping` | 2 | Natural Language Mapping into DB columns  |

### 7.2 RAGAS Metrics (ragas==0.4.3)

| Metric | RAG rows | SQL rows | Description |
|---|---|---|---|
| `faithfulness` | ✓ | ✓ (vs SQL result) | Answer grounded in retrieved content |
| `answer_relevancy` | ✓ | ✓ | Answer on-topic for the question |
| `context_recall` | ✓ | **N/A** | Ground truth covered by chunks |
| `context_precision` | ✓ | **N/A** | Retrieved chunks actually useful |
| `sql_execution_success` | — | ✓ | Query ran without SQL error |
| `result_non_empty` | — | ✓ | Query returned at least one row |

`context_recall` and `context_precision` are `None` (not `0`) for SQL rows — the SQL tool operates on a structured query pathway with no text retrieval step, making these metrics undefined rather than poor.

**Judge model:** `gemini-3.0-flash` via `LangchainLLMWrapper`. Gemini is used here to keep the judge fully decoupled from the mistralai SDK version.

### 7.3 Routing Accuracy Tracking

`evaluate_agent.py` compares `expected_tool` (from testset JSON) with `tool_used` (from `intermediate_steps`) and classifies mismatches:

| Error type | Meaning |
|---|---|
| `used_rag_not_sql` | Stats question routed to RAG |
| `used_sql_not_rag` | Narrative question routed to SQL |
| `used_none` | No tool called when one was expected |
| `used_wrong_tool` | Other mismatch |

Routing accuracy is reported per-tool and per-category, and misrouted questions are listed explicitly for critical analysis.

### 7.4 Evaluation Steps

**Step 1 — (RAG-only testset):**
Your test file must be in JSON format, containing a list of dictionaries, with each dictionary consisting of three keys:
* category
* question
* ground_truth

Place your test file in the `evaluation/eval_artifacts` folder, then run the command
```bash
python -m evaluation.evaluate_ragas\
    --testset_name eval_dataset.json \
    --testset_filled eval_agent_rag_dataset_filled.json
```
The evaluation results will be saved in csv file in eval_artifact folder, to be used later in comparing RAGAS scores before/after adding the SQL tool on identical questions.

**Step 2 — Robustness (SQL + hybrid testset):**
The same logic as evaluate_ragas, evaluation dataset must be located in eval_artifact, each dictionary has four keys: category, question, ground_truth, expected_tool
```bash
python -m evaluation/evaluate_agent \
    --testset evaluation_agent_dataset.json \
    --output  evaluation_agent_results.csv
```
Evaluates routing accuracy, SQL tool health, and hybrid question handling.

---


## 8. Setup & Installation

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- Mistral API key → [console.mistral.ai](https://console.mistral.ai)
- Google API key (for RAGAS judge) → [ai.google.dev](https://ai.google.dev)

### Installation

```bash
# Clone the repository
git clone https://github.com/reemHasan/assistant_rag_mistral.git
cd assistant_rag_mistral

# Install dependencies with uv (reads pyproject.toml + uv.lock)
uv sync

# Or with pip
pip install -e .
```

### Environment Variables

Create `.env` in the project root:

```env
MISTRAL_API_KEY=your_mistral_key_here
GOOGLE_API_KEY=your_google_key_here      # only needed for evaluation
```
### Configuration Variables
You can customize the application by modifying the settings in `utils/config.py` like changing:
- Mistral models
- Chunk size and overlap
- Default number of retrieved documents

### First-Time Setup

**1. Build the FAISS index** (place PDFs in `inputs/inputs_rag_tool/` first):
```bash
python -m rag.indexer
```

**2. Ingest Excel data into SQLite:** 
```bash
python -m utils.load_excel_to_db \
    --excel '.\inputs\inputs_sql_tool\regular+NBA+(2).xlsx'
```

**3. Run Rag generator and chat with it in CLI
```bash
python -m rag.generator 
```
---

## 9. Usage

### Streamlit Application

```bash
streamlit run MistralChat.py
```

### Agent (CLI)
```bash
python -m agent_react
```

### Evaluation

```bash
# Step 1: RAG-only baseline
python -m evaluation.evaluate_agent \
    --testset evaluation_dataset_rag.json \
    --testset_filled eval_agent_rag_dataset_filled.json

# Step 2: Full testset with routing accuracy
python -m evaluation.evaluate_ragas \
    --testset evaluation_dataset.json \
    --output  evaluation_results_agent.csv
```

---

## 10. Known Limitations

### SQL Tool Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Season totals only — no game logs | Cannot answer "last 5 games" questions | Schema feasibility guard catches these before API call |
| No home/away split | Cannot compare home vs away performance | Feasibility guard returns informative explanation |
| Single season (2024-25) | No year-over-year comparison | Feasibility guard flags prior-season questions |
| NL→SQL mapping errors | Wrong column used for ambiguous terms (e.g. "clutch") | Few-shot examples cover the most common mappings; retry allowed once on SQL error |

### Agent Routing Limitations

| Limitation | Impact |
|---|---|
| `create_react_agent` requires strict format compliance | Format violations trigger `handle_parsing_errors` recovery, adding one iteration |
| `create_tool_calling_agent` incompatible with current Mistral SDK | Prevents use of native function-calling API (see error code 3230) |
| Rate limits on free Mistral tier | Back-to-back SQL calls may still hit 429 despite throttling on high-load sessions |

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `ragas` | 0.4.3 | RAGAS evaluation (collections API) |
| `langchain` | 0.3.27 | Agent framework |
| `langchain-mistralai` | 0.2.12 | Mistral LLM wrapper |
| `langchain-google-genai` | 2.1.10 | Gemini judge model |
| `langchain-google-vertexai` | 2.0.28 | Required by ragas.llms.base |
| `mistralai` | 1.9.10 | Mistral SDK (v1.x API) |
| `faiss-cpu` | 1.10.0 | Vector similarity search |
| `pydantic` | ≥2.13.4 | Data validation |
| `easyocr` | ≥1.7.2 | OCR for image-based PDFs |
| `pymupdf` | ≥1.22.0 | PDF rendering for OCR |
| `streamlit` | 1.44.1 | Web interface |
| `logfire` | ≥4.36.0 | Observability and tracing |

---

*Generated for the NBA Analytics Assistant project — OpenClassroom MLOps formation*