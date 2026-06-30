# models.py
import logfire
from logfire.integrations.pydantic import PluginSettings
from utils.config import  MODEL_NAME, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K, EVALUATION_MODEL_NAME, EVALUATION_EMBEDDING, SQL_MODEL
import pandas as pd
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import os
load_dotenv()

#TESTSET_SIZE     = int(os.getenv("TESTSET_SIZE", "50"))
CHUNK_SIZE       = int(CHUNK_SIZE)
CHUNK_OVERLAP    = int(CHUNK_OVERLAP)
TOP_K            = int(SEARCH_K)

# Logfire — instrument everything
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
#logfire.instrument_pydantic()
# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pydantic validation models for rag tool
# ─────────────────────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"}),):
    """Metadata attached to every FAISS chunk."""
    source:    str
    page:      int       = Field(ge=0)
    thread_id: str       = ""
    topic:     str       = "general"
    chunk_idx: int       = Field(ge=0)

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source must not be empty")
        return v

class ValidatedChunk(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"})):
    """A cleaned, validated document chunk ready for embedding."""
    content:  str        = Field(min_length=50)
    metadata: ChunkMetadata

    @field_validator("content")
    @classmethod
    def no_ui_noise(cls, v: str) -> str:
        noise_patterns = [
            r"Accéder au contenu principal",
            r"Se connecter",
            r"Répondre|Partager",
            r"Comm\. du top \d+%",
            r"Sponsorisé\(e\)",
        ]
        for pat in noise_patterns:
            v = re.sub(pat, "", v)
        v = re.sub(r"\s+", " ", v).strip()
        if len(v) < 50:
            raise ValueError(f"Chunk too short after cleaning ({len(v)} chars)")
        return v


class TestRow(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"})):
    """One row in the evaluation dataset."""
    question:           str  = Field(min_length=10)
    ground_truth:       str  = Field(min_length=5)
    answer:             str  = ""
    contexts:           list[str] = Field(default_factory=list)
    category:           str  = "simple"
    #synthesizer_name:   str  = ""

    @field_validator("category")
    @classmethod
    def valid_category(cls, v: str) -> str:
        allowed = {"simple", "comparative", "multihop", "noisy", "out_of_scope"}
        if v not in allowed:
            raise ValueError(f"category must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def answer_needs_context(self) -> "TestRow":
        if self.answer and not self.contexts:
            raise ValueError("answer is set but contexts is empty — retrieval step missing")
        return self


class EvalResult(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"})):
    """RAGAS scores for one test row."""
    question:           str
    category:           str
    faithfulness:       float | None = Field(ge=0, le=1, default=None)
    answer_relevancy:   float | None = Field(ge=0, le=1, default=None)
    context_recall:     float | None = Field(ge=0, le=1, default=None)
    context_precision:  float | None = Field(ge=0, le=1, default=None)
    #answer_correctness: float | None = Field(ge=0, le=1, default=None)

    @property
    def mean_score(self) -> float:
        scores = [s for s in [
            self.faithfulness, self.answer_relevancy,
            self.context_recall, self.context_precision,
            #self.answer_correctness,
        ] if s is not None]
        return round(sum(scores) / len(scores), 4) if scores else 0.0


class PipelineConfig(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"})):
    """Validated runtime config."""
    chunk_size:      int  = Field(ge=100, le=1000, default=CHUNK_SIZE)
    chunk_overlap:   int  = Field(ge=0,   le=200,  default=CHUNK_OVERLAP)
    top_k:           int  = Field(ge=1,   le=10,   default=TOP_K)
    batch_size:      int  = Field(ge=5,   le=50,   default=EMBEDDING_BATCH_SIZE)
    #testset_size:    int  = Field(ge=5,   le=100,  default=TESTSET_SIZE)
    mistral_model:   str  = MODEL_NAME
    sql_generator_model: str = SQL_MODEL
    embed_model:     str  = EMBEDDING_MODEL
    judge_model:     str  = EVALUATION_MODEL_NAME
    judge_embedding_model: str  = EVALUATION_EMBEDDING
    
    @model_validator(mode="after")
    def overlap_less_than_chunk(self) -> "PipelineConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self
    
# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pydantic validation models for sql tool
# ─────────────────────────────────────────────────────────────────────────────

class TeamRow(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"})):
    code:      str = Field(min_length=2, max_length=3)
    full_name: str = Field(min_length=3)

    @field_validator("code")
    @classmethod
    def upper_code(cls, v: str) -> str:
        return v.strip().upper()


class PlayerStatsRow(BaseModel, plugin_settings=PluginSettings(logfire={"record": "all"})):
    """
    One row from the 'Données NBA' sheet.
    All per-game / season-total fields are optional so that
    players with incomplete data still pass validation.
    """
    player:     str = Field(min_length=1)
    team:       str = Field(min_length=2, max_length=3)
    age:        int = Field(ge=18, le=55)
    gp:         int = Field(ge=0, le=82,  description="Games played")
    w:          int = Field(ge=0, le=82,  description="Wins")
    l:          int = Field(ge=0, le=82,  description="Losses")
    min_pg:     float = Field(ge=0, le=48, description="Minutes per game")

    # Scoring
    pts:        float = Field(ge=0)
    fgm:        float = Field(ge=0)
    fga:        float = Field(ge=0)
    fg_pct:     float = Field(ge=0, le=100)
    min_after_15:   float = Field(ge=0, alias="min_after_15")
    three_pa:   float = Field(ge=0, alias="3pa")
    three_pct:  float = Field(ge=0, le=100, alias="3p_pct")
    ftm:        float = Field(ge=0)
    fta:        float = Field(ge=0)
    ft_pct:     float = Field(ge=0, le=100)

    # Rebounds
    oreb:       float = Field(ge=0)
    dreb:       float = Field(ge=0)
    reb:        float = Field(ge=0)

    # Playmaking / defense
    ast:        float = Field(ge=0)
    tov:        float = Field(ge=0)
    stl:        float = Field(ge=0)
    blk:        float = Field(ge=0)
    pf:         float = Field(ge=0)

    # Advanced
    fp:         Optional[float] = None   # Fantasy Points
    dd2:        Optional[int]   = None   # Double-doubles
    td3:        Optional[int]   = None   # Triple-doubles
    plus_minus: Optional[float] = None
    off_rtg:    Optional[float] = None
    def_rtg:    Optional[float] = None
    net_rtg:    Optional[float] = None
    ast_pct:    Optional[float] = None
    ast_to:     Optional[float] = None
    ast_ratio:  Optional[float] = None
    oreb_pct:   Optional[float] = None
    dreb_pct:   Optional[float] = None
    reb_pct:    Optional[float] = None
    to_ratio:   Optional[float] = None
    efg_pct:    Optional[float] = None
    ts_pct:     Optional[float] = None
    usg_pct:    Optional[float] = None
    pace:       Optional[float] = None
    pie:        Optional[float] = None
    poss:       Optional[int]   = None

    model_config = {"populate_by_name": True}

    @field_validator("team")
    @classmethod
    def upper_team(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("player")
    @classmethod
    def strip_player(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def wins_losses_consistent(self) -> "PlayerStatsRow":
        if self.w + self.l > self.gp:
            raise ValueError(
                f"W({self.w}) + L({self.l}) > GP({self.gp}) for {self.player}"
            )
        return self
