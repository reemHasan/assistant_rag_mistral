
import logfire
from utils.config import  MODEL_NAME, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K, EVALUATION_MODEL_NAME, EVALUATION_EMBEDDING
import pandas as pd
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
import os
load_dotenv()

#TESTSET_SIZE     = int(os.getenv("TESTSET_SIZE", "50"))
CHUNK_SIZE       = int(CHUNK_SIZE)
CHUNK_OVERLAP    = int(CHUNK_OVERLAP)
TOP_K            = int(SEARCH_K)

# Logfire — instrument everything
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_pydantic()
# ───────────────────────────────────────────────────────────────────────────
# Pydantic models — validate every data boundary
# ───────────────────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
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

class ValidatedChunk(BaseModel):
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


class TestRow(BaseModel):
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


class EvalResult(BaseModel):
    """RAGAS scores for one test row."""
    question:           str
    category:           str
    faithfulness:       float | None = Field(ge=0, le=1, default=None)
    answer_relevancy:   float | None = Field(ge=0, le=1, default=None)
    context_recall:     float | None = Field(ge=0, le=1, default=None)
    context_precision:  float | None = Field(ge=0, le=1, default=None)
    answer_correctness: float | None = Field(ge=0, le=1, default=None)

    @property
    def mean_score(self) -> float:
        scores = [s for s in [
            self.faithfulness, self.answer_relevancy,
            self.context_recall, self.context_precision,
            self.answer_correctness,
        ] if s is not None]
        return round(sum(scores) / len(scores), 4) if scores else 0.0


class PipelineConfig(BaseModel):
    """Validated runtime config."""
    chunk_size:      int  = Field(ge=100, le=1000, default=CHUNK_SIZE)
    chunk_overlap:   int  = Field(ge=0,   le=200,  default=CHUNK_OVERLAP)
    top_k:           int  = Field(ge=1,   le=10,   default=TOP_K)
    batch_size:      int  = Field(ge=5,   le=50,   default=EMBEDDING_BATCH_SIZE)
    #testset_size:    int  = Field(ge=5,   le=100,  default=TESTSET_SIZE)
    mistral_model:   str  = MODEL_NAME
    embed_model:     str  = EMBEDDING_MODEL
    judge_model:     str  = EVALUATION_MODEL_NAME
    judge_embedding_model: str  = EVALUATION_EMBEDDING
    
    @model_validator(mode="after")
    def overlap_less_than_chunk(self) -> "PipelineConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self