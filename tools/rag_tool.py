"""
rag_tool.py
===========
LangChain BaseTool wrapping the CosineRetriever (FAISS + Mistral embeddings)
for qualitative / contextual NBA questions.

This tool handles everything the SQL tool cannot:
  • Reddit fan opinions and sentiment
  • Historical narratives and player legacy debates
  • Tactical and coaching discussions
  • Any question whose answer lives in the corpus text
    rather than in a structured column
"""

from __future__ import annotations
import logging
from typing import Any, Type
from langchain_community.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import logfire
log = logging.getLogger(__name__)


class _RAGInput(BaseModel):
    question: str = Field(description="Qualitative question about NBA players, teams, tactics, or history.")


class NBAKnowledgeTool(BaseTool):
    """
    RAG tool: retrieves relevant chunks from the FAISS corpus and returns
    them as context for the LLM to synthesise.

    The tool returns the raw retrieved text — the agent's LLM then reads
    it and composes the final answer. This separation keeps retrieval and
    generation concerns cleanly decoupled.
    """

    name:        str = "nba_knowledge_rag"
    description: str = (
        "Use for qualitative NBA questions: player reputations, fan opinions, "
        "tactical analysis, coaching strategies, historical comparisons, "
        "narrative assessments ('Is X underrated?', 'How has Y's game evolved?', "
        "'What do fans think about Z?'). "
        "Also use when a question requires context from Reddit discussions "
        "or analyst commentary rather than hard statistics. "
        "Do NOT use for questions requiring exact numbers, rankings, or "
        "statistical comparisons — use nba_stats_sql for those."
    )
    args_schema: Type[BaseModel] = _RAGInput

    retriever: Any = Field(description="CosineRetriever instance")
    top_k:     int = Field(default=4, description="Number of chunks to retrieve")
    @logfire.instrument("build_answer_rag_tool")
    def _run(self, question: str) -> str:
        try:
            docs: list[Document] = self.retriever.retrieve(question, k=self.top_k)
        except Exception as exc:
            logfire.error("RAG retrieval failed: %s", exc)
            return f"Retrieval error: {exc}"

        if not docs:
            logfire.warning(f"No relevant context found in the corpus for this question: {question}")
            return "No relevant context found in the corpus for this question."
        else:
            parts = []
            for i, doc in enumerate(docs, 1):
                score  = doc.metadata.get("cosine_score", "?")
                source = doc.metadata.get("source", "unknown")
                topic  = doc.metadata.get("topic",  "general")
                parts.append(
                    f"[Chunk {i} | source: {source} | topic: {topic} | score: {score}]\n"
                    f"{doc.page_content}"
                )
            logfire.info("RAg tool found interesting chuncks in Faiss store", query_snippet = question[:60], returned= len(docs),)
            return "\n\n---\n\n".join(parts)

    async def _arun(self, question: str) -> str:
        return self._run(question)


def build_rag_tool(retriever: Any, top_k: int = 4) -> NBAKnowledgeTool:
    """
    Construct an NBAKnowledgeTool from a CosineRetriever.

    Parameters
    ----------
    retriever : CosineRetriever (from retriever.py)
    top_k     : Number of chunks to retrieve per query (default 4)

    Example
    -------
    >>> from rag.retriever import CosineRetriever
    >>> embeddings = MistralAIEmbeddings(model="mistral-embed", api_key="...")
    >>> retriever  = CosineRetriever(embeddings=embeddings,faiss_index_path)  # loads from disk
    >>> tool = build_rag_tool(retriever, top_k=4)
    """
    return NBAKnowledgeTool(retriever=retriever, top_k=top_k)
