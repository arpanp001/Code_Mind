# backend/app/models/query_models.py
#
# Models for the chat and search endpoints.

from pydantic import BaseModel, field_validator
from typing import Optional


# ── Request Models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    project_id:      str
    question:        str
    include_sources: bool          = True
    max_sources:     int           = 5
    session_id:      Optional[str] = None   # ← ADD THIS

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) < 3:
            raise ValueError("Question too short")
        return v

    @field_validator("max_sources")
    @classmethod
    def valid_source_count(cls, v: int) -> int:
        return max(1, min(10, v))


class SearchRequest(BaseModel):
    """
    Request body for POST /query/search
    Semantic search — returns relevant code chunks without generating an answer.
    Useful for "show me all files related to authentication".
    """
    project_id: str
    query: str
    top_k: int = 5


# ── Response Models ─────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """
    A single relevant code snippet returned alongside an answer.
    This is what gets displayed as a "source" card in the frontend.
    """
    file_path: str                    # e.g. "src/auth/login.py"
    language: str                     # e.g. "python"
    code: str                         # The actual code text
    start_line: Optional[int] = None  # Line number where chunk starts
    end_line: Optional[int] = None    # Line number where chunk ends
    relevance_score: float            # 0.0 to 1.0, how relevant this chunk is
    chunk_type: Optional[str] = None  # "function", "class", "block"


class ChatResponse(BaseModel):
    answer:        str
    sources:       list[SourceChunk] = []
    project_id:    str
    question:      str
    tokens_used:   Optional[int] = None
    memories_used: int           = 0    # Gemini token count (for monitoring)


class SearchResponse(BaseModel):
    """Response for POST /query/search"""
    results: list[SourceChunk]
    query: str
    total_found: int

class ExplainRequest(BaseModel):
    """Request body for POST /query/explain"""
    project_id:  str
    code:        str
    language:    str        = "python"
    file_path:   str        = ""
    question:    str        = ""         # Optional specific question about the code

    @field_validator("code")
    @classmethod
    def code_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Code cannot be empty")
        return v

class ExplainResponse(BaseModel):
    """Response for POST /query/explain"""
    explanation:  str
    language:     str
    file_path:    str
    tokens_used:  int   = 0
    success:      bool  = True