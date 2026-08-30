# backend/app/models/memory_models.py
#
# Models for the project memory feature.
# Memory lets teams store architecture decisions, bug fixes,
# and implementation notes that get included in RAG context.

from pydantic import BaseModel, field_validator
from typing import Optional
from enum import Enum


class MemoryType(str, Enum):
    """
    Three types of memory a team can store.
    - ARCHITECTURE_DECISION: "We chose PostgreSQL because..."
    - BUG_FIX: "Fixed null pointer in auth.py by checking user exists first"
    - NOTE: General implementation notes
    """
    ARCHITECTURE_DECISION = "architecture_decision"
    BUG_FIX = "bug_fix"
    NOTE = "note"


# ── Request Models ──────────────────────────────────────────────────────────

class MemoryAddRequest(BaseModel):
    """Request body for POST /memory/add"""
    project_id: str
    content: str
    memory_type: MemoryType = MemoryType.NOTE
    tags: list[str] = []              # e.g. ["authentication", "jwt"]
    title: Optional[str] = None       # Optional short title

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Memory content must be at least 10 characters")
        return v

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        """Lowercase and strip whitespace from all tags."""
        return [tag.lower().strip() for tag in v if tag.strip()]


class MemorySearchRequest(BaseModel):
    """Request body for POST /memory/search"""
    project_id: str
    query: str
    memory_type: Optional[MemoryType] = None  # Filter by type (optional)
    top_k: int = 5


# ── Response Models ─────────────────────────────────────────────────────────

class MemoryItem(BaseModel):
    """A single memory item."""
    memory_id: str
    project_id: str
    content: str
    memory_type: MemoryType
    tags: list[str]
    title: Optional[str]
    created_at: str
    relevance_score: Optional[float] = None   # Set when returned from search


class MemoryListResponse(BaseModel):
    """Response for GET /memory/{project_id}"""
    memories: list[MemoryItem]
    total: int
    project_id: str


class MemoryAddResponse(BaseModel):
    """Response for POST /memory/add"""
    memory_id: str
    status: str
    message: str


class MemoryDeleteResponse(BaseModel):
    """Response for DELETE /memory/{memory_id}"""
    message: str
    memory_id: str