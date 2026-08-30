# backend/app/models/ingest_models.py
#
# These models define the shape of data for ingestion endpoints.
# Pydantic validates incoming JSON automatically — if a required field
# is missing or the wrong type, FastAPI returns a 422 error immediately.

from pydantic import BaseModel, HttpUrl, field_validator
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SourceType(str, Enum):
    """

    Enum restricts source_type to only these two values.
    Using Enum instead of plain string means typos like "gihub" are
    caught at validation time, not buried in a runtime error later.
    """
    GITHUB = "github"
    ZIP = "zip"


class ProjectStatus(str, Enum):
    """Tracks where a project is in the processing pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


# ── Request Models (what the client sends TO the API) ──────────────────────

class GithubIngestRequest(BaseModel):
    """
    Request body for POST /ingest/github
    HttpUrl automatically validates that the URL is a valid URL format.
    """
    url: str                          # GitHub repo URL
    branch: Optional[str] = "main"   # Which branch to clone (default: main)

    @field_validator("url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        """
        Custom validator — runs after Pydantic's type check.
        Ensures the URL is actually a GitHub URL, not just any URL.
        """
        if "github.com" not in v:
            raise ValueError("URL must be a GitHub repository URL")
        return v.rstrip("/")         # Remove trailing slash if present


# ── Response Models (what the API sends BACK to the client) ────────────────

class IngestResponse(BaseModel):
    """
    Returned immediately after starting an ingestion job.
    We return right away (don't make the user wait) and they
    can poll /ingest/status/{project_id} to check progress.
    """
    project_id: str
    status: ProjectStatus
    message: str
    source_type: SourceType


class ProjectStatusResponse(BaseModel):
    """
    Returned by GET /ingest/status/{project_id}
    Shows how many files and chunks were processed.
    """
    project_id: str
    name: str
    status: ProjectStatus
    source_type: SourceType
    file_count: int = 0
    chunk_count: int = 0
    clone_progress: int = -1 
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class ProjectListItem(BaseModel):
    """A single item in the GET /projects list."""
    project_id: str
    name: str
    status: ProjectStatus
    source_type: SourceType
    file_count: int = 0
    chunk_count: int = 0
    created_at: str


class ProjectListResponse(BaseModel):
    """The full response for GET /projects."""
    projects: list[ProjectListItem]
    total: int


class DeleteResponse(BaseModel):
    """Returned after successfully deleting a project."""
    message: str
    project_id: str

@dataclass
class ParsedFile:
    """
    Represents a single source code file that has been read and is
    ready for chunking.

    We use @dataclass instead of Pydantic here because this object
    lives entirely inside the backend pipeline — it never gets
    serialized to JSON. Dataclasses are simpler and faster for
    internal data transfer between modules.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    file_path: str
    # Relative path inside the project: "src/auth/login.py"
    # (NOT the absolute path on disk — we never expose disk paths to users)

    relative_path: str
    # Same as file_path but explicitly relative to the project root

    # ── Content ───────────────────────────────────────────────────────────
    content: str
    # Full text content of the file

    language: str
    # Detected programming language: "python", "javascript", "java", etc.

    # ── Metadata ──────────────────────────────────────────────────────────
    file_size_bytes: int = 0
    line_count: int = 0
    encoding: str = "utf-8"

    # ── Context (populated during chunking) ──────────────────────────────
    project_id: Optional[str] = None


@dataclass
class IngestionResult:
    """
    Summary of what the ingestion pipeline produced.
    Returned after processing all files in a ZIP or GitHub repo.
    """
    project_id: str
    total_files_found: int = 0       # All files before filtering
    files_processed: int = 0         # Files that passed all filters
    files_skipped: int = 0           # Files filtered out
    total_chunks: int = 0            # Chunks stored in ChromaDB
    errors: list[str] = field(default_factory=list)
    parsed_files: list[ParsedFile] = field(default_factory=list)

# Add these classes to the BOTTOM of backend/app/models/ingest_models.py

@dataclass
class CodeChunk:
    """
    A single chunk of source code ready to be embedded and stored in ChromaDB.

    This is the fundamental unit of the RAG pipeline.
    Everything downstream (embedding, storage, retrieval) works with CodeChunks.

    Design decisions:
    - We store BOTH chunk_id (for ChromaDB) and all metadata separately
      so we can filter by metadata without loading the full text
    - start_line/end_line let the frontend show "src/auth/login.py L42-58"
    - chunk_type helps the LLM understand what it's looking at
    - overlap_with_previous preserves context across chunk boundaries
    """

    # ── Identity ──────────────────────────────────────────────────────────
    chunk_id:    str        # Unique ID: "proj_abc_src_auth_login_py_3"
    project_id:  str        # Which project this belongs to
    chunk_index: int        # Position in the file: 0, 1, 2, ...

    # ── Content ───────────────────────────────────────────────────────────
    text:        str        # The actual code text to embed
    file_path:   str        # Relative path: "src/auth/login.py"
    language:    str        # "python", "javascript", "java", etc.

    # ── Position ──────────────────────────────────────────────────────────
    start_line:  int = 0    # Line number where chunk starts (1-indexed)
    end_line:    int = 0    # Line number where chunk ends (1-indexed)

    # ── Semantic metadata ─────────────────────────────────────────────────
    chunk_type:     str = "block"   # "function", "class", "block", "heading"
    function_name:  str = ""        # Name if chunk_type == "function"
    class_name:     str = ""        # Name if chunk_type == "class"

    # ── Size info ─────────────────────────────────────────────────────────
    char_count:   int = 0   # Character count
    token_count:  int = 0   # Approximate token count (for LLM context window)

    # ── Context ───────────────────────────────────────────────────────────
    # First few lines of the previous chunk — injected at the start
    # of this chunk to give the embedding model continuity.
    # Without overlap, the model sees: "    return token" with no context.
    # With overlap, it sees: "def authenticate():  ...  return token"
    overlap_prefix: str = ""


@dataclass
class ChunkingResult:
    """
    Summary returned after chunking all files in a project.
    Used to update the database and log statistics.
    """
    project_id:     str
    total_chunks:   int                       = 0
    total_files:    int                       = 0
    chunks:         list[CodeChunk]           = field(default_factory=list)
    skipped_files:  int                       = 0
    errors:         list[str]                 = field(default_factory=list)

    # Stats per language (for logging / debugging)
    chunks_by_language: dict[str, int]        = field(default_factory=dict)