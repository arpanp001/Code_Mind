# backend/tests/test_chunker.py
# Run with: pytest tests/test_chunker.py -v

import pytest
from app.models.ingest_models import ParsedFile
from app.core.processing.chunker import (
    SlidingWindowChunker,
    FunctionAwareChunker,
    MarkdownChunker,
    ChunkingEngine,
    CHUNK_SIZE, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE,
)


def make_file(content: str, language: str = "python",
              path: str = "test.py") -> ParsedFile:
    """Helper to create a ParsedFile for testing."""
    return ParsedFile(
        file_path     = path,
        relative_path = path,
        content       = content,
        language      = language,
        project_id    = "test_project",
    )


# ── Sliding Window Tests ──────────────────────────────────────────────────────

class TestSlidingWindowChunker:

    def setup_method(self):
        self.chunker = SlidingWindowChunker()

    def test_small_file_single_chunk(self):
        file    = make_file("const x = 1;\nconst y = 2;\n")
        chunks  = self.chunker.chunk(file, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "block"

    def test_large_file_multiple_chunks(self):
        # Generate content larger than chunk_size
        content = "const x = 1;\n" * 200      # ~2600 chars
        file    = make_file(content, language="javascript")
        chunks  = self.chunker.chunk(file, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunks_have_correct_metadata(self):
        content = "line\n" * 100
        file    = make_file(content, path="src/utils.js", language="javascript")
        chunks  = self.chunker.chunk(file, chunk_size=200)
        for c in chunks:
            assert c.file_path  == "src/utils.js"
            assert c.language   == "javascript"
            assert c.project_id == "test_project"
            assert c.start_line >= 1
            assert c.end_line   >= c.start_line

    def test_overlap_is_applied(self):
        content = "A" * 300 + "\n" + "B" * 300
        file    = make_file(content, language="sql")
        chunks  = self.chunker.chunk(file, chunk_size=300, chunk_overlap=50)
        if len(chunks) > 1:
            # Second chunk should contain content from the first
            assert len(chunks[1].overlap_prefix) > 0

    def test_empty_content_returns_no_chunks(self):
        file   = make_file("   ", language="sql")
        chunks = self.chunker.chunk(file)
        assert len(chunks) == 0

    def test_chunk_ids_are_unique(self):
        content = "x = 1\n" * 300
        file    = make_file(content)
        chunks  = self.chunker.chunk(file, chunk_size=200)
        ids     = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))    # No duplicates


# ── Markdown Chunker Tests ────────────────────────────────────────────────────

class TestMarkdownChunker:

    def setup_method(self):
        self.chunker = MarkdownChunker()

    def test_splits_at_headings(self):
        content = (
            "# Installation\n\nRun npm install.\n\n"
            "# Authentication\n\nUse JWT tokens.\n\n"
            "# Database\n\nConnect via SQLAlchemy."
        )
        file   = make_file(content, language="markdown")
        chunks = self.chunker.chunk(file)
        assert len(chunks) == 3

    def test_chunk_type_is_heading(self):
        content = "# Overview\n\nThis is the overview section.\n"
        file    = make_file(content, language="markdown")
        chunks  = self.chunker.chunk(file)
        assert chunks[0].chunk_type == "heading"

    def test_heading_title_stored_in_function_name(self):
        content = "# Getting Started\n\nFollow these steps.\n"
        file    = make_file(content, language="markdown")
        chunks  = self.chunker.chunk(file)
        assert "Getting Started" in chunks[0].function_name

    def test_no_headings_falls_back_to_sliding(self):
        content = "This is plain text.\n" * 50
        file    = make_file(content, language="markdown")
        # Should still return chunks (sliding window fallback)
        chunks  = self.chunker.chunk(file, chunk_size=200)
        assert len(chunks) >= 1


# ── Function-Aware Chunker Tests ──────────────────────────────────────────────

class TestFunctionAwareChunker:

    def setup_method(self):
        self.chunker = FunctionAwareChunker()

    def test_splits_python_functions(self):
        content = '''
def authenticate(username, password):
    """Authenticates a user."""
    user = find_user(username)
    if not user:
        raise ValueError("User not found")
    return create_token(user)


def create_token(user):
    """Creates a JWT token."""
    payload = {"user_id": user.id}
    return jwt.encode(payload, SECRET_KEY)


def verify_token(token):
    """Verifies a JWT token."""
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
'''
        file   = make_file(content, language="python")
        chunks = self.chunker.chunk(file)
        # Should have at least 2 function chunks
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) >= 2

    def test_detects_function_names(self):
        content = '''
def login(username: str, password: str) -> str:
    token = jwt.encode({"user": username}, SECRET)
    return token


def logout(token: str) -> None:
    blacklist.add(token)
'''
        file   = make_file(content, language="python")
        chunks = self.chunker.chunk(file)
        names  = [c.function_name for c in chunks if c.function_name]
        assert "login"  in names
        assert "logout" in names

    def test_detects_class_type(self):
        content = '''
class AuthService:
    """Handles authentication."""

    def __init__(self):
        self.secret = SECRET_KEY

    def login(self, username, password):
        return self._create_token(username)
'''
        file   = make_file(content, language="python")
        chunks = self.chunker.chunk(file)
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1

    def test_handles_javascript(self):
        content = '''
function authenticateUser(req, res) {
    const { username, password } = req.body;
    const token = createJWT(username);
    res.json({ token });
}

function createJWT(username) {
    return jwt.sign({ username }, process.env.SECRET);
}

class UserService {
    constructor(db) {
        this.db = db;
    }
    async getUser(id) {
        return this.db.find(id);
    }
}
'''
        file   = make_file(content, language="javascript", path="auth.js")
        chunks = self.chunker.chunk(file)
        assert len(chunks) >= 2

    def test_falls_back_for_unknown_language(self):
        content = "SELECT * FROM users;\n" * 50
        file    = make_file(content, language="sql")
        chunks  = self.chunker.chunk(file, chunk_size=200)
        # Should still produce chunks via sliding window fallback
        assert len(chunks) >= 1

    def test_chunk_line_numbers_are_sequential(self):
        content = '''
def func_one():
    x = 1
    return x


def func_two():
    y = 2
    return y
'''
        file   = make_file(content, language="python")
        chunks = self.chunker.chunk(file)
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line   >= c.start_line


# ── ChunkingEngine Orchestrator Tests ─────────────────────────────────────────

class TestChunkingEngine:

    def setup_method(self):
        self.engine = ChunkingEngine(chunk_size=500, chunk_overlap=50)

    def test_routes_python_to_function_chunker(self):
        content = '''
def foo():
    return 1

def bar():
    return 2
'''
        file    = make_file(content, language="python")
        chunks  = self.engine.chunk_file(file)
        types   = {c.chunk_type for c in chunks}
        assert "function" in types

    def test_routes_markdown_to_heading_chunker(self):
        content = "# Intro\n\nHello.\n\n# Setup\n\nInstall it.\n"
        file    = make_file(content, language="markdown", path="README.md")
        chunks  = self.engine.chunk_file(file)
        types   = {c.chunk_type for c in chunks}
        assert "heading" in types

    def test_chunk_all_returns_summary(self):
        files = [
            make_file("def foo():\n    return 1\n", "python",     "a.py"),
            make_file("# Title\n\nContent here.\n", "markdown",   "README.md"),
            make_file("SELECT 1;\n" * 30,           "sql",        "q.sql"),
        ]
        result = self.engine.chunk_all(files, "test_proj")
        assert result.total_files  >= 1
        assert result.total_chunks >= 1
        assert isinstance(result.chunks_by_language, dict)

    def test_all_chunks_pass_size_constraints(self):
        content = "def func():\n" + "    x = 1\n" * 100
        file    = make_file(content, language="python")
        chunks  = self.engine.chunk_file(file)
        for c in chunks:
            assert len(c.text) >= MIN_CHUNK_SIZE
            assert len(c.text) <= MAX_CHUNK_SIZE

    def test_chunk_ids_unique_across_files(self):
        files = [
            make_file("def a():\n    pass\n", "python", "file1.py"),
            make_file("def b():\n    pass\n", "python", "file2.py"),
        ]
        result = self.engine.chunk_all(files, "proj")
        ids    = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))