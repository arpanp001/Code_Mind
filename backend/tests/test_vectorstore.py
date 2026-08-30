# backend/tests/test_vectorstore.py
# Run with: pytest tests/test_vectorstore.py -v

import pytest
import tempfile
import numpy as np

from app.core.rag.vectorstore import (
    ChromaDBClient,
    RetrievedChunk,
    SearchResult,
    get_collection_name,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
)
from app.core.rag.embedder import EmbeddedChunk, EMBEDDING_DIM
from app.models.ingest_models import CodeChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_embedded_chunk(
    chunk_id:      str   = "test_chunk_001",
    text:          str   = "def hello():\n    return 'world'",
    file_path:     str   = "src/test.py",
    language:      str   = "python",
    chunk_type:    str   = "function",
    function_name: str   = "hello",
    start_line:    int   = 1,
    end_line:      int   = 2,
    embedding:     list  = None,
    project_id:    str   = "test_project",
) -> EmbeddedChunk:
    """Creates a minimal EmbeddedChunk for testing."""
    if embedding is None:
        # Generate a random normalized vector
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        embedding = vec.tolist()

    chunk = CodeChunk(
        chunk_id      = chunk_id,
        project_id    = project_id,
        chunk_index   = 0,
        text          = text,
        file_path     = file_path,
        language      = language,
        start_line    = start_line,
        end_line      = end_line,
        chunk_type    = chunk_type,
        function_name = function_name,
        char_count    = len(text),
        token_count   = len(text) // 4,
    )
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


def make_test_client(tmp_dir: str) -> ChromaDBClient:
    """Creates a ChromaDBClient using a temp directory for isolation."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = ChromaDBClient()
    client._client = chromadb.PersistentClient(
        path=tmp_dir,
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
    )
    return client


# ── Collection Name Tests ─────────────────────────────────────────────────────

class TestCollectionName:

    def test_basic_project_id(self):
        name = get_collection_name("proj_abc123")
        assert "proj_abc123" in name
        assert name.startswith("cm_")

    def test_name_is_alphanumeric_plus_underscores(self):
        import re
        name = get_collection_name("proj-abc-123")
        assert re.match(r'^[a-zA-Z0-9_-]+$', name), (
            f"Invalid chars in: {name}"
        )

    def test_name_length_within_bounds(self):
        long_id = "a" * 100
        name    = get_collection_name(long_id)
        # ChromaDB limit is 63 chars
        assert len(name) <= 63, f"Name too long: {len(name)}"

    def test_different_projects_get_different_names(self):
        n1 = get_collection_name("project_one")
        n2 = get_collection_name("project_two")
        assert n1 != n2

    def test_same_project_always_same_name(self):
        n1 = get_collection_name("stable_project")
        n2 = get_collection_name("stable_project")
        assert n1 == n2


# ── Store & Retrieve Tests ────────────────────────────────────────────────────

class TestStoreAndRetrieve:

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.client  = make_test_client(self.tmp_dir)

    def test_store_single_chunk(self):
        chunk = make_embedded_chunk(project_id="proj_store")
        count = self.client.store_chunks("proj_store", [chunk])
        assert count == 1

    def test_store_multiple_chunks(self):
        chunks = [
            make_embedded_chunk(
                chunk_id   = f"chunk_{i:03d}",
                text       = f"def function_{i}():\n    return {i}",
                project_id = "proj_multi"
            )
            for i in range(5)
        ]
        count = self.client.store_chunks("proj_multi", chunks)
        assert count == 5

    def test_empty_list_returns_zero(self):
        count = self.client.store_chunks("proj_empty", [])
        assert count == 0

    def test_stored_count_matches_collection_count(self):
        chunks = [
            make_embedded_chunk(
                chunk_id   = f"c_{i}",
                project_id = "proj_count"
            )
            for i in range(3)
        ]
        self.client.store_chunks("proj_count", chunks)
        stats = self.client.get_collection_stats("proj_count")
        assert stats["total_chunks"] == 3

    def test_upsert_does_not_duplicate(self):
        """
        Storing the same chunk_id twice should update, not duplicate.
        """
        chunk = make_embedded_chunk(
            chunk_id   = "dup_chunk",
            text       = "def original():\n    pass",
            project_id = "proj_upsert"
        )
        self.client.store_chunks("proj_upsert", [chunk])
        self.client.store_chunks("proj_upsert", [chunk])  # store again

        stats = self.client.get_collection_stats("proj_upsert")
        assert stats["total_chunks"] == 1    # Still 1, not 2

    def test_projects_are_isolated(self):
        """
        Chunks stored for project A should not appear in project B's search.
        """
        vec = np.ones(EMBEDDING_DIM, dtype=np.float32)
        vec = (vec / np.linalg.norm(vec)).tolist()

        chunk_a = make_embedded_chunk(
            chunk_id   = "a_chunk",
            text       = "project A content",
            project_id = "project_a",
            embedding  = vec,
        )
        chunk_b = make_embedded_chunk(
            chunk_id   = "b_chunk",
            text       = "project B content",
            project_id = "project_b",
            embedding  = vec,
        )

        self.client.store_chunks("project_a", [chunk_a])
        self.client.store_chunks("project_b", [chunk_b])

        # Searching project_a should NOT return project_b's chunk
        result_a = self.client.search("project_a", vec, top_k=5)
        result_b = self.client.search("project_b", vec, top_k=5)

        a_ids = {c.chunk_id for c in result_a.chunks}
        b_ids = {c.chunk_id for c in result_b.chunks}

        assert "a_chunk" in a_ids
        assert "b_chunk" not in a_ids
        assert "b_chunk" in b_ids
        assert "a_chunk" not in b_ids


# ── Semantic Search Tests ─────────────────────────────────────────────────────

class TestSemanticSearch:

    def setup_method(self):
        self.tmp_dir   = tempfile.mkdtemp()
        self.client    = make_test_client(self.tmp_dir)
        self.project   = "search_project"

        # Create a known unit vector — first dim=1, rest=0
        auth_vec          = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        auth_vec[0]       = 1.0

        # Orthogonal vector — second dim=1, rest=0
        db_vec            = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        db_vec[1]         = 1.0

        # Store two chunks with known, distinct vectors
        chunks = [
            make_embedded_chunk(
                chunk_id      = "auth_chunk",
                text          = "def authenticate(user, pwd): return jwt.encode(...)",
                file_path     = "src/auth/login.py",
                chunk_type    = "function",
                function_name = "authenticate",
                language      = "python",
                project_id    = self.project,
                embedding     = auth_vec.tolist(),
            ),
            make_embedded_chunk(
                chunk_id      = "db_chunk",
                text          = "engine = create_engine(DATABASE_URL)",
                file_path     = "src/db/connection.py",
                chunk_type    = "block",
                language      = "python",
                project_id    = self.project,
                embedding     = db_vec.tolist(),
            ),
        ]
        self.client.store_chunks(self.project, chunks)
        self.auth_vec = auth_vec
        self.db_vec   = db_vec

    def test_similar_vector_returns_matching_chunk(self):
        """Query with auth vector should return auth chunk first."""
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=2)
        assert result.chunks[0].chunk_id == "auth_chunk"

    def test_search_returns_correct_file_path(self):
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=1)
        assert result.chunks[0].file_path == "src/auth/login.py"

    def test_search_returns_correct_language(self):
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=1)
        assert result.chunks[0].language == "python"

    def test_search_returns_similarity_score(self):
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=1)
        assert 0.0 <= result.chunks[0].similarity <= 1.0

    def test_identical_vector_has_high_similarity(self):
        """Searching with exact same vector should return similarity ≈ 1.0."""
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=1)
        assert result.chunks[0].similarity > 0.95

    def test_top_k_limits_results(self):
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=1)
        assert len(result.chunks) <= 1

    def test_search_empty_collection(self):
        """Searching an empty project should return empty results gracefully."""
        result = self.client.search("empty_project", self.auth_vec.tolist(), top_k=5)
        assert result.chunks == []
        assert result.total_found == 0

    def test_search_returns_text(self):
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=1)
        assert "authenticate" in result.chunks[0].text

    def test_results_sorted_by_similarity(self):
        result = self.client.search(self.project, self.auth_vec.tolist(), top_k=2)
        if len(result.chunks) > 1:
            assert (
                result.chunks[0].similarity >= result.chunks[1].similarity
            )


# ── Delete Tests ──────────────────────────────────────────────────────────────

class TestDeleteProject:

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.client  = make_test_client(self.tmp_dir)

    def test_delete_removes_collection(self):
        chunk = make_embedded_chunk(project_id="proj_del")
        self.client.store_chunks("proj_del", [chunk])
        self.client.delete_project("proj_del")

        # After deletion, searching should return empty results
        vec    = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        result = self.client.search("proj_del", vec.tolist(), top_k=5)
        assert result.chunks == []

    def test_delete_nonexistent_project_returns_false(self):
        result = self.client.delete_project("does_not_exist_xyz")
        assert result is False

    def test_delete_existing_project_returns_true(self):
        chunk = make_embedded_chunk(project_id="proj_del2")
        self.client.store_chunks("proj_del2", [chunk])
        result = self.client.delete_project("proj_del2")
        assert result is True


# ── Collection Stats Tests ────────────────────────────────────────────────────

class TestCollectionStats:

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.client  = make_test_client(self.tmp_dir)

    def test_stats_returns_correct_count(self):
        chunks = [
            make_embedded_chunk(chunk_id=f"s_{i}", project_id="proj_stats")
            for i in range(4)
        ]
        self.client.store_chunks("proj_stats", chunks)
        stats = self.client.get_collection_stats("proj_stats")
        assert stats["total_chunks"] == 4

    def test_stats_includes_collection_name(self):
        stats = self.client.get_collection_stats("proj_name")
        assert "collection_name" in stats

    def test_empty_project_stats(self):
        stats = self.client.get_collection_stats("proj_empty_stats")
        assert stats["total_chunks"] == 0