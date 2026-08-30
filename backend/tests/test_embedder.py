# backend/tests/test_embedder.py
# Run with: pytest tests/test_embedder.py -v
#
# Note: Tests that call the real model are marked @pytest.mark.slow
# Run all:   pytest tests/test_embedder.py -v
# Skip slow: pytest tests/test_embedder.py -v -m "not slow"

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import field

from app.models.ingest_models import CodeChunk
from app.core.rag.embedder import (
    EmbeddingEngine,
    EmbeddingCache,
    EmbeddedChunk,
    EmbeddingResult,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(
    text:          str  = "def hello():\n    return 'world'",
    file_path:     str  = "src/test.py",
    language:      str  = "python",
    chunk_id:      str  = "test_chunk_001",
    function_name: str  = "hello",
) -> CodeChunk:
    """Creates a minimal CodeChunk for testing."""
    return CodeChunk(
        chunk_id      = chunk_id,
        project_id    = "test_project",
        chunk_index   = 0,
        text          = text,
        file_path     = file_path,
        language      = language,
        start_line    = 1,
        end_line      = 2,
        chunk_type    = "function",
        function_name = function_name,
        char_count    = len(text),
        token_count   = len(text) // 4,
    )


def make_fake_engine() -> EmbeddingEngine:
    """
    Creates an EmbeddingEngine with a mocked model.
    The mock returns deterministic random vectors for testing
    without loading the real 90MB model.
    """
    engine = EmbeddingEngine(model_name=EMBEDDING_MODEL, use_cache=False)

    # Create a mock model that returns plausible embedding vectors
    mock_model = MagicMock()

    def fake_encode(texts, normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True, batch_size=32):
        # Return a fixed-size numpy array for each input text
        # Using ones so normalization gives predictable results
        vectors = np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32)
        if normalize_embeddings:
            # L2 normalize each row
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
        return vectors

    mock_model.encode = fake_encode

    # Inject the mock model and mark as ready
    engine._model    = mock_model
    engine._is_ready = True
    return engine


# ── EmbeddingCache Tests ──────────────────────────────────────────────────────

class TestEmbeddingCache:

    def setup_method(self, tmp_path=None):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()
        self.cache   = EmbeddingCache(cache_dir=self.tmp_dir)

    def test_miss_returns_none(self):
        result = self.cache.get("never cached text", EMBEDDING_MODEL)
        assert result is None

    def test_set_then_get_returns_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        self.cache.set("some text", EMBEDDING_MODEL, embedding)
        result = self.cache.get("some text", EMBEDDING_MODEL)
        assert result == embedding

    def test_different_text_is_cache_miss(self):
        self.cache.set("text A", EMBEDDING_MODEL, [0.1])
        result = self.cache.get("text B", EMBEDDING_MODEL)
        assert result is None

    def test_different_model_is_cache_miss(self):
        # Same text, different model → different key → cache miss
        self.cache.set("text", "model-v1", [0.1, 0.2])
        result = self.cache.get("text", "model-v2")
        assert result is None

    def test_cache_key_is_deterministic(self):
        # Same inputs always produce the same key
        k1 = self.cache._key("hello world", EMBEDDING_MODEL)
        k2 = self.cache._key("hello world", EMBEDDING_MODEL)
        assert k1 == k2

    def test_cache_size_increments(self):
        assert self.cache.size() == 0
        self.cache.set("text 1", EMBEDDING_MODEL, [0.1])
        assert self.cache.size() == 1
        self.cache.set("text 2", EMBEDDING_MODEL, [0.2])
        assert self.cache.size() == 2

    def test_same_text_overwrites_cache(self):
        self.cache.set("text", EMBEDDING_MODEL, [0.1])
        self.cache.set("text", EMBEDDING_MODEL, [0.9])   # overwrite
        result = self.cache.get("text", EMBEDDING_MODEL)
        assert result == [0.9]


# ── EmbeddingEngine Tests (mocked model) ─────────────────────────────────────

class TestEmbeddingEngine:

    def setup_method(self):
        self.engine = make_fake_engine()

    def test_embed_query_returns_correct_dimensions(self):
        vector = self.engine.embed_query("where is the login function?")
        assert len(vector) == EMBEDDING_DIM

    def test_embed_query_returns_list_of_floats(self):
        vector = self.engine.embed_query("test query")
        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_embed_query_rejects_empty_string(self):
        with pytest.raises(ValueError, match="empty"):
            self.engine.embed_query("")

    def test_embed_query_rejects_whitespace(self):
        with pytest.raises(ValueError, match="empty"):
            self.engine.embed_query("   ")

    def test_embed_chunks_returns_result_object(self):
        chunks = [make_chunk(chunk_id="c001"), make_chunk(chunk_id="c002")]
        result = self.engine.embed_chunks(chunks, "test_proj")
        assert isinstance(result, EmbeddingResult)

    def test_embed_chunks_embeds_all_chunks(self):
        chunks = [
            make_chunk(chunk_id="c001", text="def foo(): return 1"),
            make_chunk(chunk_id="c002", text="def bar(): return 2"),
            make_chunk(chunk_id="c003", text="def baz(): return 3"),
        ]
        result = self.engine.embed_chunks(chunks, "test_proj")
        assert result.total_chunks    == 3
        assert result.embedded_chunks == 3
        assert len(result.embedded)   == 3

    def test_embedded_chunks_have_correct_dimensions(self):
        chunks = [make_chunk()]
        result = self.engine.embed_chunks(chunks, "test_proj")
        assert len(result.embedded[0].embedding) == EMBEDDING_DIM

    def test_embedded_chunks_preserve_metadata(self):
        chunk  = make_chunk(
            chunk_id  = "meta_test",
            file_path = "src/auth/login.py",
            language  = "python",
        )
        result = self.engine.embed_chunks([chunk], "test_proj")
        ec     = result.embedded[0]
        assert ec.chunk.chunk_id  == "meta_test"
        assert ec.chunk.file_path == "src/auth/login.py"
        assert ec.chunk.language  == "python"

    def test_empty_chunk_list_returns_empty_result(self):
        result = self.engine.embed_chunks([], "test_proj")
        assert result.total_chunks    == 0
        assert result.embedded_chunks == 0
        assert result.embedded        == []

    def test_result_records_model_name(self):
        result = self.engine.embed_chunks([make_chunk()], "test_proj")
        assert result.model == EMBEDDING_MODEL

    def test_result_records_duration(self):
        result = self.engine.embed_chunks([make_chunk()], "test_proj")
        assert result.duration_seconds >= 0

    def test_not_ready_raises_error(self):
        engine            = EmbeddingEngine(use_cache=False)
        engine._is_ready  = False
        with pytest.raises(RuntimeError, match="not ready"):
            engine.embed_query("test")

    def test_prepare_text_includes_file_path(self):
        chunk = make_chunk(file_path="src/auth/login.py")
        text  = self.engine._prepare_text(chunk)
        assert "src/auth/login.py" in text

    def test_prepare_text_includes_language(self):
        chunk = make_chunk(language="python")
        text  = self.engine._prepare_text(chunk)
        assert "python" in text

    def test_prepare_text_includes_function_name(self):
        chunk = make_chunk(function_name="authenticate")
        text  = self.engine._prepare_text(chunk)
        assert "authenticate" in text

    def test_get_stats_returns_dict(self):
        stats = self.engine.get_stats()
        assert "model"      in stats
        assert "is_ready"   in stats
        assert "dimensions" in stats
        assert stats["is_ready"]   == True
        assert stats["dimensions"] == EMBEDDING_DIM


# ── Batch Processing Tests ────────────────────────────────────────────────────

class TestBatchProcessing:

    def test_large_project_processes_in_batches(self):
        """
        Verifies that embed_chunks handles more chunks than batch_size
        without errors. Uses batch_size=5 with 13 chunks to force
        multiple batches (3 batches: 5, 5, 3).
        """
        engine = EmbeddingEngine(
            model_name = EMBEDDING_MODEL,
            batch_size = 5,
            use_cache  = False,
        )
        # Inject mock model
        mock_model = MagicMock()
        def fake_encode(texts, **kwargs):
            return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32)
        mock_model.encode = fake_encode
        engine._model    = mock_model
        engine._is_ready = True

        # Create 13 chunks — forces 3 batches with batch_size=5
        chunks = [
            make_chunk(
                chunk_id=f"chunk_{i:03d}",
                text=f"def function_{i}():\n    return {i}"
            )
            for i in range(13)
        ]

        result = engine.embed_chunks(chunks, "batch_test")
        assert result.total_chunks    == 13
        assert result.embedded_chunks == 13

    def test_single_chunk_project(self):
        """Edge case: project with exactly one chunk."""
        engine = make_fake_engine()
        chunks = [make_chunk()]
        result = engine.embed_chunks(chunks, "single_test")
        assert result.embedded_chunks == 1

    def test_embeddings_are_normalized(self):
        """
        Verifies embeddings are L2-normalized (unit vectors).
        Required for cosine similarity to work correctly.
        For a unit vector: sum(x^2) = 1.0
        """
        engine = EmbeddingEngine(use_cache=False)

        # Use a real mock that returns actual normalized vectors
        mock_model = MagicMock()
        def normalized_encode(texts, normalize_embeddings=True, **kwargs):
            # Return genuinely normalized vectors
            vecs = np.random.randn(len(texts), EMBEDDING_DIM).astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs  = vecs / norms
            return vecs
        mock_model.encode = normalized_encode
        engine._model    = mock_model
        engine._is_ready = True

        chunks = [make_chunk()]
        result = engine.embed_chunks(chunks, "norm_test")

        vec  = result.embedded[0].embedding
        norm = sum(x**2 for x in vec) ** 0.5
        # Should be very close to 1.0 (floating point tolerance)
        assert abs(norm - 1.0) < 0.01, f"Not normalized: norm={norm}"


# ── Cache Integration Tests ───────────────────────────────────────────────────

class TestCacheIntegration:

    def setup_method(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()

    def test_second_embed_uses_cache(self):
        """
        Embeds the same chunk twice.
        Second call should be served entirely from cache
        (model.encode should only be called once).
        """
        engine = EmbeddingEngine(
            model_name = EMBEDDING_MODEL,
            use_cache  = True,
        )
        engine._cache = EmbeddingCache(cache_dir=self.tmp_dir)

        call_count = {"n": 0}
        mock_model = MagicMock()
        def counting_encode(texts, **kwargs):
            call_count["n"] += 1
            return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32)
        mock_model.encode = counting_encode
        engine._model    = mock_model
        engine._is_ready = True

        chunk = make_chunk(text="def cached_function():\n    pass")

        # First call — hits model
        r1 = engine.embed_chunks([chunk], "proj")
        assert call_count["n"] == 1

        # Second call — same chunk, should hit cache
        r2 = engine.embed_chunks([chunk], "proj")
        # model.encode should NOT have been called again
        assert call_count["n"] == 1
        assert r2.cached_chunks == 1

    def test_different_chunks_not_cached(self):
        """Different text → different cache key → model called for each."""
        engine = EmbeddingEngine(
            model_name = EMBEDDING_MODEL,
            use_cache  = True,
        )
        engine._cache = EmbeddingCache(cache_dir=self.tmp_dir)

        call_count = {"n": 0}
        mock_model = MagicMock()
        def counting_encode(texts, **kwargs):
            call_count["n"] += len(texts)
            return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32)
        mock_model.encode = counting_encode
        engine._model    = mock_model
        engine._is_ready = True

        chunk_a = make_chunk(
            chunk_id="a", text="def function_alpha():\n    return 1"
        )
        chunk_b = make_chunk(
            chunk_id="b", text="def function_beta():\n    return 2"
        )

        engine.embed_chunks([chunk_a], "proj")
        engine.embed_chunks([chunk_b], "proj")

        # Both should have hit the model (different text)
        assert call_count["n"] == 2


# ── Real Model Tests (slow — skipped by default) ─────────────────────────────

@pytest.mark.slow
class TestRealModel:
    """
    Tests that use the actual Sentence Transformer model.
    These download the model (~90MB) on first run.
    Skip in CI with: pytest -m "not slow"
    """

    @pytest.fixture(scope="class")
    def real_engine(self):
        engine = EmbeddingEngine(use_cache=False)
        engine.warm_up()
        return engine

    def test_real_embed_query_dimensions(self, real_engine):
        vec = real_engine.embed_query("where is login implemented?")
        assert len(vec) == EMBEDDING_DIM

    def test_similar_queries_have_high_similarity(self, real_engine):
        """
        Two semantically similar queries should have cosine similarity > 0.7.
        This validates the model is working correctly.
        """
        vec1 = np.array(real_engine.embed_query("user authentication login"))
        vec2 = np.array(real_engine.embed_query("user login and sign in"))
        # Cosine similarity = dot product (vectors are already normalized)
        similarity = float(np.dot(vec1, vec2))
        assert similarity > 0.7, (
            f"Similar queries have low similarity: {similarity:.3f}"
        )

    def test_different_topics_have_low_similarity(self, real_engine):
        """
        Semantically different queries should have cosine similarity < 0.5.
        """
        vec1 = np.array(real_engine.embed_query("database connection pool"))
        vec2 = np.array(real_engine.embed_query("CSS animation keyframe"))
        similarity = float(np.dot(vec1, vec2))
        assert similarity < 0.5, (
            f"Different queries have high similarity: {similarity:.3f}"
        )

    def test_real_embed_chunks(self, real_engine):
        chunks = [
            make_chunk(
                chunk_id = "real_001",
                text     = "def authenticate(user, pwd):\n    return check(user, pwd)",
                language = "python",
            )
        ]
        result = real_engine.embed_chunks(chunks, "real_test")
        assert result.embedded_chunks    == 1
        assert len(result.embedded[0].embedding) == EMBEDDING_DIM