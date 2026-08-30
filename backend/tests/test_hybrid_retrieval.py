# backend/tests/test_hybrid_retrieval.py
#
# Test suite for Improvement 1: Hybrid Retrieval + Intelligent Re-ranking
#
# Run with:
#   pytest tests/test_hybrid_retrieval.py -v
#
# All tests use mocked ChromaDB — no real backend needed.

import pytest
from dataclasses import dataclass, field
from app.core.rag.hybrid_search import (
    HybridSearcher,
    tokenize_code,
    score_file_path,
    score_exact_identifier_match,
    extract_identifiers,
)
from app.core.rag.reranker import ReRanker, RankedChunk
from app.core.rag.context_assembler import ContextAssembler, AssembledContext
from app.core.rag.vectorstore import RetrievedChunk
from app.core.rag.fusion import FusedResult


# ── Test helpers ──────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id:      str   = "test_001",
    text:          str   = "def foo(): pass",
    file_path:     str   = "src/foo.py",
    language:      str   = "python",
    chunk_type:    str   = "function",
    function_name: str   = "foo",
    class_name:    str   = "",
    similarity:    float = 0.75,
    start_line:    int   = 1,
    end_line:      int   = 3,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id      = chunk_id,
        text          = text,
        file_path     = file_path,
        language      = language,
        start_line    = start_line,
        end_line      = end_line,
        chunk_type    = chunk_type,
        function_name = function_name,
        class_name    = class_name,
        similarity    = similarity,
        project_id    = "test_project",
        chunk_index   = 0,
    )


def make_fused(chunk: RetrievedChunk, appearances: int = 1) -> FusedResult:
    return FusedResult(
        chunk           = chunk,
        rrf_score       = 0.5,
        appearances     = appearances,
        best_similarity = chunk.similarity,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tokeniser tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenizeCode:

    def test_splits_camel_case(self):
        tokens = tokenize_code("authenticateUser")
        assert "authenticate" in tokens
        assert "user" in tokens

    def test_splits_snake_case(self):
        tokens = tokenize_code("create_jwt_token")
        assert "create" in tokens
        assert "jwt"    in tokens
        assert "token"  in tokens

    def test_splits_pascal_case(self):
        tokens = tokenize_code("UserAuthService")
        assert "user"    in tokens
        assert "auth"    in tokens
        assert "service" in tokens

    def test_removes_short_tokens(self):
        tokens = tokenize_code("a b c def foo")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "foo" in tokens

    def test_removes_pure_digits(self):
        tokens = tokenize_code("line 42 variable_99 ok")
        assert "42" not in tokens
        assert "variable" in tokens or "99" not in tokens

    def test_handles_empty_string(self):
        assert tokenize_code("") == []

    def test_lowercases_all_tokens(self):
        tokens = tokenize_code("AuthService")
        assert all(t == t.lower() for t in tokens)

    def test_real_code_snippet(self):
        code   = "def DatabaseConnectionPool(timeout=30): pass"
        tokens = tokenize_code(code)
        assert "database"   in tokens
        assert "connection" in tokens
        assert "pool"       in tokens
        assert "timeout"    in tokens


# ══════════════════════════════════════════════════════════════════════════════
# 2. Path scoring tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreFilePath:

    def test_src_directory_scores_high(self):
        assert score_file_path("src/auth/login.py")   >= 0.9
        assert score_file_path("src/core/service.py") >= 0.9

    def test_app_directory_scores_high(self):
        assert score_file_path("app/models/user.py")  >= 0.9

    def test_core_directory_scores_high(self):
        assert score_file_path("core/utils/helpers.py") >= 0.9

    def test_api_directory_scores_high(self):
        assert score_file_path("api/handlers/auth.go") >= 0.8

    def test_docs_directory_scores_low(self):
        assert score_file_path("docs/authentication.md") < 0.3

    def test_readme_scores_very_low(self):
        assert score_file_path("README.md") <= 0.15

    def test_test_directory_scores_low(self):
        assert score_file_path("tests/test_auth.py") <= 0.35
        assert score_file_path("test/unit/auth_test.go") <= 0.35

    def test_examples_scores_very_low(self):
        assert score_file_path("examples/basic_usage.py") <= 0.15

    def test_tutorial_scores_very_low(self):
        assert score_file_path("tutorial/getting_started.md") <= 0.15

    def test_entry_points_score_highest(self):
        assert score_file_path("main.py")    == 1.0
        assert score_file_path("app.py")     == 1.0
        assert score_file_path("server.py")  == 1.0
        assert score_file_path("index.js")   == 1.0

    def test_implementation_beats_documentation(self):
        impl_score = score_file_path("src/auth/login.py")
        doc_score  = score_file_path("docs/auth/login_guide.md")
        assert impl_score > doc_score

    def test_services_directory_scores_high(self):
        assert score_file_path("services/auth_service.py") >= 0.9

    def test_lib_directory_scores_high(self):
        assert score_file_path("lib/crypto.js") >= 0.9


# ══════════════════════════════════════════════════════════════════════════════
# 3. Identifier extraction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractIdentifiers:

    def test_extracts_function_name(self):
        ids = extract_identifiers("def authenticate_user(username, password):")
        assert "authenticate_user" in ids
        assert "username"          in ids

    def test_extracts_class_name(self):
        ids = extract_identifiers("class UserAuthService(BaseService):")
        assert "userauthservice"  in ids or "user"    in ids
        assert "baseservice"      in ids or "service" in ids

    def test_splits_camel_case_identifiers(self):
        ids = extract_identifiers("getUserById")
        assert "user" in ids
        assert "get"  in ids

    def test_returns_set(self):
        ids = extract_identifiers("def foo(): return foo()")
        assert isinstance(ids, set)

    def test_handles_empty_string(self):
        assert extract_identifiers("") == set()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Exact identifier match scoring tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreExactIdentifierMatch:

    def test_exact_function_name_match_scores_high(self):
        chunk = make_chunk(function_name="authenticate", chunk_type="function")
        score = score_exact_identifier_match(chunk, ["authenticate"])
        assert score >= 0.9

    def test_no_match_scores_zero(self):
        chunk = make_chunk(function_name="render_page", chunk_type="function")
        score = score_exact_identifier_match(chunk, ["authenticate"])
        assert score == 0.0

    def test_partial_match_scores_partial(self):
        chunk = make_chunk(function_name="authenticate_user", chunk_type="function")
        score = score_exact_identifier_match(chunk, ["authenticate", "password"])
        assert 0.0 < score < 1.0

    def test_empty_query_tokens_returns_zero(self):
        chunk = make_chunk(function_name="authenticate")
        assert score_exact_identifier_match(chunk, []) == 0.0

    def test_class_name_match_scores_high(self):
        chunk = make_chunk(class_name="AuthService", function_name="")
        score = score_exact_identifier_match(chunk, ["auth", "service"])
        assert score >= 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 5. HybridSearcher tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridSearcher:

    def setup_method(self):
        self.searcher = HybridSearcher()

    def test_returns_list(self):
        chunks = [make_chunk()]
        result = self.searcher.hybrid_search(chunks, "test query", top_k=5)
        assert isinstance(result, list)

    def test_respects_top_k(self):
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(10)]
        result = self.searcher.hybrid_search(chunks, "test", top_k=3)
        assert len(result) <= 3

    def test_empty_input_returns_empty(self):
        assert self.searcher.hybrid_search([], "query", top_k=5) == []

    def test_single_chunk_returned_with_score(self):
        chunk  = make_chunk(similarity=0.8)
        result = self.searcher.hybrid_search([chunk], "authenticate", top_k=5)
        assert len(result) == 1
        # Score should be updated (not necessarily 0.8)
        assert 0.0 <= result[0].similarity <= 1.0

    def test_implementation_file_ranked_above_docs(self):
        """
        CORE TEST: src/ file should rank above docs/ file
        even when docs has slightly higher semantic similarity.
        """
        impl_chunk = make_chunk(
            chunk_id  = "impl",
            file_path = "src/auth/login.py",
            text      = "def authenticate(username, password): pass",
            similarity = 0.75,
        )
        doc_chunk = make_chunk(
            chunk_id  = "doc",
            file_path = "docs/authentication.md",
            text      = "# Authentication\nauthenticate users with password",
            similarity = 0.78,   # Slightly higher semantic score
            chunk_type = "heading",
        )
        result = self.searcher.hybrid_search(
            [impl_chunk, doc_chunk],
            "authenticate function",
            top_k=2,
        )
        assert result[0].chunk_id == "impl", (
            f"Expected impl file first, got {result[0].chunk_id} "
            f"(scores: {[r.similarity for r in result]})"
        )

    def test_function_chunk_beats_block_chunk(self):
        """Function chunks should score higher than generic blocks."""
        func_chunk = make_chunk(
            chunk_id   = "func",
            chunk_type = "function",
            similarity = 0.7,
        )
        block_chunk = make_chunk(
            chunk_id   = "block",
            chunk_type = "block",
            similarity = 0.72,
        )
        result = self.searcher.hybrid_search(
            [func_chunk, block_chunk],
            "authenticate user",
            top_k=2,
        )
        assert result[0].chunk_id == "func"

    def test_exact_function_name_match_boosts_ranking(self):
        """
        A chunk whose function_name matches the query should rank higher
        than one with only a slightly higher semantic score.
        """
        exact_chunk = make_chunk(
            chunk_id      = "exact",
            function_name = "DatabaseConnectionPool",
            text          = "def DatabaseConnectionPool(timeout=30): pass",
            similarity    = 0.70,
        )
        no_match_chunk = make_chunk(
            chunk_id      = "no_match",
            function_name = "render_page",
            text          = "def render_page(template): return html",
            similarity    = 0.73,
        )
        result = self.searcher.hybrid_search(
            [exact_chunk, no_match_chunk],
            "DatabaseConnectionPool",
            top_k=2,
        )
        assert result[0].chunk_id == "exact"

    def test_all_scores_in_valid_range(self):
        chunks = [
            make_chunk(chunk_id=f"c{i}", similarity=0.5 + i * 0.05)
            for i in range(5)
        ]
        result = self.searcher.hybrid_search(chunks, "test query", top_k=5)
        for chunk in result:
            assert 0.0 <= chunk.similarity <= 1.0, (
                f"Chunk {chunk.chunk_id} has invalid score {chunk.similarity}"
            )

    def test_results_are_sorted_descending(self):
        chunks = [
            make_chunk(chunk_id=f"c{i}", similarity=0.5 + i * 0.05)
            for i in range(5)
        ]
        result = self.searcher.hybrid_search(chunks, "test", top_k=5)
        scores = [c.similarity for c in result]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ReRanker tests
# ══════════════════════════════════════════════════════════════════════════════

class TestReRanker:

    def setup_method(self):
        self.reranker = ReRanker()

    def test_returns_ranked_chunks(self):
        fused  = [make_fused(make_chunk())]
        result = self.reranker.rerank(fused, "test query", top_k=5)
        assert all(isinstance(r, RankedChunk) for r in result)

    def test_empty_input_returns_empty(self):
        assert self.reranker.rerank([], "query", top_k=5) == []

    def test_respects_top_k(self):
        fused  = [make_fused(make_chunk(chunk_id=f"c{i}")) for i in range(10)]
        result = self.reranker.rerank(fused, "test", top_k=3)
        assert len(result) <= 3

    def test_final_score_in_valid_range(self):
        fused  = [make_fused(make_chunk(chunk_id=f"c{i}")) for i in range(5)]
        result = self.reranker.rerank(fused, "authenticate user", top_k=5)
        for r in result:
            assert 0.0 <= r.final_score <= 1.0

    def test_function_outranks_block_same_similarity(self):
        func_fused  = make_fused(make_chunk(chunk_id="func",  chunk_type="function"))
        block_fused = make_fused(make_chunk(chunk_id="block", chunk_type="block"))
        result      = self.reranker.rerank([func_fused, block_fused], "test", top_k=2)
        assert result[0].chunk.chunk_id == "func"

    def test_implementation_file_beats_documentation(self):
        impl_fused = make_fused(make_chunk(
            chunk_id  = "impl",
            file_path = "src/auth.py",
            similarity = 0.75,
        ))
        doc_fused = make_fused(make_chunk(
            chunk_id  = "doc",
            file_path = "docs/auth.md",
            similarity = 0.80,
            chunk_type = "heading",
        ))
        result = self.reranker.rerank([impl_fused, doc_fused], "authenticate", top_k=2)
        assert result[0].chunk.chunk_id == "impl", (
            f"Expected impl first, got {result[0].chunk.chunk_id}"
        )

    def test_name_match_boosts_score(self):
        """Chunk whose function_name matches query should rank higher."""
        matching = make_fused(make_chunk(
            chunk_id      = "match",
            function_name = "authenticate",
            similarity    = 0.70,
        ))
        no_match = make_fused(make_chunk(
            chunk_id      = "no_match",
            function_name = "render_page",
            similarity    = 0.73,
        ))
        result = self.reranker.rerank([matching, no_match], "authenticate", top_k=2)
        assert result[0].chunk.chunk_id == "match"

    def test_ranked_chunk_has_path_score(self):
        fused  = [make_fused(make_chunk(file_path="src/auth.py"))]
        result = self.reranker.rerank(fused, "test", top_k=5)
        assert hasattr(result[0], 'path_score')
        assert result[0].path_score >= 0.9   # src/ should score high

    def test_results_sorted_by_final_score(self):
        fused  = [make_fused(make_chunk(chunk_id=f"c{i}", similarity=0.5 + i * 0.05))
                  for i in range(5)]
        result = self.reranker.rerank(fused, "test", top_k=5)
        scores = [r.final_score for r in result]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ContextAssembler tests
# ══════════════════════════════════════════════════════════════════════════════

class TestContextAssembler:

    def setup_method(self):
        self.assembler = ContextAssembler(token_budget=2000)

    def _make_ranked(
        self,
        chunk_id:   str   = "c1",
        file_path:  str   = "src/auth.py",
        text:       str   = "def foo(): pass",
        final_score: float = 0.8,
        path_score:  float = 1.0,
        chunk_type:  str   = "function",
    ) -> RankedChunk:
        chunk = make_chunk(
            chunk_id   = chunk_id,
            file_path  = file_path,
            text       = text,
            chunk_type = chunk_type,
        )
        return RankedChunk(
            chunk            = chunk,
            final_score      = final_score,
            similarity_score = 0.8,
            type_score       = 1.0,
            name_score       = 0.5,
            appear_score     = 0.5,
            path_score       = path_score,
            appearances      = 1,
        )

    def test_returns_assembled_context(self):
        rc     = self._make_ranked()
        result = self.assembler.assemble([rc], "test query")
        assert isinstance(result, AssembledContext)

    def test_includes_file_path(self):
        rc     = self._make_ranked(file_path="src/auth/login.py")
        result = self.assembler.assemble([rc], "test")
        assert "src/auth/login.py" in result.context_text

    def test_includes_code_text(self):
        rc     = self._make_ranked(text="def authenticate(): return jwt_token")
        result = self.assembler.assemble([rc], "test")
        assert "authenticate" in result.context_text

    def test_includes_relevance_percentage(self):
        rc     = self._make_ranked(final_score=0.873)
        result = self.assembler.assemble([rc], "test")
        assert "87.3%" in result.context_text

    def test_implementation_before_documentation(self):
        """
        CORE TEST: Implementation files appear before documentation
        in the assembled context, regardless of raw score.
        """
        impl_rc = self._make_ranked(
            chunk_id    = "impl",
            file_path   = "src/auth/login.py",
            final_score = 0.80,
            path_score  = 1.0,   # implementation
        )
        doc_rc = self._make_ranked(
            chunk_id    = "doc",
            file_path   = "docs/auth.md",
            final_score = 0.82,   # Slightly higher score
            path_score  = 0.1,    # documentation
            chunk_type  = "heading",
        )
        result = self.assembler.assemble([impl_rc, doc_rc], "authenticate")
        impl_pos = result.context_text.find("src/auth/login.py")
        doc_pos  = result.context_text.find("docs/auth.md")
        assert impl_pos < doc_pos, (
            "Implementation file should appear before documentation "
            f"(impl_pos={impl_pos}, doc_pos={doc_pos})"
        )

    def test_token_budget_respected(self):
        assembler = ContextAssembler(token_budget=50)
        # Create many chunks that would exceed budget
        ranked = [
            self._make_ranked(
                chunk_id = f"c{i}",
                text     = "def foo(): " + "x = 1\n" * 100,
            )
            for i in range(20)
        ]
        result = assembler.assemble(ranked, "test")
        assert result.tokens_used <= 200   # some buffer for format overhead

    def test_truncated_flag_set_when_over_budget(self):
        assembler = ContextAssembler(token_budget=20)
        ranked    = [
            self._make_ranked(
                chunk_id = f"c{i}",
                text     = "def function_with_long_name(): " + "pass\n" * 30,
            )
            for i in range(10)
        ]
        result = assembler.assemble(ranked, "test")
        assert result.truncated is True

    def test_files_referenced_populated(self):
        ranked = [
            self._make_ranked(chunk_id="c1", file_path="src/a.py"),
            self._make_ranked(chunk_id="c2", file_path="src/b.py"),
        ]
        result = self.assembler.assemble(ranked, "test")
        assert "src/a.py" in result.files_referenced
        assert "src/b.py" in result.files_referenced

    def test_impl_doc_counts_tracked(self):
        ranked = [
            self._make_ranked(chunk_id="impl", file_path="src/auth.py",   path_score=1.0),
            self._make_ranked(chunk_id="doc",  file_path="docs/readme.md", path_score=0.1),
        ]
        result = self.assembler.assemble(ranked, "test")
        assert result.impl_files >= 1
        assert result.doc_files  >= 1

    def test_empty_input_returns_no_context_message(self):
        result = self.assembler.assemble([], "test")
        assert "No relevant" in result.context_text or result.chunks_used == 0