# backend/tests/test_retrieval.py
# Run with: pytest tests/test_retrieval.py -v

import pytest
import numpy as np
from app.core.rag.query_expander  import QueryExpander, ExpandedQuery
from app.core.rag.fusion          import ReciprocalRankFusion, FusedResult
from app.core.rag.reranker        import ReRanker, RankedChunk
from app.core.rag.hybrid_search   import HybridSearcher, tokenize_code
from app.core.rag.context_assembler import ContextAssembler
from app.core.rag.vectorstore     import RetrievedChunk
from app.core.rag.reranker        import RankedChunk


def make_chunk(
    chunk_id:      str   = "chunk_001",
    text:          str   = "def authenticate(user, pwd): pass",
    file_path:     str   = "src/auth.py",
    language:      str   = "python",
    chunk_type:    str   = "function",
    function_name: str   = "authenticate",
    similarity:    float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id      = chunk_id,
        text          = text,
        file_path     = file_path,
        language      = language,
        start_line    = 1,
        end_line      = 5,
        chunk_type    = chunk_type,
        function_name = function_name,
        class_name    = "",
        similarity    = similarity,
        project_id    = "test_project",
    )


def make_ranked(chunk: RetrievedChunk, score: float = 0.8) -> RankedChunk:
    return RankedChunk(
        chunk            = chunk,
        final_score      = score,
        similarity_score = chunk.similarity,
        type_score       = 0.8,
        name_score       = 0.5,
        appear_score     = 0.5,
        appearances      = 1,
    )


# ── QueryExpander Tests ───────────────────────────────────────────────────────

class TestQueryExpander:

    def setup_method(self):
        self.expander = QueryExpander(max_variations=3)

    def test_original_always_included(self):
        result = self.expander.expand("where is login implemented?")
        assert result.original in result.all_queries
        assert result.all_queries[0] == result.original

    def test_returns_expanded_query_object(self):
        result = self.expander.expand("authenticate user")
        assert isinstance(result, ExpandedQuery)

    def test_all_queries_no_duplicates(self):
        result = self.expander.expand("login authentication")
        queries = result.all_queries
        assert len(queries) == len(set(queries))

    def test_empty_query_returns_original(self):
        result = self.expander.expand("")
        assert result.all_queries == [""] or result.variations == []

    def test_domain_expansion_login(self):
        result = self.expander.expand("login function")
        # Should generate variations containing auth synonyms
        all_q  = " ".join(result.all_queries).lower()
        assert any(
            term in all_q
            for term in ["auth", "credential", "sign", "session", "login"]
        )

    def test_max_variations_respected(self):
        result = self.expander.expand("authenticate user with password")
        assert len(result.variations) <= self.expander.max_variations

    def test_all_queries_are_strings(self):
        result = self.expander.expand("database connection")
        for q in result.all_queries:
            assert isinstance(q, str)

    def test_code_reformulation_where_query(self):
        result = self.expander.expand("where is authentication implemented")
        all_q  = " ".join(result.all_queries).lower()
        # Should contain implementation-focused reformulation
        assert len(result.all_queries) > 1


# ── RRF Fusion Tests ──────────────────────────────────────────────────────────

class TestReciprocalRankFusion:

    def setup_method(self):
        self.fusion = ReciprocalRankFusion(k=60)

    def test_single_list_returns_same_order(self):
        chunks  = [make_chunk(chunk_id=f"c{i}", similarity=1.0-i*0.1) for i in range(3)]
        result  = self.fusion.fuse([chunks], top_k=3)
        ids     = [r.chunk.chunk_id for r in result]
        assert ids == ["c0", "c1", "c2"]

    def test_empty_lists_handled(self):
        result = self.fusion.fuse([], top_k=5)
        assert result == []

    def test_all_empty_sublists(self):
        result = self.fusion.fuse([[], [], []], top_k=5)
        assert result == []

    def test_chunk_in_multiple_lists_gets_higher_score(self):
        # chunk "common" appears in all 3 lists
        common = make_chunk(chunk_id="common", similarity=0.7)
        list1  = [make_chunk(chunk_id="a", similarity=0.9), common]
        list2  = [make_chunk(chunk_id="b", similarity=0.9), common]
        list3  = [make_chunk(chunk_id="c", similarity=0.9), common]

        result = self.fusion.fuse([list1, list2, list3], top_k=5)
        ids    = [r.chunk.chunk_id for r in result]

        # "common" should rank high despite not being #1 in any list
        common_result = next(r for r in result if r.chunk.chunk_id == "common")
        assert common_result.appearances == 3

    def test_top_k_limits_results(self):
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(10)]
        result = self.fusion.fuse([chunks], top_k=3)
        assert len(result) <= 3

    def test_result_has_fused_result_type(self):
        chunks = [make_chunk()]
        result = self.fusion.fuse([chunks])
        assert isinstance(result[0], FusedResult)

    def test_appearances_counted_correctly(self):
        chunk  = make_chunk(chunk_id="shared")
        list1  = [chunk]
        list2  = [chunk]
        result = self.fusion.fuse([list1, list2], top_k=5)
        assert result[0].appearances == 2


# ── ReRanker Tests ────────────────────────────────────────────────────────────

class TestReRanker:

    def setup_method(self):
        self.reranker = ReRanker()

    def _make_fused(self, chunk, appearances=1) -> FusedResult:
        return FusedResult(
            chunk=chunk, rrf_score=0.5,
            appearances=appearances, best_similarity=chunk.similarity
        )

    def test_function_outranks_block_with_same_similarity(self):
        func_chunk  = make_chunk(chunk_id="func",  chunk_type="function", similarity=0.8)
        block_chunk = make_chunk(chunk_id="block", chunk_type="block",    similarity=0.8)
        fused = [self._make_fused(func_chunk), self._make_fused(block_chunk)]
        ranked = self.reranker.rerank(fused, query="authenticate function")
        assert ranked[0].chunk.chunk_id == "func"

    def test_name_match_boosts_score(self):
        matching = make_chunk(
            chunk_id="match", function_name="authenticate", similarity=0.75
        )
        no_match = make_chunk(
            chunk_id="nomatch", function_name="render_template", similarity=0.75
        )
        fused = [self._make_fused(matching), self._make_fused(no_match)]
        ranked = self.reranker.rerank(fused, query="authenticate user")
        assert ranked[0].chunk.chunk_id == "match"

    def test_returns_ranked_chunk_objects(self):
        fused  = [self._make_fused(make_chunk())]
        ranked = self.reranker.rerank(fused, query="test")
        assert isinstance(ranked[0], RankedChunk)

    def test_empty_input_returns_empty(self):
        result = self.reranker.rerank([], query="test")
        assert result == []

    def test_top_k_respected(self):
        fused  = [self._make_fused(make_chunk(chunk_id=f"c{i}")) for i in range(10)]
        ranked = self.reranker.rerank(fused, query="test", top_k=3)
        assert len(ranked) <= 3

    def test_final_score_between_0_and_1(self):
        fused  = [self._make_fused(make_chunk())]
        ranked = self.reranker.rerank(fused, query="authenticate")
        assert 0.0 <= ranked[0].final_score <= 1.0

    def test_keyword_extraction_removes_stopwords(self):
        keywords = self.reranker._extract_keywords(
            "where is the login function implemented"
        )
        assert "where"       not in keywords
        assert "the"         not in keywords
        assert "is"          not in keywords
        assert "implemented" not in keywords
        assert "login"       in keywords


# ── HybridSearcher Tests ──────────────────────────────────────────────────────

class TestHybridSearcher:

    def setup_method(self):
        self.searcher = HybridSearcher(semantic_weight=0.7, bm25_weight=0.3)

    def test_tokenize_splits_camel_case(self):
        tokens = tokenize_code("authenticateUser")
        assert "authenticate" in tokens
        assert "user" in tokens

    def test_tokenize_splits_snake_case(self):
        tokens = tokenize_code("create_jwt_token")
        assert "create" in tokens
        assert "jwt" in tokens
        assert "token" in tokens

    def test_tokenize_removes_short_tokens(self):
        tokens = tokenize_code("x = 1; y = 2")
        assert "x" not in tokens
        assert "y" not in tokens

    def test_hybrid_search_returns_chunks(self):
        chunks = [
            make_chunk(chunk_id="c1", text="def authenticate(user, pwd): pass"),
            make_chunk(chunk_id="c2", text="def render_template(name): pass"),
        ]
        result = self.searcher.hybrid_search(chunks, "authenticate user", top_k=2)
        assert len(result) > 0

    def test_exact_match_chunk_ranked_higher(self):
        chunks = [
            make_chunk(
                chunk_id="exact",
                text="def DatabaseConnectionPool(timeout=30): pass",
                similarity=0.70,
            ),
            make_chunk(
                chunk_id="no_match",
                text="def render_page(template): return html",
                similarity=0.78,  # Slightly higher semantic score
            ),
        ]
        result = self.searcher.hybrid_search(
            chunks, "DatabaseConnectionPool timeout", top_k=2
        )
        assert result[0].chunk_id == "exact"

    def test_empty_results_handled(self):
        result = self.searcher.hybrid_search([], "test query", top_k=5)
        assert result == []

    def test_single_chunk_returned(self):
        chunks = [make_chunk()]
        result = self.searcher.hybrid_search(chunks, "authenticate", top_k=5)
        assert len(result) == 1


# ── ContextAssembler Tests ────────────────────────────────────────────────────

class TestContextAssembler:

    def setup_method(self):
        self.assembler = ContextAssembler(token_budget=500)

    def test_assembles_single_chunk(self):
        ranked = [make_ranked(make_chunk())]
        result = self.assembler.assemble(ranked, "test query")
        assert len(result.context_text) > 0
        assert result.chunks_used == 1

    def test_includes_file_path_in_context(self):
        chunk  = make_chunk(file_path="src/auth/login.py")
        ranked = [make_ranked(chunk)]
        result = self.assembler.assemble(ranked, "test")
        assert "src/auth/login.py" in result.context_text

    def test_includes_function_name_in_context(self):
        chunk  = make_chunk(function_name="authenticate")
        ranked = [make_ranked(chunk)]
        result = self.assembler.assemble(ranked, "test")
        assert "authenticate" in result.context_text

    def test_includes_code_text_in_context(self):
        chunk  = make_chunk(text="def my_unique_function_xyz(): pass")
        ranked = [make_ranked(chunk)]
        result = self.assembler.assemble(ranked, "test")
        assert "my_unique_function_xyz" in result.context_text

    def test_token_budget_respected(self):
        # Create many chunks that together exceed budget
        chunks = [
            make_ranked(make_chunk(
                chunk_id = f"c{i}",
                text     = "x = " + "a" * 200,   # ~200 chars each
            ))
            for i in range(20)
        ]
        result = self.assembler.assemble(chunks, "test", token_budget=100)
        # Should be truncated
        assert result.chunks_used < 20

    def test_truncated_flag_set(self):
        chunks = [
            make_ranked(make_chunk(
                chunk_id=f"c{i}",
                text="def func():\n" + "    pass\n" * 50
            ))
            for i in range(10)
        ]
        result = self.assembler.assemble(chunks, "test", token_budget=50)
        if result.chunks_used < 10:
            assert result.truncated is True

    def test_files_referenced_populated(self):
        chunks = [
            make_ranked(make_chunk(chunk_id="c1", file_path="src/auth.py")),
            make_ranked(make_chunk(chunk_id="c2", file_path="src/models.py")),
        ]
        result = self.assembler.assemble(chunks, "test")
        assert "src/auth.py"   in result.files_referenced
        assert "src/models.py" in result.files_referenced

    def test_format_for_prompt_includes_header(self):
        ranked = [make_ranked(make_chunk())]
        assembled = self.assembler.assemble(ranked, "where is login?")
        prompt_text = self.assembler.format_for_prompt(assembled, "where is login?")
        assert "CODEBASE CONTEXT" in prompt_text
        assert "where is login?"  in prompt_text