# backend/tests/test_integration.py
# Full pipeline integration tests (mocked external services).
# Run with: pytest tests/test_integration.py -v

import pytest
from unittest.mock import patch, MagicMock
from dataclasses   import dataclass, field

from app.core.llm.conversation_memory import ConversationMemoryManager
from app.core.llm.prompts             import PromptBuilder
from app.core.llm.rag_generator       import detect_query_type


# ── Conversation Memory Tests ─────────────────────────────────────────────────

class TestConversationMemory:

    def setup_method(self):
        self.manager = ConversationMemoryManager()

    def test_creates_new_session(self):
        session = self.manager.get_or_create("session_001", "proj_abc")
        assert session.session_id == "session_001"
        assert session.project_id == "proj_abc"

    def test_returns_same_session_on_second_call(self):
        s1 = self.manager.get_or_create("session_002", "proj_abc")
        s2 = self.manager.get_or_create("session_002", "proj_abc")
        assert s1 is s2

    def test_empty_history_returns_empty_string(self):
        session = self.manager.get_or_create("session_003", "proj_abc")
        assert session.get_history_text() == ""

    def test_history_includes_question_and_answer(self):
        session = self.manager.get_or_create("session_004", "proj_abc")
        session.add_exchange("Where is login?", "Login is in auth.py")
        history = session.get_history_text()
        assert "Where is login?"    in history
        assert "Login is in auth.py" in history

    def test_history_has_user_and_assistant_labels(self):
        session = self.manager.get_or_create("session_005", "proj_abc")
        session.add_exchange("Question?", "Answer.")
        history = session.get_history_text()
        assert "User"      in history
        assert "Assistant" in history

    def test_max_history_respected(self):
        session = self.manager.get_or_create("session_006", "proj_abc")
        # Add more than max pairs
        for i in range(10):
            session.add_exchange(f"Q{i}", f"A{i}")
        # deque maxlen ensures only last N messages are kept
        assert len(session.messages) <= 10   # MAX_HISTORY_PAIRS * 2

    def test_clear_session_removes_history(self):
        session = self.manager.get_or_create("session_007", "proj_abc")
        session.add_exchange("Q", "A")
        self.manager.clear_session("session_007")
        # After clear, getting session again creates fresh one
        new_session = self.manager.get_or_create("session_007", "proj_abc")
        assert len(new_session.messages) == 0

    def test_session_expires_check(self):
        import time
        session = self.manager.get_or_create("session_008", "proj_abc")
        # Fresh session is not expired
        assert not session.is_expired()
        # Manually set old timestamp
        session.last_active = time.time() - 7200   # 2 hours ago
        assert session.is_expired()

    def test_multiple_projects_isolated(self):
        s1 = self.manager.get_or_create("sess_A", "proj_1")
        s2 = self.manager.get_or_create("sess_B", "proj_2")
        s1.add_exchange("Project 1 question", "Project 1 answer")
        assert len(s2.messages) == 0   # Project 2 unaffected

    def test_get_stats(self):
        self.manager.get_or_create("stats_sess", "proj_abc")
        stats = self.manager.get_stats()
        assert "active_sessions"  in stats
        assert "max_history_pairs" in stats
        assert stats["active_sessions"] >= 1


# ── Prompt Builder with History Tests ─────────────────────────────────────────

class TestPromptBuilderWithHistory:

    def setup_method(self):
        self.builder = PromptBuilder()

    def test_prompt_with_history_includes_previous_conversation(self):
        result = self.builder.build_chat_prompt_with_history(
            question = "Explain it more",
            context  = "def login(): pass",
            history  = "User: Where is login?\nAssistant: In auth.py",
        )
        assert "Explain it more"    in result.user_message
        assert "Where is login?"    in result.user_message
        assert "In auth.py"         in result.user_message

    def test_prompt_without_history_same_as_basic(self):
        with_history    = self.builder.build_chat_prompt_with_history(
            question = "Where is login?",
            context  = "some code",
            history  = "",
        )
        without_history = self.builder.build_chat_prompt(
            question = "Where is login?",
            context  = "some code",
        )
        # Both should contain the question
        assert "Where is login?" in with_history.user_message
        assert "Where is login?" in without_history.user_message

    def test_history_mentions_previous_conversation(self):
        result = self.builder.build_chat_prompt_with_history(
            question = "What about the password?",
            context  = "code here",
            history  = "User: Where is login?\nAssistant: In auth.py",
        )
        # The prompt should instruct Gemini to use conversation context
        assert "previous conversation" in result.user_message.lower() or \
               "previous" in result.user_message.lower()


# ── Query Type Detection Tests ─────────────────────────────────────────────────

class TestQueryTypeDetectionExtended:

    def test_follow_up_explain_detected(self):
        assert detect_query_type("explain that function") == "explain"

    def test_what_does_detected(self):
        assert detect_query_type("what does this do") == "explain"

    def test_architecture_why_detected(self):
        assert detect_query_type("why was this approach chosen") == "architecture"

    def test_general_where_detected(self):
        assert detect_query_type("where is the config loaded") == "general"

    def test_general_find_detected(self):
        assert detect_query_type("find database initialization") == "general"

    def test_short_query_handled(self):
        # Should not crash on very short queries
        result = detect_query_type("JWT")
        assert result in ("general", "explain", "architecture")

    def test_mixed_keywords_explain_wins(self):
        # "explain" keyword present alongside general words
        result = detect_query_type("explain where login is implemented")
        assert result == "explain"


# ── Full Pipeline Mock Test ────────────────────────────────────────────────────

class TestFullPipelineMocked:
    """
    Tests the complete flow from question to answer
    without hitting real APIs.
    """

    def test_chat_pipeline_returns_answer(self):
        """
        Verifies the complete pipeline:
        question → retrieval → generation → answer
        """
        from app.core.llm.rag_generator  import RAGGenerator
        from app.core.rag.retriever      import RetrievalResponse
        from app.core.rag.reranker       import RankedChunk
        from app.core.rag.vectorstore    import RetrievedChunk
        from app.core.rag.context_assembler import AssembledContext
        from app.core.llm.gemini         import GeminiResponse

        # Build a mock retrieval response
        chunk = RetrievedChunk(
            chunk_id="c001", text="def authenticate(): pass",
            file_path="src/auth.py", language="python",
            start_line=1, end_line=3, chunk_type="function",
            function_name="authenticate", class_name="",
            similarity=0.9, project_id="proj_test",
        )
        ranked = RankedChunk(
            chunk=chunk, final_score=0.85,
            similarity_score=0.9, type_score=1.0,
            name_score=0.8, appear_score=0.5, appearances=2,
        )
        context = AssembledContext(
            context_text="def authenticate(): pass",
            chunks_used=1, tokens_used=50,
            files_referenced=["src/auth.py"],
        )
        retrieval = RetrievalResponse(
            query="where is authentication?",
            project_id="proj_test",
            ranked_chunks=[ranked],
            context=context,
            expanded_queries=["where is authentication?"],
            total_found=1,
            search_time_ms=100.0,
        )

        expected_answer = "Authentication is in src/auth.py"

        with patch('app.core.llm.rag_generator.gemini_client') as mock_gemini:
            with patch('app.core.llm.rag_generator.project_memory') as mock_mem:
                mock_gemini.generate.return_value = GeminiResponse(
                    text="Authentication is in src/auth.py",
                    model="models/gemini-2.5-flash",
                    total_tokens=120, success=True, finish_reason="STOP",
                )
                mock_mem.search_memories.return_value = []

                generator = RAGGenerator()
                answer    = generator.generate(retrieval, "test-project")

                assert answer.answer     == expected_answer
                assert answer.success    == True
                assert answer.tokens_used == 120
                assert "src/auth.py" in answer.files_referenced

    def test_pipeline_handles_no_retrieval_results(self):
        """When retrieval finds nothing, Gemini still gives a helpful response."""
        from app.core.llm.rag_generator import RAGGenerator
        from app.core.rag.retriever     import RetrievalResponse
        from app.core.llm.gemini        import GeminiResponse

        retrieval = RetrievalResponse(
            query="something completely irrelevant",
            project_id="proj_test",
            ranked_chunks=[],
            context=None,
            expanded_queries=[],
            total_found=0,
            search_time_ms=50.0,
        )

        with patch('app.core.llm.rag_generator.gemini_client') as mock_gemini:
            with patch('app.core.llm.rag_generator.prompt_builder') as mock_pb:
                from app.core.llm.prompts import PromptTemplate
                mock_gemini.generate.return_value = GeminiResponse(
                    text="No relevant code was found for your question.",
                    model="models/gemini-2.5-flash",
                    total_tokens=50, success=True, finish_reason="STOP",
                )
                mock_pb.build_no_context_prompt.return_value = PromptTemplate(
                    system_prompt="system", user_message="user"
                )

                generator = RAGGenerator()
                answer    = generator.generate(retrieval)

                # Should use no-context prompt
                mock_pb.build_no_context_prompt.assert_called_once()
                assert answer.success == True