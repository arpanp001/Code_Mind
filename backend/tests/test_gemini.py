import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses   import dataclass, field

from app.core.llm.prompts    import PromptBuilder, PromptTemplate
from app.core.llm.gemini     import GeminiClient, GeminiResponse
from app.core.llm.explainer  import CodeExplainer, ExplanationRequest
from app.core.llm.rag_generator import (
    RAGGenerator, RAGAnswer, detect_query_type
)
from app.core.rag.retriever  import RetrievalResponse
from app.core.rag.reranker   import RankedChunk
from app.core.rag.vectorstore import RetrievedChunk
from app.core.rag.context_assembler import AssembledContext


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_gemini_response(
    text:    str  = "This is a test answer.",
    success: bool = True,
    tokens:  int  = 100,
) -> GeminiResponse:
    return GeminiResponse(
        text             = text,
        model            = "gemini-1.5-flash",
        prompt_tokens    = tokens // 2,
        response_tokens  = tokens // 2,
        total_tokens     = tokens,
        duration_seconds = 0.5,
        finish_reason    = "STOP",
        success          = success,
        error_message    = "" if success else "Test error",
    )


def make_retrieved_chunk(
    chunk_id:      str   = "test_001",
    text:          str   = "def authenticate(): pass",
    file_path:     str   = "src/auth.py",
    function_name: str   = "authenticate",
    similarity:    float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id      = chunk_id,
        text          = text,
        file_path     = file_path,
        language      = "python",
        start_line    = 1,
        end_line      = 5,
        chunk_type    = "function",
        function_name = function_name,
        class_name    = "",
        similarity    = similarity,
        project_id    = "test_project",
    )


def make_ranked_chunk(chunk: RetrievedChunk = None) -> RankedChunk:
    c = chunk or make_retrieved_chunk()
    return RankedChunk(
        chunk            = c,
        final_score      = 0.85,
        similarity_score = c.similarity,
        type_score       = 0.9,
        name_score       = 0.8,
        appear_score     = 0.5,
        appearances      = 2,
    )


def make_retrieval_response(
    query:   str = "where is login implemented?",
    chunks:  int = 2,
) -> RetrievalResponse:
    ranked = [
        make_ranked_chunk(make_retrieved_chunk(
            chunk_id  = f"chunk_{i}",
            file_path = f"src/file_{i}.py",
        ))
        for i in range(chunks)
    ]
    context = AssembledContext(
        context_text     = "def authenticate(): pass\ndef login(): pass",
        chunks_used      = chunks,
        tokens_used      = 150,
        files_referenced = [f"src/file_{i}.py" for i in range(chunks)],
    )
    return RetrievalResponse(
        query            = query,
        project_id       = "test_project",
        ranked_chunks    = ranked,
        context          = context,
        expanded_queries = [query, "authentication function"],
        total_found      = chunks,
        search_time_ms   = 42.0,
    )


# ── PromptBuilder Tests ───────────────────────────────────────────────────────

class TestPromptBuilder:

    def setup_method(self):
        self.builder = PromptBuilder()

    def test_build_chat_prompt_returns_template(self):
        result = self.builder.build_chat_prompt(
            question = "where is login?",
            context  = "def login(): pass",
        )
        assert isinstance(result, PromptTemplate)

    def test_chat_prompt_includes_question(self):
        result = self.builder.build_chat_prompt(
            question = "where is login implemented?",
            context  = "some code",
        )
        assert "where is login implemented?" in result.user_message

    def test_chat_prompt_includes_context(self):
        context = "def my_unique_function(): pass"
        result  = self.builder.build_chat_prompt(
            question = "test question",
            context  = context,
        )
        assert "my_unique_function" in result.user_message

    def test_chat_prompt_has_system_prompt(self):
        result = self.builder.build_chat_prompt("q", "ctx")
        assert len(result.system_prompt) > 50
        assert "CodeMind" in result.system_prompt

    def test_explanation_prompt_includes_code(self):
        code   = "def secret_function_xyz(): return 42"
        result = self.builder.build_explanation_prompt(
            code     = code,
            language = "python",
        )
        assert "secret_function_xyz" in result.user_message

    def test_explanation_prompt_includes_language(self):
        result = self.builder.build_explanation_prompt(
            code     = "const x = 1;",
            language = "javascript",
        )
        assert "javascript" in result.user_message.lower()

    def test_explanation_prompt_includes_file_path(self):
        result = self.builder.build_explanation_prompt(
            code      = "def foo(): pass",
            language  = "python",
            file_path = "src/utils/helpers.py",
        )
        assert "src/utils/helpers.py" in result.user_message

    def test_explanation_prompt_includes_specific_question(self):
        result = self.builder.build_explanation_prompt(
            code      = "def foo(): pass",
            language  = "python",
            question  = "What are the side effects?",
        )
        assert "side effects" in result.user_message

    def test_no_context_prompt_acknowledges_missing_context(self):
        result = self.builder.build_no_context_prompt("test question")
        assert "no relevant code" in result.user_message.lower() or \
               "not found" in result.user_message.lower()

    def test_architecture_prompt_includes_question(self):
        result = self.builder.build_architecture_prompt(
            question = "Why was JWT chosen?",
            context  = "some code",
        )
        assert "Why was JWT chosen?" in result.user_message


# ── Query Type Detection Tests ─────────────────────────────────────────────────

class TestQueryTypeDetection:

    def test_explain_keywords_detected(self):
        assert detect_query_type("explain this function")   == "explain"
        assert detect_query_type("what does this code do")  == "explain"
        assert detect_query_type("how does authentication work") == "explain"

    def test_architecture_keywords_detected(self):
        assert detect_query_type("why was JWT chosen")       == "architecture"
        assert detect_query_type("what design pattern is used") == "architecture"

    def test_general_query_detected(self):
        assert detect_query_type("where is login implemented") == "general"
        assert detect_query_type("find the database connection") == "general"

    def test_case_insensitive(self):
        assert detect_query_type("EXPLAIN this function") == "explain"
        assert detect_query_type("WHERE is login")        == "general"


# ── GeminiClient Tests (mocked) ───────────────────────────────────────────────

class TestGeminiClient:

    def make_client_with_mock(self, response_text="Test answer"):
        """Creates a GeminiClient with mocked Gemini SDK."""
        client = GeminiClient()
        client._configured = True

        mock_model    = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()

        mock_candidate.finish_reason = "STOP"
        mock_response.candidates     = [mock_candidate]
        mock_response.text           = response_text

        mock_usage = MagicMock()
        mock_usage.prompt_token_count     = 50
        mock_usage.candidates_token_count = 30
        mock_usage.total_token_count      = 80
        mock_response.usage_metadata = mock_usage

        mock_model.generate_content.return_value = mock_response
        client._model = mock_model

        return client, mock_model

    def test_generate_returns_response_object(self):
        client, _ = self.make_client_with_mock()
        with patch('google.generativeai.GenerativeModel') as mock_gm:
            mock_gm.return_value._model = MagicMock()
            prompt   = PromptTemplate(
                system_prompt = "You are helpful.",
                user_message  = "Test question",
            )
            # Patch the inner model creation
            with patch.object(client, '_configure'):
                client._configured = True
                import google.generativeai as genai
                with patch.object(genai, 'GenerativeModel') as mock_model_cls:
                    mock_instance = MagicMock()
                    mock_instance.generate_content.return_value = MagicMock(
                        candidates = [MagicMock(finish_reason="STOP")],
                        text       = "The answer is 42",
                        usage_metadata = None,
                    )
                    mock_model_cls.return_value = mock_instance

                    response = client.generate(prompt)
                    assert isinstance(response, GeminiResponse)

    def test_generate_extracts_text(self):
        client, _ = self.make_client_with_mock("Login is in auth.py")
        with patch('google.generativeai.GenerativeModel') as mock_gm:
            mock_instance = MagicMock()
            mock_instance.generate_content.return_value = MagicMock(
                candidates     = [MagicMock(finish_reason="STOP")],
                text           = "Login is in auth.py",
                usage_metadata = None,
            )
            mock_gm.return_value = mock_instance
            with patch.object(client, '_configure'):
                client._configured = True
                prompt   = PromptTemplate("system", "user")
                response = client.generate(prompt)
                assert "Login is in auth.py" in response.text

    def test_handle_error_rate_limit(self):
        client = GeminiClient()
        error  = Exception("quota exceeded 429 rate limit")
        msg    = client._handle_error(error)
        assert "rate limit" in msg.lower() or "quota" in msg.lower()

    def test_handle_error_invalid_key(self):
        client = GeminiClient()
        error  = Exception("invalid api_key provided")
        msg    = client._handle_error(error)
        assert "api key" in msg.lower() or "invalid" in msg.lower()

    def test_handle_error_timeout(self):
        client = GeminiClient()
        error  = Exception("request timeout exceeded")
        msg    = client._handle_error(error)
        assert "timeout" in msg.lower()

    def test_get_stats_returns_dict(self):
        client = GeminiClient()
        stats  = client.get_stats()
        assert "model"       in stats
        assert "temperature" in stats
        assert "max_tokens"  in stats
        assert "configured"  in stats


# ── CodeExplainer Tests ───────────────────────────────────────────────────────

class TestCodeExplainer:

    def make_explainer_with_mock(self, response_text="This function does X"):
        explainer = CodeExplainer()
        return explainer, response_text

    def test_empty_code_returns_error(self):
        explainer = CodeExplainer()
        response  = explainer.explain(ExplanationRequest(
            code     = "   ",
            language = "python",
        ))
        assert response.success       == False
        assert "empty" in response.error_message.lower() or \
               "no code" in response.explanation.lower()

    def test_explain_request_preserves_language(self):
        with patch('app.core.llm.explainer.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response()
            explainer = CodeExplainer()
            response  = explainer.explain(ExplanationRequest(
                code     = "const x = 1;",
                language = "javascript",
            ))
            assert response.language == "javascript"

    def test_explain_request_preserves_file_path(self):
        with patch('app.core.llm.explainer.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response()
            explainer = CodeExplainer()
            response  = explainer.explain(ExplanationRequest(
                code      = "def foo(): pass",
                language  = "python",
                file_path = "src/utils.py",
            ))
            assert response.file_path == "src/utils.py"

    def test_successful_explanation_has_success_true(self):
        with patch('app.core.llm.explainer.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response(
                text    = "This function authenticates users",
                success = True,
                tokens  = 80,
            )
            explainer = CodeExplainer()
            response  = explainer.explain(ExplanationRequest(
                code     = "def authenticate(): pass",
                language = "python",
            ))
            assert response.success == True
            assert "authenticate" in response.explanation.lower() or \
                   len(response.explanation) > 10


# ── RAGGenerator Tests ─────────────────────────────────────────────────────────

class TestRAGGenerator:

    def test_generate_returns_rag_answer(self):
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response(
                text   = "Login is implemented in src/auth.py",
                tokens = 120,
            )
            generator  = RAGGenerator()
            retrieval  = make_retrieval_response("where is login?")
            answer     = generator.generate(retrieval)
            assert isinstance(answer, RAGAnswer)

    def test_generate_returns_gemini_text(self):
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            expected_text = "Login is in src/auth/login.py line 42"
            mock_client.generate.return_value = make_gemini_response(
                text = expected_text
            )
            generator = RAGGenerator()
            retrieval = make_retrieval_response()
            answer    = generator.generate(retrieval)
            assert answer.answer == expected_text

    def test_generate_records_token_count(self):
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response(tokens=200)
            generator = RAGGenerator()
            answer    = generator.generate(make_retrieval_response())
            assert answer.tokens_used == 200

    def test_generate_records_files_referenced(self):
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response()
            generator = RAGGenerator()
            retrieval = make_retrieval_response(chunks=3)
            answer    = generator.generate(retrieval)
            assert len(answer.files_referenced) > 0

    def test_no_context_uses_fallback_prompt(self):
        """When no chunks found, should use no_context prompt."""
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response(
                text = "No relevant code was found"
            )
            with patch('app.core.llm.rag_generator.prompt_builder') as mock_pb:
                mock_pb.build_no_context_prompt.return_value = PromptTemplate(
                    system_prompt = "system", user_message = "user"
                )
                mock_pb.build_chat_prompt.return_value = PromptTemplate(
                    system_prompt = "system", user_message = "user"
                )
                generator = RAGGenerator()
                # Retrieval with no chunks
                retrieval = make_retrieval_response(chunks=0)
                generator.generate(retrieval)
                mock_pb.build_no_context_prompt.assert_called_once()

    def test_architecture_query_uses_architecture_prompt(self):
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response()
            with patch('app.core.llm.rag_generator.prompt_builder') as mock_pb:
                mock_pb.build_architecture_prompt.return_value = PromptTemplate(
                    system_prompt = "system", user_message = "user"
                )
                mock_pb.build_chat_prompt.return_value = PromptTemplate(
                    system_prompt = "system", user_message = "user"
                )
                generator = RAGGenerator()
                retrieval = make_retrieval_response(
                    query = "why was JWT chosen for authentication?"
                )
                generator.generate(retrieval)
                mock_pb.build_architecture_prompt.assert_called_once()

    def test_success_flag_propagates(self):
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response(success=True)
            generator = RAGGenerator()
            answer    = generator.generate(make_retrieval_response())
            assert answer.success == True

    def test_expanded_queries_included_in_answer(self):
        with patch('app.core.llm.rag_generator.gemini_client') as mock_client:
            mock_client.generate.return_value = make_gemini_response()
            generator = RAGGenerator()
            retrieval = make_retrieval_response()
            answer    = generator.generate(retrieval)
            assert isinstance(answer.expanded_queries, list)