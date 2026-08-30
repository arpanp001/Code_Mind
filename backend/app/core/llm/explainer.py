## Step 5 — Code Explanation Engine


### `backend/app/core/llm/explainer.py`


# backend/app/core/llm/explainer.py
#
# Code explanation engine.
# Takes a code snippet + optional question and returns a
# structured explanation using Gemini.
#
# This is separate from the main RAG chat because:
#   1. It uses a different system prompt (explanation-focused)
#   2. It doesn't need retrieval — the code is provided directly
#   3. It can be called from the /explain endpoint independently

from dataclasses import dataclass
from typing      import Optional

from app.core.llm.gemini   import gemini_client, GeminiResponse
from app.core.llm.prompts  import prompt_builder
from app.utils.logger      import get_logger

logger = get_logger(__name__)


@dataclass
class ExplanationRequest:
    """Request to explain a piece of code."""
    code:       str
    language:   str
    file_path:  str              = ""
    question:   Optional[str]   = None   # Specific question about the code
    project_id: Optional[str]   = None


@dataclass
class ExplanationResponse:
    """Response from the code explanation engine."""
    explanation:     str
    language:        str
    file_path:       str
    tokens_used:     int    = 0
    success:         bool   = True
    error_message:   str    = ""


class CodeExplainer:
    """
    Explains code snippets using Gemini.

    This engine handles the /explain endpoint and is also
    used by the chat engine when a user asks to explain
    a specific function shown in a source card.
    """

    def explain(self, request: ExplanationRequest) -> ExplanationResponse:
        """
        Explains a code snippet.

        Args:
            request: ExplanationRequest with code and context

        Returns:
            ExplanationResponse with the explanation text
        """
        if not request.code.strip():
            return ExplanationResponse(
                explanation   = "No code provided to explain.",
                language      = request.language,
                file_path     = request.file_path,
                success       = False,
                error_message = "Empty code",
            )

        logger.info(
            f"📖 Explaining {request.language} code "
            f"({len(request.code)} chars) "
            f"from {request.file_path or 'unknown'}"
        )

        # Build the explanation prompt
        prompt = prompt_builder.build_explanation_prompt(
            code      = request.code,
            language  = request.language,
            file_path = request.file_path,
            question  = request.question,
        )

        # Call Gemini
        gemini_response = gemini_client.generate(prompt)

        if not gemini_response.success:
            return ExplanationResponse(
                explanation   = gemini_response.text,
                language      = request.language,
                file_path     = request.file_path,
                tokens_used   = gemini_response.total_tokens,
                success       = False,
                error_message = gemini_response.error_message,
            )

        return ExplanationResponse(
            explanation = gemini_response.text,
            language    = request.language,
            file_path   = request.file_path,
            tokens_used = gemini_response.total_tokens,
            success     = True,
        )


# Module-level singleton
code_explainer = CodeExplainer()
