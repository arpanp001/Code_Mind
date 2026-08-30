# backend/app/core/llm/gemini.py

import time
from dataclasses import dataclass
from typing      import Optional, Generator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core           import exceptions as google_exceptions

from app.core.llm.prompts import PromptTemplate
from app.config           import settings
from app.utils.logger     import get_logger

logger = get_logger(__name__)


# ── Model Configuration ────────────────────────────────────────────────────────

PRIMARY_MODEL  = "models/gemini-2.5-flash"
FALLBACK_MODEL = "models/gemini-2.0-flash"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 1024
DEFAULT_TOP_P       = 0.8
DEFAULT_TOP_K       = 40

# Retry delays in seconds — waits between each retry attempt
RETRY_DELAYS = [10, 20, 45]
MAX_RETRIES  = len(RETRY_DELAYS)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


@dataclass
class GeminiResponse:
    """Structured response from a Gemini API call."""
    text:             str
    model:            str
    prompt_tokens:    int   = 0
    response_tokens:  int   = 0
    total_tokens:     int   = 0
    duration_seconds: float = 0.0
    finish_reason:    str   = ""
    success:          bool  = True
    error_message:    str   = ""


# ── Error Classification Helpers ───────────────────────────────────────────────

def _is_rate_limit(error: Exception) -> bool:
    """Detects quota / rate-limit errors."""
    if isinstance(error, google_exceptions.ResourceExhausted):
        return True
    msg = str(error).lower()
    return any(k in msg for k in [
        "429", "quota", "resource exhausted",
        "too many requests", "ratelimitexceeded",
    ])


def _is_retryable(error: Exception) -> bool:
    """Returns True if we should retry this error."""
    if _is_rate_limit(error):
        return True
    if isinstance(error, (
        google_exceptions.ServiceUnavailable,
        google_exceptions.GatewayTimeout,
        google_exceptions.DeadlineExceeded,
    )):
        return True
    msg = str(error).lower()
    return any(k in msg for k in ["500", "503", "timeout", "server error"])


# ── Gemini Client ──────────────────────────────────────────────────────────────

class GeminiClient:
    """
    Gemini API client with auto model discovery and retry on rate limits.

    Retry strategy:
      Attempt 0 → immediate        (primary model)
      Attempt 1 → wait 15s         (primary model)
      Attempt 2 → wait 30s         (fallback model)
      Attempt 3 → wait 60s         (fallback model)
    """

    def __init__(
        self,
        model_name:  str   = PRIMARY_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
    ):
        self.model_name  = model_name
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._configured = False

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _configure(self) -> None:
        """
        Configures the SDK and auto-discovers a working model.
        Called lazily on first use so startup is fast.
        """
        if self._configured:
            return

        api_key = settings.gemini_api_key
        if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com"
            )

        genai.configure(api_key=api_key)

        # Auto-discover: if configured model gives 404, find one that works
        discovered = self._discover_model()
        if discovered != self.model_name:
            logger.info(
                f"🔄 Model '{self.model_name}' → using '{discovered}'"
            )
            self.model_name = discovered

        self._configured = True
        logger.info(f"✅ Gemini configured: {self.model_name}")

    def _discover_model(self) -> str:
        """
        Lists models available to this API key and returns the best one.
        Falls back through a priority list until one is found.
        """
        # Priority list — best models first
        PRIORITY = [
            self.model_name,                   # Try the configured model first
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-lite",
            "models/gemini-flash-latest",
            "models/gemini-pro-latest",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-flash",
        ]

        try:
            available = {
                m.name
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            }
            logger.debug(f"Available models: {sorted(available)}")

            for model in PRIORITY:
                if model in available:
                    return model

            # None of our preferred models — use whatever is available
            if available:
                chosen = sorted(available)[0]
                logger.warning(
                    f"No preferred model found — using: {chosen}"
                )
                return chosen

        except Exception as e:
            logger.warning(f"Model discovery failed: {e}")

        return self.model_name

    # ── Core Generation ────────────────────────────────────────────────────────

    def _build_model(self, model_name: str, system_prompt: str):
        """Creates a GenerativeModel instance."""
        return genai.GenerativeModel(
            model_name         = model_name,
            generation_config  = genai.types.GenerationConfig(
                temperature       = self.temperature,
                max_output_tokens = self.max_tokens,
                top_p             = DEFAULT_TOP_P,
                top_k             = DEFAULT_TOP_K,
            ),
            safety_settings    = SAFETY_SETTINGS,
            system_instruction = system_prompt,
        )

    def _parse_response(
        self,
        response,
        model_name: str,
        duration:   float,
    ) -> GeminiResponse:
        """Parses a raw Gemini API response into GeminiResponse."""
        if not response.candidates:
            return GeminiResponse(
                text             = (
                    "I was unable to generate a response. "
                    "The content may have triggered safety filters. "
                    "Please rephrase your question."
                ),
                model            = model_name,
                duration_seconds = duration,
                finish_reason    = "SAFETY",
                success          = False,
                error_message    = "No candidates returned",
            )

        finish_reason = str(response.candidates[0].finish_reason)

        try:
            text = response.text
        except Exception:
            text          = "Response was filtered. Try rephrasing."
            finish_reason = "SAFETY"

        # Token usage
        pt = rt = tt = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            u  = response.usage_metadata
            pt = getattr(u, "prompt_token_count",    0) or 0
            rt = getattr(u, "candidates_token_count", 0) or 0
            tt = getattr(u, "total_token_count",      0) or 0

        return GeminiResponse(
            text             = text,
            model            = model_name,
            prompt_tokens    = pt,
            response_tokens  = rt,
            total_tokens     = tt,
            duration_seconds = duration,
            finish_reason    = finish_reason,
            success          = True,
        )

    def generate(self, prompt: PromptTemplate) -> GeminiResponse:
        """
        Sends a prompt to Gemini with automatic retry on rate limits.

        Attempt 0: primary model,   no wait
        Attempt 1: primary model,   wait 15s
        Attempt 2: fallback model,  wait 30s
        Attempt 3: fallback model,  wait 60s
        """
        self._configure()

        overall_start          = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):

            # Use fallback model from attempt 2 onward
            model_name = (
                FALLBACK_MODEL
                if attempt >= 2
                else self.model_name
            )

            # Sleep before retry attempts
            if attempt > 0:
                delay = RETRY_DELAYS[attempt - 1]
                logger.warning(
                    f"⏳ Gemini retry {attempt}/{MAX_RETRIES} — "
                    f"waiting {delay}s (model: {model_name})"
                )
                time.sleep(delay)

            call_start = time.time()
            try:
                model    = self._build_model(model_name, prompt.system_prompt)
                response = model.generate_content(prompt.user_message)
                duration = time.time() - call_start

                result = self._parse_response(response, model_name, duration)

                if result.success:
                    logger.info(
                        f"✅ Gemini OK [{model_name}]: "
                        f"{result.total_tokens} tokens, "
                        f"{duration:.2f}s"
                        + (f" (after {attempt} retries)" if attempt > 0 else "")
                    )

                return result

            except Exception as e:
                last_error = e
                duration   = time.time() - call_start

                if _is_retryable(e) and attempt < MAX_RETRIES:
                    logger.warning(
                        f"⚠️  Gemini error "
                        f"(attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                        f"{type(e).__name__}: {str(e)[:80]}"
                    )
                    continue   # go to next iteration → sleep → retry

                # Non-retryable OR out of retries
                logger.error(
                    f"❌ Gemini failed after {attempt + 1} attempt(s): "
                    f"{type(e).__name__}: {str(e)[:150]}"
                )
                break

        # All attempts exhausted — return user-friendly error
        return GeminiResponse(
            text             = self._user_message(last_error),
            model            = self.model_name,
            duration_seconds = time.time() - overall_start,
            finish_reason    = "ERROR",
            success          = False,
            error_message    = self._log_message(last_error),
        )

    def generate_stream(self, prompt: PromptTemplate) -> Generator[str, None, None]:
        """Streams response tokens."""
        self._configure()
        try:
            model    = self._build_model(self.model_name, prompt.system_prompt)
            response = model.generate_content(
                prompt.user_message,
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except google_exceptions.ResourceExhausted:
            yield "\n\n⚠️ Rate limit reached. Wait 60s and retry."
        except Exception as e:
            yield f"\n\n⚠️ Error: {self._log_message(e)}"

    # ── Error Formatting ───────────────────────────────────────────────────────

    # backend/app/core/llm/gemini.py
# Add to _log_message and update _user_facing_error:

    def _user_facing_error(self, error: Optional[Exception]) -> str:
        """Maps technical errors to user-friendly messages."""
        if error is None:
            return "⚠️ Unknown error occurred."

        if _is_rate_limit(error):
            return (
                "⚠️ **API quota reached.** The free tier limit was hit.\n\n"
                "Your source files are shown below. "
                "Wait 60 seconds and try again."
            )

        if isinstance(error, google_exceptions.NotFound):
            return (
                "⚠️ AI model unavailable. "
                "The system will automatically use a backup model next time."
            )

        if isinstance(error, google_exceptions.PermissionDenied):
            return (
                "⚠️ API key issue. "
                "Please check your GEMINI_API_KEY in the .env file."
            )

        msg = str(error).lower()

        if "timeout" in msg or "deadline" in msg:
            return (
                "⚠️ The AI took too long to respond. "
                "Try a more specific question, or ask about a single file/function."
            )

        if "safety" in msg or "blocked" in msg:
            return (
                "⚠️ Response was filtered. "
                "Try rephrasing your question without sensitive keywords."
            )

        if "500" in msg or "503" in msg or "server" in msg:
            return (
                "⚠️ AI service is temporarily unavailable. "
                "Please try again in a moment."
            )

        if "network" in msg or "connection" in msg:
            return (
                "⚠️ Network error connecting to AI service. "
                "Check your internet connection."
            )

        return (
            "⚠️ Something went wrong with the AI response. "
            "Your source files are shown below — try a different question."
        )

    def _log_message(self, error: Optional[Exception]) -> str:
        """Concise error string for server logs."""
        if error is None:
            return "Unknown error"
        if _is_rate_limit(error):
            return "Rate limit exceeded (429 ResourceExhausted)"
        if isinstance(error, google_exceptions.NotFound):
            return f"Model not found (404): {str(error)[:100]}"
        if isinstance(error, google_exceptions.PermissionDenied):
            return "Permission denied — check API key"
        msg = str(error).lower()
        if "timeout" in msg or "deadline" in msg:
            return "Request timeout"
        if "500" in msg or "503" in msg:
            return "Gemini server error (temporary)"
        return f"{type(error).__name__}: {str(error)[:100]}"

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _handle_error(self, error: Exception) -> str:
        """Alias for backwards compatibility with tests."""
        return self._log_message(error)

    def test_connection(self) -> bool:
        """Tests that the API key and model are valid."""
        try:
            self._configure()
            model = genai.GenerativeModel(self.model_name)
            r     = model.generate_content("Reply with OK only.")
            return bool(r.text)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_stats(self) -> dict:
        return {
            "model":        self.model_name,
            "fallback":     FALLBACK_MODEL,
            "temperature":  self.temperature,
            "max_tokens":   self.max_tokens,
            "configured":   self._configured,
            "retry_delays": RETRY_DELAYS,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

gemini_client = GeminiClient(
    model_name  = getattr(settings, "gemini_model", PRIMARY_MODEL),
    temperature = DEFAULT_TEMPERATURE,
    max_tokens  = DEFAULT_MAX_TOKENS,
)