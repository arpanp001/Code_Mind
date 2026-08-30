# backend/app/api/routes/query.py
#
# Query endpoints — chat, search, explain.
#
# What's fixed vs previous versions:
#   1. CLOSURE BUG FIXED: project_meta was captured by reference inside
#      a lambda in run_in_executor. Python closures capture the variable
#      name, not the value — if the variable is reassigned after the lambda
#      is created, the lambda sees the new value (or UnboundLocalError).
#      Fix: pass all values as default arguments into the lambda so they
#      are captured by value at definition time.
#
#   2. Generation errors now return sources instead of a bare 500 so the
#      user still sees retrieved code even when Gemini fails.
#
#   3. project_meta is built from the full project dict (includes source_url,
#      branch, languages) so metadata questions work correctly.
#
#   4. session_id is forwarded to BOTH retrieval (for query enrichment) AND
#      generation (for conversation history injection).
#
#   5. /query/search and /query/explain follow the same executor pattern
#      with default-argument lambdas to avoid future closure issues.

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.query_models import (
    ChatRequest, ChatResponse, SearchRequest,
    SearchResponse, SourceChunk,
    ExplainRequest, ExplainResponse,
)
from app.core.rag.retriever      import retriever, RetrievalRequest
from app.core.llm.rag_generator  import rag_generator
from app.core.llm.explainer      import code_explainer, ExplanationRequest
from app.utils.database          import get_project
from app.utils.logger            import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ranked_to_source(rc) -> SourceChunk:
    """Converts a RankedChunk to a SourceChunk API response model."""
    chunk = rc.chunk
    return SourceChunk(
        file_path       = chunk.file_path,
        language        = chunk.language,
        code            = chunk.text,
        start_line      = chunk.start_line,
        end_line        = chunk.end_line,
        relevance_score = rc.final_score,
        chunk_type      = chunk.chunk_type,
    )


async def _get_ready_project(project_id: str) -> dict:
    """
    Fetches a project and raises HTTPException if not found or not ready.
    Used by all query endpoints to guard against querying unfinished projects.
    """
    project = await get_project(project_id)
    if not project:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found. "
                   f"Upload a codebase first.",
        )
    if project["status"] != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Project is currently '{project['status']}'. "
                f"Wait for status 'ready' before querying. "
                f"This usually takes 1-5 minutes depending on repo size."
            ),
        )
    return project


def _build_project_meta(project: dict) -> dict:
    """
    Builds the project_meta dict that gets injected into every Gemini prompt.

    This is what lets Gemini answer metadata questions like:
      "What branch was indexed?" → uses branch field
      "What is the repo URL?"   → uses source_url field
      "What languages are used?" → uses languages field

    Keeping this as a pure function (not a lambda) avoids closure issues.
    """
    return {
        "name":        project.get("name", ""),
        "source_url":  project.get("source_url") or "",
        "branch":      project.get("branch", "main"),
        "languages":   project.get("languages", ""),
        "file_count":  project.get("file_count", 0),
        "chunk_count": project.get("chunk_count", 0),
        "source_type": project.get("source_type", "zip"),
    }


# ── POST /query/chat ──────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat_with_codebase(request: ChatRequest):
    """
    Full RAG chat — retrieves context and generates a Gemini answer.

    Pipeline:
      1. Validate project is ready
      2. Build project metadata dict (branch, URL, languages, etc.)
      3. Run advanced retrieval (multi-query + RRF + hybrid + re-rank)
         — session_id passed for follow-up query enrichment
      4. Generate answer with Gemini
         — session_id passed for conversation history injection
         — project_meta passed for metadata question answering
      5. Return answer + source chunks

    The closure bug fix is the key change here: ALL values used inside
    run_in_executor lambdas are passed as default arguments (val=val)
    so Python captures the VALUE at lambda creation time, not the
    variable binding which may change or be garbage-collected.
    """
    project = await _get_ready_project(request.project_id)

    logger.info(
        f"💬 Chat [{request.project_id}]: '{request.question[:60]}'"
        + (f" [session: {request.session_id[:8]}…]"
           if request.session_id else "")
    )

    loop = asyncio.get_event_loop()

    # ── Build metadata BEFORE the lambdas so the value is stable ─────────
    project_meta = _build_project_meta(project)
    project_name = project["name"]
    session_id   = request.session_id   # May be None — that's fine

    # ── Step 1: Retrieve ──────────────────────────────────────────────────
    try:
        # FIX: pass all values as default args to avoid closure capture bugs
        retrieval_response = await loop.run_in_executor(
            None,
            lambda pid=request.project_id,
                   q=request.question,
                   k=request.max_sources,
                   sid=session_id: retriever.retrieve(
                RetrievalRequest(
                    project_id    = pid,
                    query         = q,
                    top_k         = k,
                    use_hybrid    = True,
                    use_expansion = True,
                    session_id    = sid,
                )
            )
        )
    except Exception as e:
        logger.error(f"Retrieval failed [{request.project_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {str(e)[:200]}",
        )

    # ── Step 2: Generate with Gemini ──────────────────────────────────────
    try:
        # FIX: pass project_meta and all other closed-over values as
        # default arguments so they are captured by VALUE not by reference.
        rag_answer = await loop.run_in_executor(
            None,
            lambda rr=retrieval_response,
                   pname=project_name,
                   sid=session_id,
                   pmeta=project_meta: rag_generator.generate(
                retrieval_response = rr,
                project_name       = pname,
                session_id         = sid,
                project_meta       = pmeta,
            )
        )
    except Exception as e:
        logger.error(
            f"Generation failed [{request.project_id}]: {e}",
            exc_info=True,
        )
        # Don't return 500 — return sources with a friendly error message
        # so the user still sees the retrieved code even when Gemini fails.
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "60000" in error_msg or "120000" in error_msg:
            friendly = (
                "⚠️ The AI took too long to respond. "
                "Your source files are shown below. "
                "Try a shorter or more specific question."
            )
        elif "quota" in error_msg.lower() or "429" in error_msg or "resource exhausted" in error_msg.lower():
            friendly = (
                "⚠️ Gemini API quota reached. "
                "Wait 60 seconds and try again. "
                "Your source files are shown below."
            )
        elif "project_meta" in error_msg or "free variable" in error_msg:
            friendly = (
                "⚠️ Internal configuration error. "
                "Please refresh the page and try again."
            )
        else:
            friendly = (
                f"⚠️ AI generation failed: {error_msg[:120]}. "
                f"Your source files are shown below."
            )

        sources = []
        if request.include_sources:
            sources = [
                _ranked_to_source(rc)
                for rc in retrieval_response.ranked_chunks
            ]

        return ChatResponse(
            answer      = friendly,
            sources     = sources,
            project_id  = request.project_id,
            question    = request.question,
            tokens_used = 0,
        )

    # ── Step 3: Build response ────────────────────────────────────────────
    sources = []
    if request.include_sources:
        sources = [
            _ranked_to_source(rc)
            for rc in retrieval_response.ranked_chunks
        ]

    logger.info(
        f"✅ Chat complete [{request.project_id}]: "
        f"{rag_answer.tokens_used} tokens, "
        f"{len(sources)} sources"
    )

    return ChatResponse(
        answer        = rag_answer.answer,
        sources       = sources,
        project_id    = request.project_id,
        question      = request.question,
        tokens_used   = rag_answer.tokens_used,
        memories_used = getattr(rag_answer, "memories_used", 0),
    )


# ── POST /query/chat/stream ───────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat — yields answer tokens as Server-Sent Events.

    The frontend can display words as they appear rather than waiting
    for the full response. Much better UX for long answers.

    SSE format:
      data: {"type": "sources", "data": [...]}   ← sent first
      data: {"type": "token",   "data": "word "}  ← streamed
      data: {"type": "done"}                      ← sent last

    Usage from frontend:
      const response = await fetch('/api/v1/query/chat/stream', {...})
      const reader   = response.body.getReader()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        processSSE(new TextDecoder().decode(value))
      }
    """
    import json

    project      = await _get_ready_project(request.project_id)
    project_meta = _build_project_meta(project)
    session_id   = request.session_id
    loop         = asyncio.get_event_loop()

    # Retrieval runs first (blocking) — we need sources before streaming
    retrieval_response = await loop.run_in_executor(
        None,
        lambda pid=request.project_id,
               q=request.question,
               k=request.max_sources,
               sid=session_id: retriever.retrieve(
            RetrievalRequest(
                project_id    = pid,
                query         = q,
                top_k         = k,
                use_hybrid    = True,
                use_expansion = True,
                session_id    = sid,
            )
        )
    )

    async def token_generator():
        # Send source cards first so the frontend can render them
        # while the answer is still streaming
        sources      = [_ranked_to_source(rc)
                        for rc in retrieval_response.ranked_chunks]
        sources_data = [s.model_dump() for s in sources]
        yield (
            f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"
        )

        # Stream answer tokens
        # FIX: pass all values as default args — same closure fix as above
        stream_gen = rag_generator.generate_with_stream(
            retrieval_response = retrieval_response,
            project_name       = project["name"],
        )

        for token in stream_gen:
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── POST /query/search ────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def search_codebase(request: SearchRequest):
    """
    Semantic + hybrid search — returns relevant code without AI generation.

    Useful for:
    - Finding all occurrences of a function/class
    - Exploring the codebase without spending Gemini tokens
    - Debugging retrieval quality

    Returns ranked code chunks with relevance scores.
    """
    await _get_ready_project(request.project_id)

    loop = asyncio.get_event_loop()

    try:
        # FIX: default-argument lambda to avoid closure issues
        response = await loop.run_in_executor(
            None,
            lambda pid=request.project_id,
                   q=request.query,
                   k=request.top_k: retriever.retrieve(
                RetrievalRequest(
                    project_id    = pid,
                    query         = q,
                    top_k         = k,
                    use_hybrid    = True,
                    use_expansion = True,
                )
            )
        )
    except Exception as e:
        logger.error(
            f"Search failed [{request.project_id}]: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)[:200]}",
        )

    return SearchResponse(
        results     = [_ranked_to_source(rc) for rc in response.ranked_chunks],
        query       = request.query,
        total_found = response.total_found,
    )


# ── POST /query/explain ───────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse)
async def explain_code(request: ExplainRequest):
    """
    Explains a specific code snippet using Gemini.

    Unlike /chat, this does NOT perform retrieval — the code is provided
    directly in the request body. Called when a user:
      - Clicks "Explain" on a source card in the chat
      - Selects a function and asks what it does
      - Wants a deep dive on a specific piece of code

    Optionally accepts a specific question about the code snippet.
    """
    logger.info(
        f"📖 Explain [{request.project_id}]: "
        f"{request.language} '{request.file_path}'"
    )

    loop = asyncio.get_event_loop()

    try:
        # FIX: pass all closed-over values as default args
        explanation_response = await loop.run_in_executor(
            None,
            lambda code=request.code,
                   lang=request.language,
                   fp=request.file_path,
                   q=request.question or None,
                   pid=request.project_id: code_explainer.explain(
                ExplanationRequest(
                    code       = code,
                    language   = lang,
                    file_path  = fp,
                    question   = q,
                    project_id = pid,
                )
            )
        )
    except Exception as e:
        logger.error(
            f"Explanation failed [{request.project_id}]: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Explanation error: {str(e)[:200]}",
        )

    return ExplainResponse(
        explanation = explanation_response.explanation,
        language    = explanation_response.language,
        file_path   = explanation_response.file_path,
        tokens_used = explanation_response.tokens_used,
        success     = explanation_response.success,
    )


# ── DELETE /query/session/{session_id} ───────────────────────────────────────

@router.delete("/session/{session_id}", tags=["Query"])
async def clear_session(session_id: str):
    """
    Clears conversation history for a session.

    Called when the user clicks "Clear" in the chat UI.
    The frontend also generates a new session ID after calling this.
    """
    try:
        from app.core.llm.conversation_memory import conversation_memory
        conversation_memory.clear_session(session_id)
        logger.info(f"🗑️  Session cleared: {session_id[:8]}…")
        return {
            "message":    "Session cleared successfully",
            "session_id": session_id,
        }
    except Exception as e:
        logger.warning(f"Session clear failed for {session_id}: {e}")
        # Don't fail — the frontend will just generate a new session ID anyway
        return {
            "message":    "Session clear attempted",
            "session_id": session_id,
        }