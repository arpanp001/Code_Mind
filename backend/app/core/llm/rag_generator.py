# backend/app/core/llm/rag_generator.py
#
# IMPROVEMENT 3 INTEGRATION: Architecture context injected into prompts
#
# What changed vs previous version:
#   - generate() now calls architecture_memory.get_architecture_context()
#     when the query is detected as an architecture question
#   - Architecture context (entry points, dependencies, flow) is prepended
#     to the code context so Gemini answers with BOTH structural knowledge
#     AND specific code citations
#   - Mermaid diagram (if generated) is included in the response metadata
#   - RAGAnswer now carries architecture_context and mermaid_diagram fields
#   - Non-architecture queries are unaffected (zero performance cost)

import re
from dataclasses import dataclass, field
from typing      import Optional

from app.core.llm.gemini              import gemini_client, GeminiResponse
from app.core.llm.prompts             import prompt_builder
from app.core.rag.context_assembler   import AssembledContext
from app.core.rag.retriever           import RetrievalResponse
from app.utils.logger                 import get_logger

logger = get_logger(__name__)


# ── Query type detection ───────────────────────────────────────────────────────

EXPLAIN_KEYWORDS = {
    "explain", "what does", "what is", "how does", "describe",
    "walk me through", "break down", "understand", "meaning of",
    "purpose of", "why does",
}

ARCHITECTURE_KEYWORDS = {
    "architecture", "design", "pattern", "structure", "why was",
    "why did", "decision", "approach", "strategy", "best practice",
    "should i", "recommend", "tradeoff", "compare",
    "entry point", "request flow", "flow", "depends on",
    "dependency", "dependencies", "how is it organized",
    "components", "modules overview",
}


def detect_query_type(query: str) -> str:
    """
    Returns one of: "explain", "architecture", "general"
    """
    query_lower = query.lower()
    if any(kw in query_lower for kw in EXPLAIN_KEYWORDS):
        return "explain"
    if any(kw in query_lower for kw in ARCHITECTURE_KEYWORDS):
        return "architecture"
    return "general"


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class RAGAnswer:
    """
    Complete answer from the RAG generator.
    Now includes optional architecture context and Mermaid diagram.
    """
    answer:               str
    query:                str
    project_id:           str
    query_type:           str        = "general"
    files_referenced:     list[str]  = field(default_factory=list)
    tokens_used:          int        = 0
    context_chunks:       int        = 0
    memories_used:        int        = 0
    success:              bool       = True
    error_message:        str        = ""
    expanded_queries:     list[str]  = field(default_factory=list)
    # Improvement 3 additions:
    architecture_context: str        = ""   # Plain-text architecture summary
    mermaid_diagram:      str        = ""   # Mermaid diagram (if applicable)


# ── RAG Generator ──────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    Generates answers by combining:
    1. Retrieved code chunks (vector search)
    2. Project memories (architecture decisions, bug fixes)
    3. Conversation history (session context)
    4. [NEW] Dependency graph context (architecture questions)

    For architecture questions, step 4 provides structural knowledge
    (entry points, import relationships, class hierarchies) that
    vector search alone cannot answer reliably.
    """

    def generate(
        self,
        retrieval_response: RetrievalResponse,
        project_name:       str              = "the project",
        session_id:         Optional[str]    = None,
        project_meta:       Optional[dict]   = None,
    ) -> RAGAnswer:
        """
        Generates a complete RAG answer.

        Pipeline:
        1. Detect query type (general / architecture / explain)
        2. Get conversation history (if session_id provided)
        3. Search project memories for relevant context
        4. [NEW] If architecture question → query dependency graph
        5. Assemble code context from retrieved chunks
        6. Combine all context sources into one prompt
        7. Call Gemini
        8. Save exchange to conversation history
        """
        from app.core.memory.project_memory    import project_memory
        from app.core.llm.conversation_memory  import conversation_memory
        # Improvement 3: architecture context
        from app.core.analysis.architecture_memory import architecture_memory

        query      = retrieval_response.query
        project_id = retrieval_response.project_id

        logger.info(
            f"🤖 Generating answer [{project_id}]: '{query[:60]}'"
        )

        query_type = detect_query_type(query)
        logger.debug(f"  Query type: {query_type}")

        # ── Step 1: Conversation history ──────────────────────────────────
        history_text = ""
        session      = None
        if session_id:
            try:
                session      = conversation_memory.get_or_create(session_id, project_id)
                history_text = session.get_history_text()
            except Exception as e:
                logger.warning(f"History retrieval failed: {e}")

        # ── Step 2: No code context case ──────────────────────────────────
        if not retrieval_response.ranked_chunks:
            # Even with no code context, try architecture context
            arch_ctx = None
            try:
                arch_ctx = architecture_memory.get_architecture_context(
                    project_id, query
                )
            except Exception as e:
                logger.warning(f"Architecture context failed: {e}")

            if arch_ctx and arch_ctx.answered:
                # Architecture question with graph available — answer from graph
                arch_prompt = prompt_builder.build_architecture_prompt(
                    question = query,
                    context  = arch_ctx.context_text,
                )
                gemini_resp = gemini_client.generate(arch_prompt)
                if session:
                    session.add_exchange(query, gemini_resp.text)
                return RAGAnswer(
                    answer               = gemini_resp.text,
                    query                = query,
                    project_id           = project_id,
                    query_type           = "architecture",
                    tokens_used          = gemini_resp.total_tokens,
                    success              = gemini_resp.success,
                    error_message        = gemini_resp.error_message,
                    expanded_queries     = retrieval_response.expanded_queries,
                    architecture_context = arch_ctx.context_text,
                    mermaid_diagram      = arch_ctx.mermaid_diagram,
                )

            # Pure no-context fallback
            prompt      = prompt_builder.build_no_context_prompt(query)
            gemini_resp = gemini_client.generate(prompt)
            if session:
                session.add_exchange(query, gemini_resp.text)
            return RAGAnswer(
                answer           = gemini_resp.text,
                query            = query,
                project_id       = project_id,
                query_type       = query_type,
                tokens_used      = gemini_resp.total_tokens,
                success          = gemini_resp.success,
                error_message    = gemini_resp.error_message,
                expanded_queries = retrieval_response.expanded_queries,
            )

        # ── Step 3: Assemble code context ─────────────────────────────────
        context = retrieval_response.context
        if not context:
            from app.core.rag.context_assembler import ContextAssembler
            assembler = ContextAssembler()
            context   = assembler.assemble(
                retrieval_response.ranked_chunks, query
            )

        context_text = context.context_text

        # ── Step 4: Memory injection ──────────────────────────────────────
        memory_text   = ""
        memories_used = 0
        try:
            memory_results = project_memory.search_memories(
                project_id = project_id,
                query      = query,
                top_k      = 3,
            )
            if memory_results:
                memory_lines = ["\n--- Project Memory ---"]
                for mr in memory_results:
                    type_label = mr.memory.memory_type.replace("_", " ").title()
                    title      = f" ({mr.memory.title})" if mr.memory.title else ""
                    memory_lines.append(
                        f"[{type_label}{title}]: {mr.memory.content}"
                    )
                memory_text   = "\n".join(memory_lines)
                memories_used = len(memory_results)
        except Exception as e:
            logger.warning(f"Memory injection failed: {e}")

        # ── Step 5: Architecture context injection (Improvement 3) ────────
        arch_context_text = ""
        mermaid_diagram   = ""
        try:
            arch_ctx = architecture_memory.get_architecture_context(
                project_id, query
            )
            if arch_ctx and arch_ctx.answered:
                arch_context_text = arch_ctx.context_text
                mermaid_diagram   = arch_ctx.mermaid_diagram
                logger.info(
                    f"🏗️  Architecture context injected "
                    f"(type={arch_ctx.query_type})"
                )
        except Exception as e:
            logger.warning(f"Architecture context injection failed: {e}")

        # ── Step 6: Build full context string ─────────────────────────────
        full_context_parts = [context_text]
        if memory_text:
            full_context_parts.append(memory_text)
        if arch_context_text:
            full_context_parts.append(
                f"\n--- Architecture Knowledge ---\n{arch_context_text}"
            )
        full_context = "\n".join(full_context_parts)

        # ── Step 7: Build prompt ───────────────────────────────────────────
        # Use project_meta safely via default-argument capture
        pmeta = project_meta or {}

        if query_type == "architecture":
            prompt = prompt_builder.build_architecture_prompt(
                question = query,
                context  = full_context,
            )
        elif history_text:
            prompt = prompt_builder.build_chat_prompt_with_history(
                question     = query,
                context      = full_context,
                history      = history_text,
                project_name = project_name,
            )
        else:
            prompt = prompt_builder.build_chat_prompt(
                question     = query,
                context      = full_context,
                project_name = project_name,
                project_meta = pmeta,
            )

        # ── Step 8: Call Gemini ────────────────────────────────────────────
        gemini_resp = gemini_client.generate(prompt)

        # ── Step 9: Save to conversation history ──────────────────────────
        if session:
            session.add_exchange(query, gemini_resp.text)

        return RAGAnswer(
            answer               = gemini_resp.text,
            query                = query,
            project_id           = project_id,
            query_type           = query_type,
            files_referenced     = context.files_referenced if context else [],
            tokens_used          = gemini_resp.total_tokens,
            context_chunks       = context.chunks_used if context else 0,
            memories_used        = memories_used,
            success              = gemini_resp.success,
            error_message        = gemini_resp.error_message,
            expanded_queries     = retrieval_response.expanded_queries,
            architecture_context = arch_context_text,
            mermaid_diagram      = mermaid_diagram,
        )

    def generate_with_stream(
        self,
        retrieval_response: RetrievalResponse,
        project_name:       str = "the project",
    ):
        """Streaming version — yields text chunks. Architecture context not streamed."""
        query      = retrieval_response.query
        query_type = detect_query_type(query)

        if not retrieval_response.ranked_chunks:
            prompt = prompt_builder.build_no_context_prompt(query)
        else:
            context = retrieval_response.context
            context_text = context.context_text if context else ""
            prompt = prompt_builder.build_chat_prompt(
                question     = query,
                context      = context_text,
                project_name = project_name,
            )

        yield from gemini_client.generate_stream(prompt)


# Module-level singleton
rag_generator = RAGGenerator()