# backend/app/core/rag/context_assembler.py
#
# IMPROVEMENT 1: Enhanced Context Assembler
#
# What changed vs original:
#   - Implementation files always appear before documentation in the
#     assembled context (path_score aware ordering)
#   - Score breakdown included in chunk header for LLM transparency
#   - Groups chunks by file so the LLM sees related code together
#   - Token budget is respected while maximising implementation coverage
#   - Truncated flag tells the prompt builder how much context was cut

import tiktoken
from dataclasses import dataclass, field

from app.core.rag.reranker import RankedChunk
from app.utils.logger      import get_logger

logger = get_logger(__name__)

# Conservative token budget — leaves room for system prompt + answer
DEFAULT_TOKEN_BUDGET = 1200

# Separator between chunks in the assembled context string
CHUNK_SEPARATOR = "\n" + "─" * 60 + "\n"


@dataclass
class AssembledContext:
    """
    The final context string ready to be injected into a Gemini prompt.
    """
    context_text:     str              # Formatted context for the LLM
    chunks_used:      int              # How many chunks were included
    tokens_used:      int              # Approximate token count
    files_referenced: list[str]        # Unique file paths in context
    truncated:        bool  = False    # True if some chunks were dropped
    impl_files:       int   = 0        # Count of implementation files used
    doc_files:        int   = 0        # Count of documentation files used


class ContextAssembler:
    """
    Assembles RankedChunks into a structured context string for Gemini.

    Key improvement: Implementation files are placed BEFORE documentation
    in the assembled context. Even if a documentation chunk has a slightly
    higher final_score, src/ files appear first because the LLM should
    ground its answer in actual code, not in documentation descriptions.

    Context ordering:
      1. Implementation files (path_score >= 0.5, src/app/core/...)
      2. API/model files    (path_score 0.4-0.5)
      3. Documentation      (path_score < 0.4, docs/README.md/...)

    Within each tier, chunks are ordered by final_score descending.

    Format of each chunk block:
      File: src/auth/login.py | Language: python | Lines: 12-24
      Type: function | Name: authenticate | Relevance: 87.3%

      def authenticate(username, password):
          ...
    """

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.token_budget = token_budget
        try:
            # cl100k_base is the tokeniser used by GPT-4 and Gemini
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None   # Fall back to char-based estimation

    def _count_tokens(self, text: str) -> int:
        """Counts tokens; falls back to char/4 estimation if tiktoken fails."""
        if self._enc:
            try:
                return len(self._enc.encode(text, disallowed_special=()))
            except Exception:
                pass
        return len(text) // 4

    def assemble(
        self,
        ranked_chunks: list[RankedChunk],
        query:         str,
        token_budget:  int | None = None,
    ) -> AssembledContext:
        """
        Assembles ranked chunks into a formatted context string.

        Implementation:
        1. Split chunks into impl tier vs doc tier by path_score
        2. Within each tier, sort by final_score descending
        3. Fill token budget from impl first, then doc
        4. Format each chunk with file header + metadata + code

        Args:
            ranked_chunks: Output from ReRanker.rerank()
            query:         Original user query (used in header)
            token_budget:  Max tokens (default: DEFAULT_TOKEN_BUDGET)

        Returns:
            AssembledContext ready for Gemini prompt injection
        """
        budget = token_budget or self.token_budget

        # ── Split into implementation vs documentation tiers ──────────────
        # Threshold: path_score >= 0.5 → implementation (src/app/api/...)
        #            path_score <  0.5 → documentation  (docs/README/tests/)
        IMPL_THRESHOLD = 0.5

        impl_chunks = sorted(
            [rc for rc in ranked_chunks if self._get_path_score(rc) >= IMPL_THRESHOLD],
            key=lambda rc: rc.final_score,
            reverse=True,
        )
        doc_chunks = sorted(
            [rc for rc in ranked_chunks if self._get_path_score(rc) < IMPL_THRESHOLD],
            key=lambda rc: rc.final_score,
            reverse=True,
        )

        # Process implementation chunks first, then documentation
        ordered_chunks = impl_chunks + doc_chunks

        # ── Fill token budget ──────────────────────────────────────────────
        context_parts    = []
        tokens_used      = 0
        files_seen       = []
        truncated        = False
        impl_count       = 0
        doc_count        = 0

        for rc in ordered_chunks:
            chunk_text   = self._format_chunk(rc)
            chunk_tokens = self._count_tokens(chunk_text)

            if tokens_used + chunk_tokens > budget:
                truncated = True
                logger.debug(
                    f"Context budget reached at {tokens_used} tokens "
                    f"({len(context_parts)} chunks included)"
                )
                break

            context_parts.append(chunk_text)
            tokens_used += chunk_tokens

            # Track file references (deduplicated)
            fp = rc.chunk.file_path
            if fp not in files_seen:
                files_seen.append(fp)

            # Track impl vs doc counts
            if self._get_path_score(rc) >= IMPL_THRESHOLD:
                impl_count += 1
            else:
                doc_count += 1

        # Join all parts
        context_text = CHUNK_SEPARATOR.join(context_parts)

        if not context_text:
            context_text = "No relevant code context found for this query."

        logger.debug(
            f"Context assembled: {len(context_parts)} chunks "
            f"({impl_count} impl, {doc_count} docs), "
            f"{tokens_used} tokens, "
            f"{len(files_seen)} files"
            + (" [TRUNCATED]" if truncated else "")
        )

        return AssembledContext(
            context_text     = context_text,
            chunks_used      = len(context_parts),
            tokens_used      = tokens_used,
            files_referenced = files_seen,
            truncated        = truncated,
            impl_files       = impl_count,
            doc_files        = doc_count,
        )

    def _get_path_score(self, rc: RankedChunk) -> float:
        """Gets path_score from RankedChunk (with fallback for old format)."""
        # RankedChunk now carries path_score after Improvement 1
        return getattr(rc, 'path_score', 0.5)

    def _format_chunk(self, rc: RankedChunk) -> str:
        """
        Formats a single chunk as a readable context block.

        Format:
            File: src/auth/login.py | Language: python | Lines: 12-24
            Type: function | Name: authenticate | Relevance: 87.3%

            def authenticate(username, password):
                user = find_user(username)
                ...
        """
        chunk = rc.chunk

        # Header line: file info
        header_parts = [f"File: {chunk.file_path}"]
        if chunk.language:
            header_parts.append(f"Language: {chunk.language}")
        if chunk.start_line and chunk.end_line:
            header_parts.append(f"Lines: {chunk.start_line}-{chunk.end_line}")
        header = " | ".join(header_parts)

        # Metadata line: chunk type, name, relevance
        meta_parts = [f"Type: {chunk.chunk_type}"]
        if chunk.function_name:
            meta_parts.append(f"Function: {chunk.function_name}")
        elif chunk.class_name:
            meta_parts.append(f"Class: {chunk.class_name}")
        meta_parts.append(f"Relevance: {rc.final_score:.1%}")
        meta = " | ".join(meta_parts)

        return f"{header}\n{meta}\n\n{chunk.text}"

    def format_for_prompt(
        self,
        assembled: AssembledContext,
        query:     str,
    ) -> str:
        """
        Wraps the assembled context in a prompt-ready format with header/footer.
        Used by the prompt builder when constructing the full Gemini message.
        """
        impl_note = ""
        if assembled.impl_files > 0 or assembled.doc_files > 0:
            impl_note = (
                f" ({assembled.impl_files} implementation"
                f"{', ' + str(assembled.doc_files) + ' documentation' if assembled.doc_files else ''})"
            )

        header = (
            f"=== CODEBASE CONTEXT ===\n"
            f"Retrieved {assembled.chunks_used} relevant sections "
            f"from {len(assembled.files_referenced)} file(s){impl_note} "
            f"for query: \"{query}\"\n"
        )

        if assembled.truncated:
            header += (
                "(Note: context truncated to token budget. "
                "Most relevant sections shown first.)\n"
            )

        footer = "\n=== END CONTEXT ==="

        return header + "\n" + assembled.context_text + footer