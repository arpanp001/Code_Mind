# backend/app/core/rag/retriever.py
#
# IMPROVEMENT 1: Enhanced Advanced Retriever
#
# What changed vs original:
#   - Passes session_id through for conversation-aware query enrichment
#   - Vague follow-up queries ("explain that") enriched from conversation history
#   - Fallback keyword extraction when primary retrieval returns nothing
#   - Hybrid search now uses enhanced HybridSearcher (5 signals)
#   - Reranker now includes path_score in RankedChunk
#   - Context assembler sees full 5-signal ranked results

from dataclasses import dataclass, field
from typing      import Optional

from app.core.rag.embedder           import embedding_engine
from app.core.rag.vectorstore        import chroma_client, RetrievedChunk
from app.core.rag.query_expander     import QueryExpander
from app.core.rag.fusion             import ReciprocalRankFusion, FusedResult
from app.core.rag.reranker           import ReRanker, RankedChunk
from app.core.rag.hybrid_search      import HybridSearcher, tokenize_code
from app.core.rag.context_assembler  import ContextAssembler, AssembledContext
from app.utils.logger                import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalRequest:
    """Parameters for a retrieval operation."""
    project_id:    str
    query:         str
    top_k:         int           = 5
    language:      Optional[str] = None
    chunk_type:    Optional[str] = None
    file_path:     Optional[str] = None
    use_hybrid:    bool          = True
    use_expansion: bool          = True
    session_id:    Optional[str] = None   # For conversation-aware enrichment


@dataclass
class RetrievalResponse:
    """
    Complete response from the advanced retrieval pipeline.
    Contains ranked chunks AND the assembled context for Gemini.
    """
    query:            str
    project_id:       str
    ranked_chunks:    list[RankedChunk]         = field(default_factory=list)
    context:          Optional[AssembledContext] = None
    expanded_queries: list[str]                 = field(default_factory=list)
    total_found:      int                       = 0
    search_time_ms:   float                     = 0.0


class AdvancedRetriever:
    """
    Full production retrieval pipeline with Improvement 1 integrated.

    Pipeline:
      query
        → [optional] conversation-aware enrichment (vague follow-ups)
        → QueryExpander      (3-4 query variations)
        → ChromaDB search    (each variation)
        → RRF fusion         (merge + deduplicate)
        → HybridSearcher     (5-signal re-scoring: sem+BM25+path+id+type)
        → ReRanker           (5-signal weighted final score)
        → ContextAssembler   (token-budget-aware context string)
        → RetrievalResponse
    """

    def __init__(self):
        self.expander  = QueryExpander(max_variations=3)
        self.fusion    = ReciprocalRankFusion()
        self.reranker  = ReRanker()
        self.hybrid    = HybridSearcher()
        self.assembler = ContextAssembler()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Runs the full retrieval pipeline synchronously.
        Call via run_in_executor() from async route handlers.
        """
        import time
        start = time.time()

        if not request.query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(
            f"🔍 Advanced retrieval [{request.project_id}]: "
            f"'{request.query[:60]}'"
        )

        # ── Step 1: Enrich vague follow-up queries ────────────────────────
        # "Explain that function" → "Explain authenticate function in src/auth"
        effective_query = self._enrich_query(request)
        if effective_query != request.query:
            logger.debug(f"  Query enriched: '{effective_query[:60]}'")

        # ── Step 2: Expand the (possibly enriched) query ──────────────────
        expanded    = self.expander.expand(effective_query)
        all_queries = expanded.all_queries if request.use_expansion else [effective_query]

        logger.debug(f"  Queries: {all_queries}")

        # ── Step 3: Build metadata filters ───────────────────────────────
        filters = self._build_filters(request)

        # ── Step 4: Search ChromaDB with each query variation ─────────────
        search_top_k  = min(request.top_k * 2, 8)
        result_lists  : list[list[RetrievedChunk]] = []

        for q in all_queries:
            try:
                query_embedding = embedding_engine.embed_query(q)
                result = chroma_client.search(
                    project_id      = request.project_id,
                    query_embedding = query_embedding,
                    top_k           = search_top_k,
                    filters         = filters,
                )
                if result.chunks:
                    result_lists.append(result.chunks)
            except Exception as e:
                logger.warning(f"  Query '{q[:30]}' search failed: {e}")

        # ── Fallback: if primary retrieval found nothing, try keyword search
        if not result_lists:
            logger.warning(
                f"No results from any query [{request.project_id}] — "
                f"trying keyword fallback"
            )
            result_lists = self._keyword_fallback(request)

        if not result_lists:
            return RetrievalResponse(
                query      = request.query,
                project_id = request.project_id,
            )

        # ── Step 5: RRF fusion ────────────────────────────────────────────
        fused = self.fusion.fuse(result_lists, top_k=request.top_k * 2)

        # ── Step 6: Hybrid re-scoring (5 signals) ─────────────────────────
        if request.use_hybrid and fused:
            fused_chunks  = [f.chunk for f in fused]
            hybrid_chunks = self.hybrid.hybrid_search(
                semantic_results = fused_chunks,
                query            = effective_query,
                top_k            = request.top_k * 2,
            )

            # Rebuild FusedResult list preserving RRF metadata
            chunk_id_to_fused = {f.chunk.chunk_id: f for f in fused}
            rebuilt = []
            for hc in hybrid_chunks:
                orig = chunk_id_to_fused.get(hc.chunk_id)
                if orig:
                    rebuilt.append(FusedResult(
                        chunk           = hc,
                        rrf_score       = orig.rrf_score,
                        appearances     = orig.appearances,
                        best_similarity = hc.similarity,
                    ))
            if rebuilt:
                fused = rebuilt

        # ── Step 7: Re-rank with 5-signal scorer ──────────────────────────
        ranked = self.reranker.rerank(
            fused_results = fused,
            query         = effective_query,
            top_k         = request.top_k,
        )

        # ── Step 8: Assemble context for LLM ─────────────────────────────
        context = self.assembler.assemble(
            ranked_chunks = ranked,
            query         = effective_query,
        )

        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"✅ Retrieval complete [{request.project_id}]: "
            f"{len(ranked)} chunks, {elapsed_ms:.1f}ms, "
            f"{len(all_queries)} queries"
        )

        return RetrievalResponse(
            query            = request.query,
            project_id       = request.project_id,
            ranked_chunks    = ranked,
            context          = context,
            expanded_queries = all_queries,
            total_found      = len(ranked),
            search_time_ms   = elapsed_ms,
        )

    def _enrich_query(self, request: RetrievalRequest) -> str:
        """
        Detects vague follow-up queries and enriches them with context
        from conversation history so retrieval has concrete keywords.

        Examples:
          "Explain that"           → "Explain authenticate function src/auth.py"
          "Tell me more about it"  → "authenticate function JWT token"
          "What does it return?"   → "authenticate function return value"
          "Where is login?"        → "Where is login?" (not enriched — specific enough)
        """
        query = request.query.strip()

        import re
        VAGUE_PATTERNS = [
            r'^(explain|describe|tell me|show me|what|how|why)\s+(that|this|it|more)',
            r'^(that|this)\s+(function|method|class|code|file|module)',
            r'^(more detail|in detail|more about|elaborate)',
            r'^(what does it|what is it|how does it|what does that)',
        ]

        is_vague = any(
            re.match(p, query, re.IGNORECASE)
            for p in VAGUE_PATTERNS
        ) or (
            len(query.split()) <= 4 and
            any(w in query.lower() for w in ['it', 'that', 'this', 'there', 'them', 'those'])
        )

        if not is_vague or not request.session_id:
            return query

        try:
            from app.core.llm.conversation_memory import conversation_memory
            session  = conversation_memory.get_or_create(
                request.session_id, request.project_id
            )
            messages = list(session.messages)
            if not messages:
                return query

            # Extract context from last exchange (last 2 messages)
            context_parts = []
            for msg in messages[-2:]:
                if msg.role == 'user':
                    context_parts.append(msg.content[:80])
                elif msg.role == 'assistant':
                    # First sentence of the last AI answer
                    first_sentence = msg.content.split('.')[0][:120]
                    context_parts.append(first_sentence)

            if context_parts:
                enriched = f"{query} context: {' '.join(context_parts)}"
                return enriched

        except Exception as e:
            logger.warning(f"Query enrichment failed: {e}")

        return query

    def _keyword_fallback(
        self,
        request: RetrievalRequest,
    ) -> list[list[RetrievedChunk]]:
        """
        Fallback when primary retrieval finds nothing.
        Strips stop words and searches with core keywords only.
        """
        import re
        STOP = {
            "what", "where", "how", "which", "who", "when", "why",
            "is", "are", "was", "were", "does", "do", "did",
            "the", "a", "an", "in", "at", "to", "for", "of", "and",
            "or", "it", "this", "that", "with", "by", "from", "on",
            "show", "tell", "explain", "find", "get", "give", "list",
            "implemented", "implementation", "function", "method", "code",
        }
        words    = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', request.query.lower())
        keywords = [w for w in words if w not in STOP and len(w) > 2]

        if not keywords:
            return []

        keyword_query = ' '.join(keywords[:5])
        logger.info(f"  Fallback keyword search: '{keyword_query}'")

        try:
            fallback_embedding = embedding_engine.embed_query(keyword_query)
            result = chroma_client.search(
                project_id      = request.project_id,
                query_embedding = fallback_embedding,
                top_k           = request.top_k,
                filters         = None,   # No filters in fallback
            )
            if result.chunks:
                logger.info(
                    f"  Fallback found {len(result.chunks)} results"
                )
                return [result.chunks]
        except Exception as e:
            logger.warning(f"  Fallback search failed: {e}")

        return []

    def _build_filters(self, request: RetrievalRequest) -> Optional[dict]:
        """Builds ChromaDB where-clause from optional filter parameters."""
        conditions = []
        if request.language:   conditions.append({"language":   request.language})
        if request.chunk_type: conditions.append({"chunk_type": request.chunk_type})
        if request.file_path:  conditions.append({"file_path":  request.file_path})
        if not conditions:     return None
        if len(conditions) == 1: return conditions[0]
        return {"$and": conditions}

    def get_project_stats(self, project_id: str) -> dict:
        """Returns ChromaDB stats for a project."""
        return chroma_client.get_collection_stats(project_id)


# Module-level singleton
retriever = AdvancedRetriever()