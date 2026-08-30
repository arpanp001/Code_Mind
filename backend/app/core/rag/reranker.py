# backend/app/core/rag/reranker.py
#
# IMPROVEMENT 1: Enhanced Re-Ranker
#
# What changed vs original:
#   - Path-aware scoring integrated (src/ > docs/ > tests/)
#   - Exact identifier match detection (function names, class names)
#   - Implementation files ranked before documentation automatically
#   - Score breakdown stored on RankedChunk for transparency
#   - All 5 signals merged into one weighted final score

import re
from dataclasses import dataclass

from app.core.rag.fusion      import FusedResult
from app.core.rag.hybrid_search import (
    score_file_path,
    score_exact_identifier_match,
    tokenize_code,
    CHUNK_TYPE_SCORES,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Scoring weights ───────────────────────────────────────────────────────────
# These weights are applied AFTER hybrid_search already incorporated
# semantic + BM25 + path + exact_id + chunk_type.
# The reranker's job is a second pass using multi-query appearance info
# and the fused RRF score as additional signals.

WEIGHT_SIMILARITY   = 0.55   # Best similarity from any query
WEIGHT_CHUNK_TYPE   = 0.15   # Function/class > block
WEIGHT_NAME_MATCH   = 0.15   # Keyword match in function/class name
WEIGHT_APPEARANCES  = 0.08   # Appeared in multiple queries (RRF bonus)
WEIGHT_PATH         = 0.07   # Path quality (src/ > docs/)

assert abs(
    WEIGHT_SIMILARITY + WEIGHT_CHUNK_TYPE + WEIGHT_NAME_MATCH +
    WEIGHT_APPEARANCES + WEIGHT_PATH - 1.0
) < 1e-9, "Weights must sum to 1.0"


@dataclass
class RankedChunk:
    """
    A chunk after re-ranking, with full score breakdown for transparency.

    The score breakdown helps with:
    - Debugging why a chunk ranked where it did
    - Tuning weights based on retrieval quality
    - Displaying relevance explanation in the UI
    """
    chunk:            object    # RetrievedChunk
    final_score:      float     # Weighted combined score (the ranking key)
    similarity_score: float     # Raw cosine similarity component
    type_score:       float     # Chunk type bonus
    name_score:       float     # Function/class name keyword match
    appear_score:     float     # Multi-query appearance bonus
    path_score:       float     # File path quality
    appearances:      int       # How many queries found this chunk


class ReRanker:
    """
    Re-ranks fused retrieval results using 5 signals:
    1. Semantic similarity (cosine)
    2. Chunk type quality (function > class > block)
    3. Exact name match (function/class name contains query keywords)
    4. Multi-query appearance (RRF bonus for consistent relevance)
    5. Path quality (src/ implementation files > docs/ > tests/)

    The key improvement over the original:
    Implementation files always rank above documentation for the same
    semantic similarity — "src/auth/login.py" beats "docs/auth.md"
    even if both have similar cosine scores.
    """

    def rerank(
        self,
        fused_results: list[FusedResult],
        query:         str,
        top_k:         int = 5,
    ) -> list[RankedChunk]:
        """
        Re-ranks fused results using multi-signal weighted scoring.

        Args:
            fused_results: Output from ReciprocalRankFusion.fuse()
            query:         Original user query
            top_k:         Number of results to return

        Returns:
            Re-ranked list of RankedChunk objects
        """
        if not fused_results:
            return []

        # Extract query tokens for identifier matching
        query_tokens = tokenize_code(query)

        # Normalize appearance counts for relative scoring
        max_appearances = max(r.appearances for r in fused_results)

        ranked = []
        for fused in fused_results:
            chunk = fused.chunk

            # ── Signal 1: Semantic similarity ─────────────────────────────
            sim_score = fused.best_similarity

            # ── Signal 2: Chunk type ──────────────────────────────────────
            type_score = CHUNK_TYPE_SCORES.get(chunk.chunk_type, 0.4)

            # ── Signal 3: Name match (function/class name contains keywords)
            name_score = self._compute_name_score(chunk, query_tokens)

            # ── Signal 4: Multi-query appearance ──────────────────────────
            appear_score = fused.appearances / max_appearances if max_appearances > 0 else 0.0

            # ── Signal 5: Path quality ────────────────────────────────────
            path_score = score_file_path(chunk.file_path)

            # ── Weighted combination ──────────────────────────────────────
            final_score = (
                WEIGHT_SIMILARITY  * sim_score    +
                WEIGHT_CHUNK_TYPE  * type_score   +
                WEIGHT_NAME_MATCH  * name_score   +
                WEIGHT_APPEARANCES * appear_score +
                WEIGHT_PATH        * path_score
            )
            final_score = round(min(1.0, max(0.0, final_score)), 4)

            ranked.append(RankedChunk(
                chunk            = chunk,
                final_score      = final_score,
                similarity_score = round(sim_score,    4),
                type_score       = round(type_score,   4),
                name_score       = round(name_score,   4),
                appear_score     = round(appear_score, 4),
                path_score       = round(path_score,   4),
                appearances      = fused.appearances,
            ))

        # Sort by final score descending
        ranked.sort(key=lambda r: r.final_score, reverse=True)

        if ranked:
            logger.debug(
                f"Re-ranked {len(ranked)} results. "
                f"Top: {ranked[0].chunk.file_path} "
                f"(final={ranked[0].final_score:.4f}, "
                f"sim={ranked[0].similarity_score:.4f}, "
                f"path={ranked[0].path_score:.2f})"
            )

        return ranked[:top_k]

    def _compute_name_score(self, chunk, query_tokens: list[str]) -> float:
        """
        Scores how well the chunk's function/class name matches query keywords.

        1.0 → All query keywords appear in the function/class name
        0.5 → Some query keywords appear in the function/class name
        0.3 → Keywords appear as substring in the name
        0.0 → No match
        """
        if not query_tokens:
            return 0.0

        name_raw = (chunk.function_name or chunk.class_name or "").strip()
        if not name_raw:
            return 0.0

        name_lower = name_raw.lower()

        # Split name into component words (camelCase + snake_case)
        name_words = set(
            re.findall(
                r'[a-z]+',
                re.sub(r'([a-z])([A-Z])', r'\1 \2', name_lower)
            )
        )

        # Stop words that shouldn't count as matches
        STOP = {
            "get", "set", "is", "has", "the", "a", "an", "to", "of",
            "for", "in", "on", "by", "do", "be", "it", "or", "and",
        }
        meaningful_tokens = [t for t in query_tokens if t not in STOP and len(t) > 2]

        if not meaningful_tokens:
            return 0.0

        # Count matches in name component words
        exact_matches = sum(1 for t in meaningful_tokens if t in name_words)
        if exact_matches > 0:
            return min(1.0, exact_matches / len(meaningful_tokens))

        # Count substring matches
        sub_matches = sum(1 for t in meaningful_tokens if t in name_lower)
        if sub_matches > 0:
            return min(0.5, 0.5 * sub_matches / len(meaningful_tokens))

        return 0.0

    def _extract_keywords(self, query: str) -> set[str]:
        """
        Extracts meaningful keywords from a query.
        Kept for backward compatibility with existing tests.
        """
        STOP_WORDS = {
            "where", "what", "how", "which", "who", "when", "why",
            "is", "are", "does", "do", "the", "a", "an", "in", "at",
            "to", "for", "of", "and", "or", "it", "this", "that",
            "with", "by", "from", "on", "be", "was", "were", "been",
            "implemented", "implementation", "function", "method",
            "code", "file", "located", "found", "written",
        }
        words    = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', query.lower())
        keywords = {w for w in words if w not in STOP_WORDS and len(w) > 2}
        return keywords