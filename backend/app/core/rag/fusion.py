# backend/app/core/rag/fusion.py
#
# Reciprocal Rank Fusion (RRF) — merges results from multiple search queries.
#
# The problem:
#   Query 1 returns: [auth.py(0.95), login.py(0.88), jwt.py(0.72)]
#   Query 2 returns: [login.py(0.91), session.py(0.85), auth.py(0.70)]
#   Query 3 returns: [middleware.py(0.89), auth.py(0.81), login.py(0.75)]
#
# Naive merge (by score): auth.py wins because it has 0.95 highest score
# But login.py appears in ALL THREE lists → it's more robustly relevant
#
# RRF fix:
#   Each document gets score = sum(1 / (rank + k)) across all lists
#   login.py: 1/(2+60) + 1/(1+60) + 1/(2+60) = 0.048 ← highest!
#   auth.py:  1/(1+60) + 1/(3+60) + 1/(2+60) = 0.047
#
# RRF rewards documents that appear consistently across multiple queries
# even if they're not the top result in any single query.
#
# Reference: Cormack, Clarke & Buettcher (2009) — SIGIR

from dataclasses import dataclass, field
from app.core.rag.vectorstore import RetrievedChunk
from app.utils.logger         import get_logger

logger = get_logger(__name__)

# RRF smoothing constant.
# k=60 is the standard value from the original paper.
# Higher k → less weight on top-ranked results
# Lower k  → more weight on top-ranked results
RRF_K = 60


@dataclass
class FusedResult:
    """
    A chunk after RRF fusion — has a combined score across all queries.
    """
    chunk:      RetrievedChunk
    rrf_score:  float              # Combined score (higher = more relevant)
    appearances: int               # How many queries returned this chunk
    best_similarity: float         # Highest raw similarity from any query


class ReciprocalRankFusion:
    """
    Merges multiple ranked result lists into a single ranked list
    using the Reciprocal Rank Fusion algorithm.

    Usage:
        fusion  = ReciprocalRankFusion()
        results = fusion.fuse([results_q1, results_q2, results_q3])
    """

    def __init__(self, k: int = RRF_K):
        """
        Args:
            k: Smoothing constant. Default 60 from original paper.
        """
        self.k = k

    def fuse(
        self,
        result_lists: list[list[RetrievedChunk]],
        top_k:        int = 10,
    ) -> list[FusedResult]:
        """
        Fuses multiple ranked result lists into one.

        Algorithm:
        1. For each result list, assign each chunk a rank (1-indexed)
        2. Compute RRF score: sum(1 / (rank + k)) for each appearance
        3. Sort by RRF score descending
        4. Return top_k results

        Args:
            result_lists: List of result lists, one per query
            top_k:        Number of final results to return

        Returns:
            Fused and re-ranked list of FusedResult objects
        """
        if not result_lists:
            return []

        # Filter out empty lists
        non_empty = [r for r in result_lists if r]
        if not non_empty:
            return []

        # If only one list, wrap and return (no fusion needed)
        if len(non_empty) == 1:
            return [
                FusedResult(
                    chunk           = chunk,
                    rrf_score       = chunk.similarity,
                    appearances     = 1,
                    best_similarity = chunk.similarity,
                )
                for chunk in non_empty[0][:top_k]
            ]

        # ── RRF computation ───────────────────────────────────────────────
        # scores[chunk_id]         = accumulated RRF score
        # appearances[chunk_id]    = how many lists contained this chunk
        # best_sim[chunk_id]       = highest raw similarity score
        # chunk_map[chunk_id]      = the actual RetrievedChunk object

        scores      : dict[str, float]          = {}
        appearances : dict[str, int]            = {}
        best_sim    : dict[str, float]          = {}
        chunk_map   : dict[str, RetrievedChunk] = {}

        for result_list in non_empty:
            for rank, chunk in enumerate(result_list, start=1):
                cid = chunk.chunk_id

                # RRF formula: 1 / (rank + k)
                rrf_contribution = 1.0 / (rank + self.k)

                scores[cid]       = scores.get(cid, 0.0) + rrf_contribution
                appearances[cid]  = appearances.get(cid, 0) + 1
                best_sim[cid]     = max(best_sim.get(cid, 0.0), chunk.similarity)
                chunk_map[cid]    = chunk   # Last occurrence wins for metadata

        # ── Sort by RRF score ─────────────────────────────────────────────
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        fused = [
            FusedResult(
                chunk           = chunk_map[cid],
                rrf_score       = round(scores[cid], 6),
                appearances     = appearances[cid],
                best_similarity = round(best_sim[cid], 4),
            )
            for cid in sorted_ids[:top_k]
        ]

        logger.debug(
            f"RRF fusion: {sum(len(r) for r in non_empty)} total results "
            f"from {len(non_empty)} lists → {len(fused)} unique"
        )

        return fused