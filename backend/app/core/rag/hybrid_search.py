# backend/app/core/rag/hybrid_search.py
#
# IMPROVEMENT 1: Enhanced Hybrid Search
#
# What changed vs original:
#   - Switched BM25Okapi → BM25Plus (fixes IDF=0 bug on small corpora)
#   - Path-aware scoring: boost src/app/core/services, penalise docs/examples
#   - Exact identifier match detection (function names, class names, var names)
#   - Chunk-type boost: function/class chunks ranked above generic blocks
#   - All signals merged into a single weighted final score
#   - Implementation files returned before documentation automatically
#
# Scoring formula:
#   final = (0.55 * semantic)
#         + (0.25 * bm25_normalised)
#         + (0.10 * path_score)
#         + (0.07 * exact_id_score)
#         + (0.03 * chunk_type_score)
#
# Path score ranges:
#   1.0 → src/, app/, core/, services/, lib/, pkg/, internal/
#   0.8 → api/, handlers/, controllers/, models/, schemas/
#   0.6 → config/, utils/, helpers/, common/
#   0.3 → tests/, test/, spec/, __tests__/
#   0.1 → docs/, documentation/, examples/, tutorial/, README.md

import re
import math
from dataclasses import dataclass, field

from rank_bm25 import BM25Plus          # BM25Plus avoids IDF=0 on small corpora
from app.core.rag.vectorstore import RetrievedChunk
from app.utils.logger         import get_logger

logger = get_logger(__name__)


# ── Weights (must sum to 1.0) ─────────────────────────────────────────────────

WEIGHT_SEMANTIC    = 0.55   # ChromaDB cosine similarity (most important)
WEIGHT_BM25        = 0.25   # Keyword match score
WEIGHT_PATH        = 0.10   # File path quality (src/ > docs/)
WEIGHT_EXACT_ID    = 0.07   # Exact identifier match in code
WEIGHT_CHUNK_TYPE  = 0.03   # Function/class > block

assert abs(
    WEIGHT_SEMANTIC + WEIGHT_BM25 + WEIGHT_PATH + WEIGHT_EXACT_ID + WEIGHT_CHUNK_TYPE - 1.0
) < 1e-9, "Weights must sum to 1.0"


# ── Path scoring tables ───────────────────────────────────────────────────────
#
# Directories that contain implementation code → high score
# Directories that contain tests/docs → low score
# The intuition: when a user asks "where is login implemented?", they want
# src/auth/login.py (score 1.0), NOT docs/authentication.md (score 0.1)

PATH_SCORES: list[tuple[tuple[str, ...], float]] = [
    # Tier 1 — Core implementation (1.0)
    (("src/", "app/", "core/", "services/", "lib/", "pkg/", "internal/",
      "main/", "server/", "backend/", "api/src/", "src/main/"), 1.0),

    # Tier 2 — API / data layer (0.85)
    (("api/", "handlers/", "controllers/", "models/", "schemas/",
      "routes/", "middleware/", "repositories/", "dao/", "store/"), 0.85),

    # Tier 3 — Utilities / config (0.65)
    (("config/", "configs/", "utils/", "helpers/", "common/",
      "shared/", "base/", "types/", "interfaces/"), 0.65),

    # Tier 4 — Tests (0.30)
    (("test/", "tests/", "spec/", "specs/", "__tests__/",
      "testing/", "fixtures/", "mocks/", "stubs/"), 0.30),

    # Tier 5 — Documentation / examples (0.10)
    (("docs/", "doc/", "documentation/", "examples/", "example/",
      "samples/", "demo/", "demos/", "tutorial/", "tutorials/",
      "guide/", "guides/"), 0.10),
]

# Filename-based overrides (applied after directory scoring)
FILENAME_BOOSTS: dict[str, float] = {
    # Entry points always score 1.0 regardless of directory
    "main.py":          1.0,
    "app.py":           1.0,
    "server.py":        1.0,
    "index.js":         1.0,
    "index.ts":         1.0,
    "main.go":          1.0,
    "main.rs":          1.0,
    "manage.py":        1.0,
    "wsgi.py":          0.9,
    "asgi.py":          0.9,
    "cli.py":           0.8,
    # Config files — medium value
    "settings.py":      0.7,
    "config.py":        0.7,
    "setup.py":         0.6,
    "pyproject.toml":   0.6,
    # Docs — always low
    "readme.md":        0.1,
    "changelog.md":     0.1,
    "contributing.md":  0.1,
    "license":          0.05,
}

# Chunk type → score (how valuable is this kind of chunk as a RAG answer?)
CHUNK_TYPE_SCORES: dict[str, float] = {
    "function":   1.0,   # Function definition — the most direct answer
    "class":      0.9,   # Class definition — high value for structure questions
    "method":     0.95,  # Method definition (subset of class, very specific)
    "heading":    0.6,   # Markdown section — useful for doc queries
    "block":      0.4,   # Generic code block — lower baseline
    "module":     0.7,   # Module-level code — useful for import questions
    "interface":  0.85,  # Interface/type definition
}


# ── Code tokeniser ────────────────────────────────────────────────────────────

def tokenize_code(text: str) -> list[str]:
    """
    Tokenises code for BM25 indexing.

    Handles:
    - camelCase splitting:   "authenticateUser" → ["authenticate", "user"]
    - snake_case splitting:  "create_jwt_token" → ["create", "jwt", "token"]
    - PascalCase splitting:  "UserService" → ["user", "service"]
    - Removes punctuation, digits-only tokens, tokens < 2 chars
    """
    # Split camelCase and PascalCase
    # "authenticateUser" → "authenticate User"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)

    # Replace non-alphanumeric with spaces
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Lowercase + split
    tokens = text.lower().split()

    # Keep only tokens with ≥ 2 chars and not purely numeric
    tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]

    return tokens


def extract_identifiers(text: str) -> set[str]:
    """
    Extracts code identifiers (function names, class names, variables)
    from a code snippet for exact-match boosting.

    Examples:
      "def authenticate_user(username, password):"
      → {"authenticate_user", "authenticate", "user", "username", "password"}

      "class UserAuthService(BaseService):"
      → {"UserAuthService", "BaseService", "user", "auth", "service", "base"}
    """
    identifiers = set()

    # Python/JS/TS/Go/Java identifier pattern
    raw_ids = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)

    for id_ in raw_ids:
        identifiers.add(id_.lower())
        # Also add split components
        parts = re.findall(r'[a-z]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', id_).lower())
        identifiers.update(parts)

    return identifiers


# ── Path scoring ──────────────────────────────────────────────────────────────

def score_file_path(file_path: str) -> float:
    """
    Assigns a quality score to a file path.

    Higher score = more likely to be an implementation file the user wants.
    Lower score = documentation, tests, examples.

    Returns a float in [0.0, 1.0].
    """
    path_lower  = file_path.lower().replace('\\', '/')
    filename    = path_lower.split('/')[-1]

    # Check filename-specific overrides first (most specific)
    if filename in FILENAME_BOOSTS:
        return FILENAME_BOOSTS[filename]

    # Check directory patterns (ordered from best to worst)
    for dirs, score in PATH_SCORES:
        if any(d in path_lower for d in dirs):
            return score

    # Default for unclassified paths (neutral)
    return 0.5


# ── Exact identifier matching ─────────────────────────────────────────────────

def score_exact_identifier_match(
    chunk:         RetrievedChunk,
    query_tokens:  list[str],
) -> float:
    """
    Scores how well the chunk's identifiers match the query tokens.

    Three levels:
    1.0 → Query token appears exactly in function_name or class_name
    0.7 → Query token appears in the code identifiers
    0.0 → No match

    Why this matters:
    A query for "authenticate" should strongly prefer a chunk where
    function_name="authenticate" over one where "authenticate" only appears
    in a comment.
    """
    if not query_tokens:
        return 0.0

    # Level 1: exact name match
    func_name  = (chunk.function_name or "").lower()
    class_name = (chunk.class_name    or "").lower()
    name_text  = f"{func_name} {class_name}"

    # Split name into component words
    name_words = set(re.findall(r'[a-z]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', name_text)))

    matching_in_name = sum(1 for t in query_tokens if t in name_words or t in name_text)
    if matching_in_name > 0:
        return min(1.0, matching_in_name / len(query_tokens))

    # Level 2: identifier match in code body
    code_ids = extract_identifiers(chunk.text[:500])  # First 500 chars
    matching_in_code = sum(1 for t in query_tokens if t in code_ids)
    if matching_in_code > 0:
        return min(0.7, 0.7 * matching_in_code / len(query_tokens))

    return 0.0


# ── Main HybridSearcher class ─────────────────────────────────────────────────

@dataclass
class HybridScore:
    """Detailed score breakdown for a single chunk (for debugging/logging)."""
    chunk_id:        str
    semantic:        float
    bm25_norm:       float
    path:            float
    exact_id:        float
    chunk_type:      float
    final:           float


class HybridSearcher:
    """
    Enhanced hybrid search combining:
    - Semantic similarity (ChromaDB cosine)
    - BM25+ keyword matching (camelCase/snake_case aware)
    - Path-aware scoring (src/ > docs/)
    - Exact identifier boosting (function/class name match)
    - Chunk type scoring (function > block)

    Usage:
        searcher = HybridSearcher()
        results  = searcher.hybrid_search(
            semantic_results = chunks_from_chromadb,
            query            = "authenticate user JWT",
            top_k            = 5,
        )
    """

    def __init__(
        self,
        semantic_weight:   float = WEIGHT_SEMANTIC,
        bm25_weight:       float = WEIGHT_BM25,
        path_weight:       float = WEIGHT_PATH,
        exact_id_weight:   float = WEIGHT_EXACT_ID,
        chunk_type_weight: float = WEIGHT_CHUNK_TYPE,
    ):
        self.semantic_weight   = semantic_weight
        self.bm25_weight       = bm25_weight
        self.path_weight       = path_weight
        self.exact_id_weight   = exact_id_weight
        self.chunk_type_weight = chunk_type_weight

    def hybrid_search(
        self,
        semantic_results: list[RetrievedChunk],
        query:            str,
        top_k:            int = 5,
    ) -> list[RetrievedChunk]:
        """
        Re-scores and re-ranks semantic search results using all signals.

        Returns the same RetrievedChunk objects but with updated similarity
        scores reflecting the combined hybrid score.

        Args:
            semantic_results: Chunks from ChromaDB semantic search
            query:            Original user query
            top_k:            Maximum results to return

        Returns:
            Re-scored list sorted by hybrid score descending
        """
        if not semantic_results:
            return []

        if len(semantic_results) == 1:
            # Apply path and type scoring even for single result
            chunk = semantic_results[0]
            hybrid_score = self._score_single(chunk, query, [], 1.0)
            return [self._update_similarity(chunk, hybrid_score.final)]

        # ── Build BM25+ index over candidate chunks ────────────────────────
        query_tokens = tokenize_code(query)
        corpus       = [tokenize_code(c.text) for c in semantic_results]

        try:
            # BM25Plus is used instead of BM25Okapi because BM25Okapi produces
            # IDF=0 for all terms when the corpus has fewer than ~4 documents.
            # This happens frequently during retrieval where we might only have
            # 2-5 candidate chunks. BM25Plus uses a different IDF formula that
            # always produces positive values regardless of corpus size.
            bm25       = BM25Plus(corpus)
            bm25_raw   = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(corpus)
        except Exception as e:
            logger.warning(f"BM25+ index build failed: {e} — using semantic only")
            bm25_raw = [0.0] * len(semantic_results)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in bm25_raw]

        # ── Compute final score for each chunk ─────────────────────────────
        scored_chunks: list[tuple[float, RetrievedChunk, HybridScore]] = []

        for chunk, bm25_score in zip(semantic_results, bm25_norm):
            score = self._score_single(chunk, query, query_tokens, bm25_score)
            scored_chunks.append((score.final, chunk, score))

        # ── Sort by final score descending ─────────────────────────────────
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Log top-3 scores for debugging
        logger.debug(
            f"Hybrid search: {len(semantic_results)} chunks re-scored "
            f"(sem={self.semantic_weight}, bm25={self.bm25_weight}, "
            f"path={self.path_weight})"
        )
        for final_score, chunk, breakdown in scored_chunks[:3]:
            logger.debug(
                f"  [{chunk.chunk_id[:8]}] {chunk.file_path} "
                f"final={breakdown.final:.3f} "
                f"sem={breakdown.semantic:.3f} "
                f"bm25={breakdown.bm25_norm:.3f} "
                f"path={breakdown.path:.3f} "
                f"id={breakdown.exact_id:.3f} "
                f"type={breakdown.chunk_type:.3f}"
            )

        # Return updated RetrievedChunk objects with hybrid scores
        result = []
        for final_score, chunk, _ in scored_chunks[:top_k]:
            result.append(self._update_similarity(chunk, final_score))

        return result

    def _score_single(
        self,
        chunk:        RetrievedChunk,
        query:        str,
        query_tokens: list[str],
        bm25_score:   float,
    ) -> HybridScore:
        """Computes all scoring signals for one chunk."""
        # Signal 1: Semantic similarity (already 0-1 from ChromaDB)
        semantic = chunk.similarity

        # Signal 2: BM25+ keyword score (already normalised 0-1)
        bm25_norm = bm25_score

        # Signal 3: Path quality score
        path = score_file_path(chunk.file_path)

        # Signal 4: Exact identifier match
        exact_id = score_exact_identifier_match(chunk, query_tokens)

        # Signal 5: Chunk type quality
        chunk_type = CHUNK_TYPE_SCORES.get(chunk.chunk_type, 0.4)

        # Weighted combination
        final = (
            self.semantic_weight   * semantic   +
            self.bm25_weight       * bm25_norm  +
            self.path_weight       * path       +
            self.exact_id_weight   * exact_id   +
            self.chunk_type_weight * chunk_type
        )
        final = round(min(1.0, max(0.0, final)), 4)

        return HybridScore(
            chunk_id   = chunk.chunk_id,
            semantic   = round(semantic,   4),
            bm25_norm  = round(bm25_norm,  4),
            path       = round(path,       4),
            exact_id   = round(exact_id,   4),
            chunk_type = round(chunk_type, 4),
            final      = final,
        )

    @staticmethod
    def _update_similarity(chunk: RetrievedChunk, new_score: float) -> RetrievedChunk:
        """Returns a copy of the chunk with updated similarity score."""
        return RetrievedChunk(
            chunk_id      = chunk.chunk_id,
            text          = chunk.text,
            file_path     = chunk.file_path,
            language      = chunk.language,
            start_line    = chunk.start_line,
            end_line      = chunk.end_line,
            chunk_type    = chunk.chunk_type,
            function_name = chunk.function_name,
            class_name    = chunk.class_name,
            similarity    = new_score,
            project_id    = chunk.project_id,
            chunk_index   = chunk.chunk_index,
        )