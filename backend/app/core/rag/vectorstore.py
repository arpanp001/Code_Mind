# backend/app/core/rag/vectorstore.py
#
# ChromaDB integration layer.
#
# Responsibilities:
#   1. Initialize and maintain the ChromaDB client
#   2. Create/get collections (one per project)
#   3. Store EmbeddedChunks with metadata
#   4. Perform semantic similarity search
#   5. Delete project collections on project removal
#
# Architecture note:
#   ChromaDB collections are named "{project_id}_code"
#   This isolates each project completely — a query on project A
#   never accidentally returns results from project B.

import re
import chromadb

from chromadb.config import Settings as ChromaSettings
from dataclasses     import dataclass, field
from typing          import Optional

from app.core.rag.embedder import EmbeddedChunk
from app.config            import settings
from app.utils.logger      import get_logger

logger = get_logger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Collection name pattern per project
# ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens/underscores
# We prefix with "cm_" (CodeMind) to avoid collisions with other apps
COLLECTION_NAME_PREFIX = "cm"

# Default number of results to return from similarity search
DEFAULT_TOP_K = 5

# Minimum similarity score to include in results (0.0 to 1.0)
# Cosine similarity after L2 normalization — 0.3 means 30% similar
# Anything below this is probably noise, not a genuine match
MIN_SIMILARITY_THRESHOLD = 0.35


# ── Result Data Class ─────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A chunk returned from ChromaDB similarity search.
    Combines the original stored data with the similarity score.
    """
    chunk_id:       str
    text:           str
    file_path:      str
    language:       str
    start_line:     int
    end_line:       int
    chunk_type:     str
    function_name:  str
    class_name:     str
    similarity:     float          # Cosine similarity: 0.0 (different) to 1.0 (identical)
    project_id:     str
    chunk_index:    int  = 0


@dataclass
class SearchResult:
    """
    Full result of a semantic search operation.
    """
    query:          str
    project_id:     str
    chunks:         list[RetrievedChunk]  = field(default_factory=list)
    total_found:    int                   = 0
    search_time_ms: float                 = 0.0


# ── Collection Name Helper ─────────────────────────────────────────────────────

def get_collection_name(project_id: str) -> str:
    """
    Generates a valid ChromaDB collection name for a project.

    ChromaDB naming rules:
    - 3 to 63 characters
    - Must start and end with alphanumeric character
    - Can contain hyphens and underscores in the middle
    - Cannot contain two consecutive periods

    We sanitize the project_id to ensure compliance.
    """
    # Replace any non-alphanumeric (except hyphen/underscore) with underscore
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', project_id)
    # Truncate to leave room for prefix
    safe_id = safe_id[:50]
    # Build name: "cm_<project_id>_code"
    name = f"{COLLECTION_NAME_PREFIX}_{safe_id}_code"
    # Ensure it starts with a letter (ChromaDB requirement)
    if not name[0].isalpha():
        name = f"cm_{name}"
    return name


# ── ChromaDB Client Singleton ──────────────────────────────────────────────────

class ChromaDBClient:
    """
    Manages the ChromaDB client and all collection operations.

    Uses lazy initialization — the client is created on first use,
    not at import time. This prevents startup failures if ChromaDB
    has a cold-start issue.

    All methods that need the client call _get_client() which
    initializes it if needed.
    """

    def __init__(self):
        self._client: Optional[chromadb.ClientAPI] = None

    def _get_client(self) -> chromadb.ClientAPI:
        """
        Returns the ChromaDB client, initializing it if necessary.
        Uses PersistentClient so data survives server restarts.
        """
        if self._client is None:
            logger.info(
                f"🗄️  Initializing ChromaDB at: "
                f"{settings.chroma_persist_path}"
            )
            self._client = chromadb.PersistentClient(
                path=settings.chroma_persist_path,
                settings=ChromaSettings(
                    # Disable telemetry — we don't want usage data sent to Chroma
                    anonymized_telemetry=False,
                    # Allow resetting collections (useful for dev/testing)
                    allow_reset=True,
                )
            )
            logger.info("✅ ChromaDB client initialized")
        return self._client

    def get_or_create_collection(
        self,
        project_id: str,
    ) -> chromadb.Collection:
        """
        Gets an existing collection for a project or creates a new one.

        Uses get_or_create_collection (not get or create separately)
        so this is safe to call multiple times — idempotent.

        The collection uses cosine similarity (via IP with normalized vectors).
        Since our embeddings are L2-normalized (Phase 8), inner product
        equals cosine similarity, which is the standard for semantic search.
        """
        client          = self._get_client()
        collection_name = get_collection_name(project_id)

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                # "hnsw:space" controls the distance metric
                # "cosine" → ChromaDB uses cosine similarity natively
                "hnsw:space":          "cosine",
                # HNSW index params — tuned for our use case:
                # M: number of connections per node (higher = better recall, more RAM)
                "hnsw:M":              16,
                # ef_construction: index build quality (higher = better, slower build)
                "hnsw:construction_ef": 100,
                # ef_search: query-time quality (higher = better recall, slower query)
                "hnsw:search_ef":       50,
                # Store project metadata in collection for reference
                "project_id":          project_id,
            }
        )

        logger.debug(
            f"Collection ready: {collection_name} "
            f"({collection.count()} existing docs)"
        )
        return collection

    def store_chunks(
        self,
        project_id:      str,
        embedded_chunks: list[EmbeddedChunk],
        batch_size:      int = 100,
    ) -> int:
        """
        Stores EmbeddedChunks into the project's ChromaDB collection.

        Uses UPSERT (not insert) — if a chunk_id already exists,
        it's updated rather than duplicated. This makes re-ingestion safe.

        Args:
            project_id:      The project these chunks belong to
            embedded_chunks: List of chunks with embeddings from Phase 8
            batch_size:      Chunks per ChromaDB upsert call
                             (ChromaDB recommends ≤500 per batch)

        Returns:
            Number of chunks successfully stored
        """
        if not embedded_chunks:
            logger.warning(f"store_chunks called with empty list [{project_id}]")
            return 0

        collection    = self.get_or_create_collection(project_id)
        stored_count  = 0
        total_batches = (len(embedded_chunks) + batch_size - 1) // batch_size

        logger.info(
            f"💾 Storing {len(embedded_chunks)} chunks "
            f"in {total_batches} batches [{project_id}]"
        )

        for batch_start in range(0, len(embedded_chunks), batch_size):
            batch      = embedded_chunks[batch_start : batch_start + batch_size]
            batch_num  = (batch_start // batch_size) + 1

            # ChromaDB upsert requires parallel lists:
            #   ids[]        → unique identifier for each document
            #   embeddings[] → the vector for each document
            #   documents[]  → the raw text (stored for retrieval)
            #   metadatas[]  → dict of filterable metadata per document
            ids        = []
            embeddings = []
            documents  = []
            metadatas  = []

            for ec in batch:
                chunk = ec.chunk

                ids.append(chunk.chunk_id)
                embeddings.append(ec.embedding)
                documents.append(chunk.text)

                # Metadata is what makes filtered search possible.
                # ChromaDB supports filtering on any metadata field.
                # Example: where={"language": "python", "chunk_type": "function"}
                #
                # IMPORTANT: ChromaDB metadata values must be
                # str, int, float, or bool — no lists or nested dicts.
                metadatas.append({
                    "project_id":    chunk.project_id,
                    "file_path":     chunk.file_path,
                    "language":      chunk.language,
                    "chunk_type":    chunk.chunk_type,
                    "function_name": chunk.function_name or "",
                    "class_name":    chunk.class_name    or "",
                    "start_line":    chunk.start_line,
                    "end_line":      chunk.end_line,
                    "chunk_index":   chunk.chunk_index,
                    "char_count":    chunk.char_count,
                    "token_count":   chunk.token_count,
                })

            try:
                # upsert = insert if new, update if exists
                # This is idempotent — safe to call multiple times
                collection.upsert(
                    ids        = ids,
                    embeddings = embeddings,
                    documents  = documents,
                    metadatas  = metadatas,
                )
                stored_count += len(batch)
                logger.debug(
                    f"  Batch {batch_num}/{total_batches}: "
                    f"stored {len(batch)} chunks"
                )

            except Exception as e:
                logger.error(
                    f"Failed to store batch {batch_num}: {e}",
                    exc_info=True
                )
                # Continue with next batch — partial storage is better
                # than total failure

        logger.info(
            f"✅ Stored {stored_count}/{len(embedded_chunks)} "
            f"chunks in ChromaDB [{project_id}]"
        )
        return stored_count

    def search(
        self,
        project_id:    str,
        query_embedding: list[float],
        top_k:         int             = DEFAULT_TOP_K,
        filters:       Optional[dict]  = None,
    ) -> SearchResult:
        """
        Performs semantic similarity search in a project's collection.

        Args:
            project_id:       Which project to search
            query_embedding:  384-dim vector from embed_query()
            top_k:            Number of results to return
            filters:          Optional ChromaDB where clause for metadata filtering
                              Examples:
                                {"language": "python"}
                                {"chunk_type": "function"}
                                {"language": {"$in": ["python", "javascript"]}}

        Returns:
            SearchResult with chunks ranked by similarity score
        """
        import time
        start = time.time()

        try:
            collection = self.get_or_create_collection(project_id)
            count      = collection.count()

            if count == 0:
                logger.warning(
                    f"Search on empty collection [{project_id}]"
                )
                return SearchResult(
                    query      = "",
                    project_id = project_id,
                )

            # Clamp top_k to available documents
            # (ChromaDB raises if you request more than exist)
            actual_top_k = min(top_k, count)

            # Build query kwargs
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results":        actual_top_k,
                # include what fields to return in results
                "include":          ["documents", "metadatas", "distances"],
            }

            # Apply metadata filters if provided
            if filters:
                query_kwargs["where"] = filters

            # Execute the similarity search
            raw = collection.query(**query_kwargs)

            # ── Parse ChromaDB results ────────────────────────────────────
            # ChromaDB returns parallel lists (one entry per query):
            # raw["ids"]        = [[id1, id2, ...]]       ← outer list = batch
            # raw["documents"]  = [[doc1, doc2, ...]]
            # raw["metadatas"]  = [[meta1, meta2, ...]]
            # raw["distances"]  = [[dist1, dist2, ...]]
            #
            # distances are cosine DISTANCES (0=identical, 2=opposite)
            # We convert to similarity: similarity = 1 - distance

            retrieved = []

            ids       = raw["ids"][0]        if raw["ids"]       else []
            docs      = raw["documents"][0]  if raw["documents"] else []
            metas     = raw["metadatas"][0]  if raw["metadatas"] else []
            distances = raw["distances"][0]  if raw["distances"] else []

            for chunk_id, doc, meta, distance in zip(
                ids, docs, metas, distances
            ):
                # Convert distance to similarity score
                # ChromaDB cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
                # Similarity = 1 - distance (so 1 = identical, 0 = orthogonal)
                similarity = max(0.0, 1.0 - float(distance))

                # Filter out low-relevance results
                if similarity < MIN_SIMILARITY_THRESHOLD:
                    continue

                retrieved.append(RetrievedChunk(
                    chunk_id      = chunk_id,
                    text          = doc,
                    file_path     = meta.get("file_path",     ""),
                    language      = meta.get("language",      ""),
                    start_line    = int(meta.get("start_line", 0)),
                    end_line      = int(meta.get("end_line",   0)),
                    chunk_type    = meta.get("chunk_type",    "block"),
                    function_name = meta.get("function_name", ""),
                    class_name    = meta.get("class_name",    ""),
                    similarity    = round(similarity, 4),
                    project_id    = project_id,
                    chunk_index   = int(meta.get("chunk_index", 0)),
                ))

            elapsed_ms = (time.time() - start) * 1000

            logger.info(
                f"🔍 Search complete [{project_id}]: "
                f"{len(retrieved)} results in {elapsed_ms:.1f}ms"
            )

            return SearchResult(
                query          = "",     # Caller fills this in
                project_id     = project_id,
                chunks         = retrieved,
                total_found    = len(retrieved),
                search_time_ms = elapsed_ms,
            )

        except Exception as e:
            logger.error(
                f"Search failed [{project_id}]: {e}",
                exc_info=True
            )
            return SearchResult(query="", project_id=project_id)

    def delete_project(self, project_id: str) -> bool:
        """
        Deletes a project's ChromaDB collection and all its chunks.
        Called when a project is deleted via DELETE /projects/{id}.

        Returns True if deleted, False if collection didn't exist.
        """
        client          = self._get_client()
        collection_name = get_collection_name(project_id)

        try:
            client.delete_collection(collection_name)
            logger.info(
                f"🗑️  Deleted ChromaDB collection: "
                f"{collection_name} [{project_id}]"
            )
            return True
        except Exception as e:
            # Collection might not exist (project failed before storage)
            logger.warning(
                f"Could not delete collection {collection_name}: {e}"
            )
            return False

    def get_collection_stats(self, project_id: str) -> dict:
        """
        Returns statistics about a project's collection.
        Used by debug endpoints and health checks.
        """
        try:
            collection = self.get_or_create_collection(project_id)
            count      = collection.count()
            return {
                "collection_name": get_collection_name(project_id),
                "total_chunks":    count,
                "project_id":      project_id,
            }
        except Exception as e:
            return {
                "collection_name": get_collection_name(project_id),
                "total_chunks":    0,
                "error":           str(e),
            }

    def list_all_collections(self) -> list[str]:
        """Returns names of all collections in the database."""
        try:
            client = self._get_client()
            return [c.name for c in client.list_collections()]
        except Exception:
            return []

    def reset_all(self) -> None:
        """
        ⚠️  DANGER: Deletes ALL collections and data.
        Only for development/testing. Never call in production.
        """
        if settings.app_env != "development":
            raise RuntimeError(
                "reset_all() is only allowed in development environment"
            )
        client = self._get_client()
        client.reset()
        logger.warning("⚠️  ChromaDB RESET — all data deleted")


# ── Module-level singleton ─────────────────────────────────────────────────────
chroma_client = ChromaDBClient()