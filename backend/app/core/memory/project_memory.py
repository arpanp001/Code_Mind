# backend/app/core/memory/project_memory.py
#
# Project Memory System.
#
# Stores three types of memory:
#   ARCHITECTURE_DECISION: "We chose JWT over sessions because..."
#   BUG_FIX:               "Fixed null pointer in auth.py by checking..."
#   NOTE:                  General implementation notes
#
# Memories are stored in ChromaDB (semantic search)
# AND in SQLite (listing, filtering, deletion).
#
# When a user asks a question, we also search memories for relevant
# context and inject them into the Gemini prompt alongside code chunks.

import uuid
from datetime    import datetime, timezone
from dataclasses import dataclass, field
from typing      import Optional

from app.core.rag.embedder    import embedding_engine
from app.core.rag.vectorstore import chroma_client
from app.utils.logger         import get_logger

logger = get_logger(__name__)

# ChromaDB collection name for memories
MEMORY_COLLECTION_SUFFIX = "memory"


def get_memory_collection_name(project_id: str) -> str:
    """Collection name for a project's memories."""
    import re
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', project_id)[:40]
    return f"cm_{safe}_memory"


@dataclass
class MemoryEntry:
    """A single project memory."""
    memory_id:   str
    project_id:  str
    content:     str
    memory_type: str          # "architecture_decision", "bug_fix", "note"
    title:       str  = ""
    tags:        list = field(default_factory=list)
    created_at:  str  = ""


@dataclass
class MemorySearchResult:
    """A memory returned from semantic search."""
    memory:     MemoryEntry
    similarity: float


class ProjectMemoryManager:
    """
    Manages project memories with semantic search.

    Memories are stored in a separate ChromaDB collection per project
    (distinct from the code chunks collection).
    """

    def add_memory(
        self,
        project_id:  str,
        content:     str,
        memory_type: str,
        title:       str       = "",
        tags:        list[str] = None,
    ) -> MemoryEntry:
        """
        Adds a new memory and stores it in ChromaDB.
        """
        tags       = tags or []
        memory_id  = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        entry = MemoryEntry(
            memory_id   = memory_id,
            project_id  = project_id,
            content     = content,
            memory_type = memory_type,
            title       = title,
            tags        = tags,
            created_at  = created_at,
        )

        # Generate embedding for the memory content
        try:
            embedding = embedding_engine.embed_query(content)
        except Exception as e:
            logger.error(f"Failed to embed memory: {e}")
            raise

        # Store in ChromaDB
        collection_name = get_memory_collection_name(project_id)
        try:
            client     = chroma_client._get_client()
            collection = client.get_or_create_collection(
                name     = collection_name,
                metadata = {
                    "hnsw:space": "cosine",
                    "project_id": project_id,
                    "type":       "memory",
                }
            )
            collection.upsert(
                ids        = [memory_id],
                embeddings = [embedding],
                documents  = [content],
                metadatas  = [{
                    "project_id":  project_id,
                    "memory_type": memory_type,
                    "title":       title,
                    "tags":        ",".join(tags),
                    "created_at":  created_at,
                }],
            )
            logger.info(
                f"💾 Memory saved [{project_id}]: "
                f"{memory_type} - {title or content[:40]}"
            )
        except Exception as e:
            logger.error(f"Failed to store memory in ChromaDB: {e}")
            raise

        return entry

    def search_memories(
        self,
        project_id:  str,
        query:       str,
        memory_type: Optional[str] = None,
        top_k:       int           = 3,
    ) -> list[MemorySearchResult]:
        """
        Searches memories by semantic similarity to the query.
        Used to inject relevant memories into the chat prompt.
        """
        try:
            query_embedding = embedding_engine.embed_query(query)
        except Exception as e:
            logger.warning(f"Memory search embedding failed: {e}")
            return []

        collection_name = get_memory_collection_name(project_id)

        try:
            client = chroma_client._get_client()
            # Check if collection exists
            existing = [c.name for c in client.list_collections()]
            if collection_name not in existing:
                return []   # No memories for this project yet

            collection = client.get_or_create_collection(collection_name)

            if collection.count() == 0:
                return []

            # Build filters
            where = None
            if memory_type:
                where = {"memory_type": memory_type}

            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results":        min(top_k, collection.count()),
                "include":          ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            raw = collection.query(**query_kwargs)

            results = []
            ids       = raw["ids"][0]       if raw["ids"]       else []
            docs      = raw["documents"][0] if raw["documents"]  else []
            metas     = raw["metadatas"][0] if raw["metadatas"]  else []
            distances = raw["distances"][0] if raw["distances"]  else []

            for mid, doc, meta, dist in zip(ids, docs, metas, distances):
                similarity = max(0.0, 1.0 - float(dist))
                if similarity < 0.3:
                    continue

                tags_str = meta.get("tags", "")
                tags     = [t for t in tags_str.split(",") if t] if tags_str else []

                entry = MemoryEntry(
                    memory_id   = mid,
                    project_id  = project_id,
                    content     = doc,
                    memory_type = meta.get("memory_type", "note"),
                    title       = meta.get("title", ""),
                    tags        = tags,
                    created_at  = meta.get("created_at", ""),
                )
                results.append(MemorySearchResult(
                    memory     = entry,
                    similarity = round(similarity, 4),
                ))

            logger.info(
                f"🧠 Memory search [{project_id}]: "
                f"{len(results)} relevant memories found"
            )
            return results

        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    def list_memories(
        self,
        project_id:  str,
        memory_type: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """Lists all memories for a project."""
        collection_name = get_memory_collection_name(project_id)

        try:
            client   = chroma_client._get_client()
            existing = [c.name for c in client.list_collections()]
            if collection_name not in existing:
                return []

            collection = client.get_or_create_collection(collection_name)
            if collection.count() == 0:
                return []

            get_kwargs = {"include": ["documents", "metadatas"]}
            if memory_type:
                get_kwargs["where"] = {"memory_type": memory_type}

            raw   = collection.get(**get_kwargs)
            ids   = raw.get("ids",       [])
            docs  = raw.get("documents", [])
            metas = raw.get("metadatas", [])

            entries = []
            for mid, doc, meta in zip(ids, docs, metas):
                tags_str = meta.get("tags", "")
                tags     = [t for t in tags_str.split(",") if t] if tags_str else []
                entries.append(MemoryEntry(
                    memory_id   = mid,
                    project_id  = project_id,
                    content     = doc,
                    memory_type = meta.get("memory_type", "note"),
                    title       = meta.get("title", ""),
                    tags        = tags,
                    created_at  = meta.get("created_at", ""),
                ))

            # Sort newest first
            entries.sort(key=lambda e: e.created_at, reverse=True)
            return entries

        except Exception as e:
            logger.warning(f"List memories failed: {e}")
            return []

    def delete_memory(self, project_id: str, memory_id: str) -> bool:
        """Deletes a specific memory."""
        collection_name = get_memory_collection_name(project_id)
        try:
            client     = chroma_client._get_client()
            collection = client.get_or_create_collection(collection_name)
            collection.delete(ids=[memory_id])
            logger.info(f"🗑️  Memory deleted: {memory_id}")
            return True
        except Exception as e:
            logger.warning(f"Delete memory failed: {e}")
            return False

    def delete_project_memories(self, project_id: str) -> None:
        """Deletes all memories for a project (called on project delete)."""
        collection_name = get_memory_collection_name(project_id)
        try:
            client = chroma_client._get_client()
            client.delete_collection(collection_name)
            logger.info(f"🗑️  All memories deleted for: {project_id}")
        except Exception as e:
            logger.warning(f"Could not delete memory collection: {e}")


# Module-level singleton
project_memory = ProjectMemoryManager()