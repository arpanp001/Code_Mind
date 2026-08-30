# backend/app/core/analysis/architecture_memory.py
#
# IMPROVEMENT 3: Architecture Memory Integration
#
# Stores and retrieves the dependency graph from the project's
# ChromaDB memory collection so it's available at query time.
#
# Storage format:
#   memory_type: "note"
#   title:       "Architecture Dependency Graph"
#   tags:        ["architecture", "dependency-graph", "auto-generated"]
#   content:     JSON-serialised DependencyGraph
#
# At query time, the graph is loaded from memory and used by
# ArchitectureQueryEngine to answer structural questions.

import json
from typing import Optional

from app.core.analysis.dependency_analyzer import DependencyGraph, dependency_analyzer
from app.core.analysis.architecture_query  import (
    ArchitectureQueryEngine,
    ArchitectureContext,
    is_architecture_question,
    architecture_query_engine,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Tags used to identify the graph in the memory collection
GRAPH_MEMORY_TAGS  = ["architecture", "dependency-graph", "auto-generated"]
GRAPH_MEMORY_TITLE = "Architecture Dependency Graph"
MERMAID_TITLE      = "Architecture Mermaid Diagram"


class ArchitectureMemoryManager:
    """
    Persists the dependency graph in ChromaDB memory and retrieves it
    at query time.

    Why store in memory (not a separate collection)?
    - The memory collection already has semantic search
    - The graph summary is searchable as natural language
    - No new ChromaDB collections needed
    - Deletion is handled automatically when project is deleted

    The graph is stored as TWO memory entries:
    1. JSON graph (for programmatic queries)
    2. Human-readable summary (for semantic search)
    """

    def store_graph(
        self,
        project_id: str,
        graph:      DependencyGraph,
    ) -> None:
        """
        Stores the dependency graph and its summary in project memory.
        Called at the end of the ingestion pipeline.
        """
        from app.core.memory.project_memory import project_memory

        # 1. Store the full JSON graph (for exact queries)
        graph_json = json.dumps(graph.to_dict())

        try:
            project_memory.add_memory(
                project_id  = project_id,
                content     = graph_json,
                memory_type = "note",
                title       = GRAPH_MEMORY_TITLE,
                tags        = GRAPH_MEMORY_TAGS,
            )
            logger.info(f"💾 Dependency graph stored [{project_id}]")
        except Exception as e:
            logger.warning(f"Failed to store dependency graph: {e}")
            return

        # 2. Store the human-readable summary (for semantic search)
        summary = dependency_analyzer.generate_summary(graph)
        if summary:
            try:
                project_memory.add_memory(
                    project_id  = project_id,
                    content     = summary,
                    memory_type = "note",
                    title       = "Project Architecture Summary",
                    tags        = ["architecture", "summary", "auto-generated"],
                )
                logger.info(f"📋 Architecture summary stored [{project_id}]")
            except Exception as e:
                logger.warning(f"Failed to store architecture summary: {e}")

        # 3. Store the Mermaid diagram separately
        mermaid = dependency_analyzer.generate_mermaid(graph)
        if mermaid:
            try:
                project_memory.add_memory(
                    project_id  = project_id,
                    content     = f"```mermaid\n{mermaid}\n```",
                    memory_type = "note",
                    title       = MERMAID_TITLE,
                    tags        = ["architecture", "mermaid", "diagram", "auto-generated"],
                )
                logger.info(f"📊 Mermaid diagram stored [{project_id}]")
            except Exception as e:
                logger.warning(f"Failed to store Mermaid diagram: {e}")

    def load_graph(self, project_id: str) -> Optional[DependencyGraph]:
        """
        Loads the dependency graph from project memory.
        Returns None if no graph has been stored yet.
        """
        from app.core.memory.project_memory import project_memory

        try:
            memories = project_memory.list_memories(
                project_id  = project_id,
                memory_type = "note",
            )

            for mem in memories:
                if (
                    GRAPH_MEMORY_TITLE in (mem.title or "") and
                    "dependency-graph" in (mem.tags or [])
                ):
                    graph = DependencyGraph.from_dict(json.loads(mem.content))
                    logger.debug(
                        f"Loaded dependency graph [{project_id}]: "
                        f"{len(graph.nodes)} nodes"
                    )
                    return graph

        except Exception as e:
            logger.warning(f"Failed to load dependency graph [{project_id}]: {e}")

        return None

    def get_architecture_context(
        self,
        project_id: str,
        query:      str,
    ) -> Optional[ArchitectureContext]:
        """
        High-level method: loads the graph and queries it.
        Returns None if no graph is available or query is not architectural.

        Called from rag_generator.py when building the prompt.
        """
        if not is_architecture_question(query):
            return None

        graph = self.load_graph(project_id)
        if not graph:
            logger.debug(
                f"No dependency graph for [{project_id}] — "
                f"skipping architecture context"
            )
            return None

        try:
            context = architecture_query_engine.query(graph, query)
            logger.info(
                f"🏗️  Architecture context generated [{project_id}]: "
                f"{len(context.context_text)} chars, "
                f"type={context.query_type}"
            )
            return context
        except Exception as e:
            logger.warning(f"Architecture query failed: {e}")
            return None


# Module-level singleton
architecture_memory = ArchitectureMemoryManager()