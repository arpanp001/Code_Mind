# backend/app/core/analysis/architecture_query.py
#
# IMPROVEMENT 3: Architecture Query Engine
#
# Answers structural questions using the dependency graph instead of
# (or alongside) vector search. When a user asks:
#
#   "What is the request flow?"
#   "Which file is the entry point?"
#   "What does UserService depend on?"
#   "Show me the authentication architecture"
#
# The query engine:
#   1. Detects it's an architecture question (keyword matching)
#   2. Loads the project's dependency graph from memory
#   3. Queries the graph for relevant structural information
#   4. Returns a structured context string + optional Mermaid diagram
#
# This context is injected alongside the vector search results so
# Gemini answers with both code snippets AND structural knowledge.

import re
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

from app.core.analysis.dependency_analyzer import DependencyGraph, ModuleNode
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Architecture question detection ───────────────────────────────────────────

ARCHITECTURE_KEYWORDS = {
    # Entry point questions
    "entry point", "entrypoint", "main file", "start", "startup",
    "how does it start", "where does it start", "bootstrap",

    # Flow questions
    "request flow", "how does a request", "flow", "pipeline",
    "how does the app work", "lifecycle", "middleware chain",

    # Dependency questions
    "depends on", "depend on","dependency", "dependencies", "import",
    "what uses", "who calls", "what calls",

    # Architecture questions
    "architecture", "structure", "components", "modules",
    "how is it organized", "folder structure", "overview",

    # Relationship questions
    "inherits", "inherit", "extends", "implements", "relationship",
    "connects to", "talks to", "interacts with",
}


def is_architecture_question(query: str) -> bool:
    """
    Returns True if the query is asking about project structure,
    not about specific code implementation.

    Examples that return True:
      "What is the entry point?"
      "How does a request flow through the system?"
      "What does UserService depend on?"
      "Show the architecture"

    Examples that return False:
      "Where is JWT implemented?"     → code search question
      "Explain the authenticate() function" → code explanation
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in ARCHITECTURE_KEYWORDS)


# ── Query result types ─────────────────────────────────────────────────────────

@dataclass
class ArchitectureContext:
    """
    Structured answer from the architecture query engine.
    Contains text context + optional Mermaid diagram.
    """
    context_text:    str              # Plain text architecture context
    mermaid_diagram: str   = ""       # Optional Mermaid diagram
    files_referenced: list[str] = field(default_factory=list)
    query_type:      str   = "general"  # "entry_point", "dependency", "flow", "general"
    answered:        bool  = True       # False if graph had insufficient data


# ── Architecture Query Engine ──────────────────────────────────────────────────

class ArchitectureQueryEngine:
    """
    Queries the dependency graph to answer structural questions.

    Supports four query types:
    1. Entry point queries    → "where does the app start?"
    2. Dependency queries     → "what does X depend on?"
    3. Flow queries           → "how does a request flow?"
    4. General overview       → "describe the architecture"

    Usage:
        engine  = ArchitectureQueryEngine()
        context = engine.query(graph, "What is the entry point?")
        # context.context_text → injected into Gemini prompt
        # context.mermaid_diagram → shown in UI as a diagram
    """

    def query(
        self,
        graph: DependencyGraph,
        query: str,
    ) -> ArchitectureContext:
        """
        Main entry point. Routes to the appropriate sub-query.
        """
        if not graph or not graph.nodes:
            return ArchitectureContext(
                context_text = "No dependency graph available for this project.",
                answered     = False,
            )

        query_lower = query.lower()

        # Route to specialised handlers
        if any(kw in query_lower for kw in [
            "entry point", "entrypoint", "main file", "start", "startup",
            "bootstrap", "where does it start", "how does it start",
        ]):
            return self._query_entry_points(graph, query)

        if any(kw in query_lower for kw in [
            "depends on", "depend on",                
            "dependency", "dependencies", "import",
            "what uses", "who calls", "requires",
        ]):
            return self._query_dependencies(graph, query)

        if any(kw in query_lower for kw in [
            "request flow", "flow", "pipeline", "lifecycle",
            "how does a request", "middleware", "request path",
        ]):
            return self._query_request_flow(graph, query)

        if any(kw in query_lower for kw in [
            "inherits", "inherit",                   
            "extends", "implements", "inheritance",
            "class hierarchy", "base class",
        ]):
            return self._query_inheritance(graph, query)

        # Default: general overview
        return self._query_overview(graph, query)

    # ── Sub-queries ────────────────────────────────────────────────────────

    def _query_entry_points(
        self,
        graph: DependencyGraph,
        query: str,
    ) -> ArchitectureContext:
        """Answers: 'Where does the app start? What is the entry point?'"""
        lines = ["**Entry Points**\n"]
        files_referenced = []

        if not graph.entry_points:
            lines.append(
                "No explicit entry points detected. "
                "Check for `main.py`, `app.py`, `index.js`, or `main()` functions."
            )
        else:
            for ep_path in graph.entry_points[:8]:
                node = graph.nodes.get(ep_path)
                if not node:
                    continue

                files_referenced.append(ep_path)
                name = Path(ep_path).name
                lines.append(f"**`{name}`** (`{ep_path}`)")

                if node.entry_points:
                    for ep_detail in node.entry_points[:4]:
                        lines.append(f"  - {ep_detail}")

                if node.classes:
                    lines.append(
                        f"  - Defines: {', '.join(f'`{c}`' for c in node.classes[:4])}"
                    )

                # What does this entry point import?
                direct_imports = [
                    e.target for e in graph.edges
                    if e.source == ep_path and e.edge_type == "import"
                ]
                if direct_imports:
                    names = [Path(t).stem for t in direct_imports[:4]]
                    lines.append(
                        f"  - Imports: {', '.join(f'`{n}`' for n in names)}"
                    )
                lines.append("")

        return ArchitectureContext(
            context_text     = "\n".join(lines),
            files_referenced = files_referenced,
            query_type       = "entry_point",
        )

    def _query_dependencies(
        self,
        graph: DependencyGraph,
        query: str,
    ) -> ArchitectureContext:
        """
        Answers: 'What does X depend on?' or 'What depends on X?'
        Extracts the module name from the query and walks the graph.
        """
        # Try to extract a module/class name from the query
        target_name = self._extract_module_name(query)
        lines       = []
        files_referenced = []

        if target_name:
            # Find the file that defines or matches this name
            matched_file = self._find_file(graph, target_name)

            if matched_file:
                node = graph.nodes[matched_file]
                files_referenced.append(matched_file)
                lines.append(f"**Dependencies of `{Path(matched_file).name}`**\n")

                # What it imports
                imports_from = [
                    e.target for e in graph.edges
                    if e.source == matched_file and e.edge_type == "import"
                ]
                if imports_from:
                    lines.append("**Imports from:**")
                    for t in imports_from[:8]:
                        lines.append(f"  - `{Path(t).name}` (`{t}`)")
                        files_referenced.append(t)
                else:
                    lines.append("  No internal imports detected.")

                # What imports IT
                imported_by = [
                    e.source for e in graph.edges
                    if e.target == matched_file and e.edge_type == "import"
                ]
                if imported_by:
                    lines.append("\n**Imported by:**")
                    for s in imported_by[:8]:
                        lines.append(f"  - `{Path(s).name}` (`{s}`)")
                        files_referenced.append(s)

                # Inheritance
                if node.inherits_from:
                    lines.append(
                        f"\n**Inherits from:** "
                        f"{', '.join(f'`{b}`' for b in node.inherits_from)}"
                    )
            else:
                # Name not found — show general dependency stats
                lines.append(
                    f"Could not find a module named `{target_name}`. "
                    f"Showing most depended-on modules instead.\n"
                )
                most_imported = graph.stats.get("most_imported", [])
                for fp in most_imported[:5]:
                    count = sum(
                        1 for e in graph.edges
                        if e.target == fp and e.edge_type == "import"
                    )
                    lines.append(f"- `{Path(fp).name}` — imported by {count} modules")
                    files_referenced.append(fp)
        else:
            # General dependency overview
            lines.append("**Module Dependency Overview**\n")
            most_imported = graph.stats.get("most_imported", [])
            if most_imported:
                lines.append("**Most depended-on modules:**")
                for fp in most_imported[:6]:
                    count = sum(
                        1 for e in graph.edges
                        if e.target == fp and e.edge_type == "import"
                    )
                    lines.append(
                        f"  - `{Path(fp).name}` (`{fp}`) — "
                        f"imported by {count} module(s)"
                    )
                    files_referenced.append(fp)

        # Generate a focused Mermaid diagram
        mermaid = self._generate_dependency_mermaid(graph, files_referenced)

        return ArchitectureContext(
            context_text     = "\n".join(lines),
            mermaid_diagram  = mermaid,
            files_referenced = list(dict.fromkeys(files_referenced)),  # dedup
            query_type       = "dependency",
        )

    def _query_request_flow(
        self,
        graph: DependencyGraph,
        query: str,
    ) -> ArchitectureContext:
        """
        Answers: 'How does a request flow through the system?'
        Traces the path from entry points through routers to handlers.
        """
        lines            = ["**Request Flow**\n"]
        files_referenced = []

        if not graph.entry_points:
            lines.append(
                "No entry points detected. "
                "Cannot trace request flow automatically."
            )
            return ArchitectureContext(
                context_text = "\n".join(lines),
                query_type   = "flow",
                answered     = False,
            )

        # Start from entry points
        for ep_path in graph.entry_points[:3]:
            node = graph.nodes.get(ep_path)
            if not node:
                continue

            files_referenced.append(ep_path)
            lines.append(f"**Starting from `{Path(ep_path).name}`:**")

            # Show route handlers defined here
            if node.entry_points:
                lines.append("  Route handlers:")
                for ep in node.entry_points[:6]:
                    lines.append(f"    - {ep}")

            # Trace one level of imports
            first_level = [
                e.target for e in graph.edges
                if e.source == ep_path and e.edge_type == "import"
            ]
            if first_level:
                lines.append("  Delegates to:")
                for t in first_level[:4]:
                    t_node = graph.nodes.get(t)
                    files_referenced.append(t)
                    if t_node:
                        desc = f"{len(t_node.functions)} functions"
                        if t_node.classes:
                            desc += f", classes: {', '.join(t_node.classes[:2])}"
                        lines.append(f"    → `{Path(t).name}` ({desc})")
                    else:
                        lines.append(f"    → `{Path(t).name}`")

                # Trace second level
                for t in first_level[:2]:
                    second_level = [
                        e.target for e in graph.edges
                        if e.source == t and e.edge_type == "import"
                    ]
                    if second_level:
                        for s in second_level[:3]:
                            files_referenced.append(s)
                            lines.append(
                                f"        → `{Path(s).name}`"
                            )
            lines.append("")

        # Generate flow Mermaid diagram
        mermaid = self._generate_flow_mermaid(graph, files_referenced)

        return ArchitectureContext(
            context_text     = "\n".join(lines),
            mermaid_diagram  = mermaid,
            files_referenced = list(dict.fromkeys(files_referenced)),
            query_type       = "flow",
        )

    def _query_inheritance(
        self,
        graph: DependencyGraph,
        query: str,
    ) -> ArchitectureContext:
        """Answers: 'Show class hierarchy' or 'What does X inherit from?'"""
        lines            = ["**Class Inheritance Hierarchy**\n"]
        files_referenced = []
        found_any        = False

        for fp, node in graph.nodes.items():
            if not node.inherits_from:
                continue
            found_any = True
            files_referenced.append(fp)

            for cls in node.classes[:3]:
                for base in node.inherits_from[:3]:
                    lines.append(f"  `{cls}` extends `{base}` (in `{Path(fp).name}`)")

        if not found_any:
            lines.append(
                "No class inheritance relationships detected in this project."
            )

        return ArchitectureContext(
            context_text     = "\n".join(lines),
            files_referenced = files_referenced,
            query_type       = "inheritance",
        )

    def _query_overview(
        self,
        graph: DependencyGraph,
        query: str,
    ) -> ArchitectureContext:
        """General architecture overview."""
        from app.core.analysis.dependency_analyzer import dependency_analyzer
        lines            = []
        files_referenced = list(graph.entry_points[:5])

        s = graph.stats
        lines.append("**Project Architecture Overview**\n")
        lines.append(
            f"- **Modules:** {s.get('total_modules', 0)} files analyzed"
        )
        lines.append(
            f"- **Classes:** {s.get('total_classes', 0)} total"
        )
        lines.append(
            f"- **Functions:** {s.get('total_functions', 0)} total"
        )
        lines.append(
            f"- **Dependencies:** {s.get('total_edges', 0)} import relationships"
        )

        langs = s.get("languages", {})
        if langs:
            lang_str = ", ".join(
                f"{v} {k}" for k, v in
                sorted(langs.items(), key=lambda x: x[1], reverse=True)
            )
            lines.append(f"- **Languages:** {lang_str}")

        if graph.entry_points:
            ep_names = [Path(ep).name for ep in graph.entry_points[:4]]
            lines.append(f"\n**Entry Points:** {', '.join(f'`{n}`' for n in ep_names)}")

        most_imported = s.get("most_imported", [])
        if most_imported:
            lines.append("\n**Core modules (most imported):**")
            for fp in most_imported[:4]:
                count = sum(
                    1 for e in graph.edges
                    if e.target == fp and e.edge_type == "import"
                )
                lines.append(
                    f"  - `{Path(fp).name}` — used by {count} module(s)"
                )
                files_referenced.append(fp)

        # Full Mermaid diagram
        mermaid = dependency_analyzer.generate_mermaid(graph, max_nodes=15, max_edges=20)

        return ArchitectureContext(
            context_text     = "\n".join(lines),
            mermaid_diagram  = mermaid,
            files_referenced = list(dict.fromkeys(files_referenced)),
            query_type       = "general",
        )

    # ── Mermaid helpers ────────────────────────────────────────────────────

    def _generate_dependency_mermaid(
        self,
        graph:            DependencyGraph,
        focus_files:      list[str],
        max_edges:        int = 15,
    ) -> str:
        """Generates a Mermaid diagram focused on the given files."""
        if not focus_files:
            return ""

        focus_set  = set(focus_files)
        lines      = ["graph LR"]
        edge_count = 0

        def nid(fp: str) -> str:
            return re.sub(r'[^a-zA-Z0-9]', '_', Path(fp).stem)[:20]

        added_nodes = set()
        for fp in focus_files[:10]:
            if fp in graph.nodes and fp not in added_nodes:
                label = Path(fp).stem
                if fp in graph.entry_points:
                    lines.append(f'    {nid(fp)}["{label}"] :::entry')
                else:
                    lines.append(f'    {nid(fp)}("{label}")')
                added_nodes.add(fp)

        for edge in graph.edges:
            if edge_count >= max_edges:
                break
            if edge.source in focus_set or edge.target in focus_set:
                for fp in (edge.source, edge.target):
                    if fp not in added_nodes and fp in graph.nodes:
                        lines.append(f'    {nid(fp)}("{Path(fp).stem}")')
                        added_nodes.add(fp)
                arrow = "<|--" if edge.edge_type == "inheritance" else "-->"
                lines.append(f"    {nid(edge.source)} {arrow} {nid(edge.target)}")
                edge_count += 1

        lines.append("    classDef entry fill:#6366f1,color:#fff")
        return "\n".join(lines)

    def _generate_flow_mermaid(
        self,
        graph:       DependencyGraph,
        flow_files:  list[str],
    ) -> str:
        """Generates a top-down flow diagram."""
        if not flow_files:
            return ""

        lines = ["graph TD"]

        def nid(fp: str) -> str:
            return re.sub(r'[^a-zA-Z0-9]', '_', Path(fp).stem)[:20]

        added = set()
        for fp in flow_files[:12]:
            if fp not in graph.nodes or fp in added:
                continue
            label = Path(fp).stem
            if fp in graph.entry_points:
                lines.append(f'    {nid(fp)}["{label}"] :::entry')
            else:
                lines.append(f'    {nid(fp)}("{label}")')
            added.add(fp)

        for edge in graph.edges:
            if edge.source in set(flow_files) and edge.target in set(flow_files):
                lines.append(f"    {nid(edge.source)} --> {nid(edge.target)}")

        lines.append("    classDef entry fill:#6366f1,color:#fff")
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_module_name(self, query: str) -> Optional[str]:
        """
        Tries to extract a specific module/class name from a query.
        E.g. "What does UserService depend on?" → "UserService"
        """
        # Look for PascalCase names (class names)
        pascal = re.findall(r'\b[A-Z][a-zA-Z0-9]+(?:Service|Handler|Router|Manager|Controller|Model|Repository|Client|Engine|Store)?\b', query)
        if pascal:
            return pascal[0]

        # Look for snake_case file names
        snake = re.findall(r'\b([a-z][a-z0-9_]+\.(?:py|js|ts|go|java))\b', query)
        if snake:
            return snake[0].rsplit(".", 1)[0]

        # Look for quoted names
        quoted = re.findall(r'[\'"`]([^\'"`]+)[\'"`]', query)
        if quoted:
            return quoted[0]

        return None

    def _find_file(
        self,
        graph:       DependencyGraph,
        target_name: str,
    ) -> Optional[str]:
        """
        Finds the file that best matches a module/class name.
        """
        target_lower = target_name.lower()

        # Exact file stem match
        for fp in graph.nodes:
            if Path(fp).stem.lower() == target_lower:
                return fp

        # Class name match
        for fp, node in graph.nodes.items():
            if any(c.lower() == target_lower for c in node.classes):
                return fp

        # Partial path match
        for fp in graph.nodes:
            if target_lower in fp.lower():
                return fp

        return None


# Module-level singleton
architecture_query_engine = ArchitectureQueryEngine()