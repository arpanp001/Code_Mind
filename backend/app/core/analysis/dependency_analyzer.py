# backend/app/core/analysis/dependency_analyzer.py
#
# IMPROVEMENT 3: Repository Architecture & Dependency Intelligence
#
# This module analyzes a codebase during indexing to build a dependency graph:
#
#   Nodes: modules (files)
#   Edges: import relationships, inheritance, function calls
#
# What gets extracted per file:
#   - imports:       what other modules this file imports
#   - exports:       functions/classes defined and exported
#   - calls:         external function calls (to other modules)
#   - inherits_from: class inheritance relationships
#   - entry_points:  detected main functions, route handlers, CLI entrypoints
#
# The graph is stored as a JSON blob in the project's memory collection
# so it's available at query time without re-analyzing.
#
# Architecture questions like:
#   "What is the request flow?"
#   "Which file is the entry point?"
#   "What does UserService depend on?"
# are answered using this graph instead of (or alongside) vector search.

import ast
import re
import json
from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional

from app.models.ingest_models import ParsedFile
from app.utils.logger         import get_logger

logger = get_logger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ModuleNode:
    """
    Represents a single file (module) in the dependency graph.
    Each field is populated by the language-specific extractor.
    """
    file_path:     str                    # Relative path e.g. "src/auth/login.py"
    language:      str                    # "python", "javascript", etc.
    module_name:   str                    # Derived name e.g. "auth.login"
    imports:       list[str]  = field(default_factory=list)  # Modules this file imports
    exports:       list[str]  = field(default_factory=list)  # Names defined here
    calls:         list[str]  = field(default_factory=list)  # External calls detected
    inherits_from: list[str]  = field(default_factory=list)  # Base classes
    entry_points:  list[str]  = field(default_factory=list)  # main/routes/cli handlers
    is_entry_point: bool      = False                         # True if this IS an entrypoint
    classes:       list[str]  = field(default_factory=list)  # Class names defined here
    functions:     list[str]  = field(default_factory=list)  # Function names defined here
    size_lines:    int        = 0                             # Line count


@dataclass
class DependencyEdge:
    """A directed relationship between two modules."""
    source:        str   # file_path of the importing module
    target:        str   # file_path of the imported module (resolved if possible)
    edge_type:     str   # "import", "inheritance", "call", "route"
    label:         str   # Human-readable label e.g. "imports UserService"


@dataclass
class DependencyGraph:
    """
    Complete dependency graph for a project.
    Serialisable to JSON for storage in ChromaDB memory.
    """
    project_id:    str
    nodes:         dict[str, ModuleNode]    = field(default_factory=dict)
    edges:         list[DependencyEdge]     = field(default_factory=list)
    entry_points:  list[str]               = field(default_factory=list)
    summary:       str                     = ""   # Human-readable overview
    stats:         dict                    = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts to JSON-serialisable dict for storage."""
        return {
            "project_id":  self.project_id,
            "nodes": {
                fp: {
                    "file_path":     n.file_path,
                    "language":      n.language,
                    "module_name":   n.module_name,
                    "imports":       n.imports,
                    "exports":       n.exports,
                    "calls":         n.calls,
                    "inherits_from": n.inherits_from,
                    "entry_points":  n.entry_points,
                    "is_entry_point": n.is_entry_point,
                    "classes":       n.classes,
                    "functions":     n.functions,
                    "size_lines":    n.size_lines,
                }
                for fp, n in self.nodes.items()
            },
            "edges": [
                {
                    "source":    e.source,
                    "target":    e.target,
                    "edge_type": e.edge_type,
                    "label":     e.label,
                }
                for e in self.edges
            ],
            "entry_points": self.entry_points,
            "summary":      self.summary,
            "stats":        self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        """Deserialises from stored JSON dict."""
        graph = cls(project_id=data.get("project_id", "unknown"))
        for fp, nd in data.get("nodes", {}).items():
            graph.nodes[fp] = ModuleNode(**{
                k: nd[k] for k in ModuleNode.__dataclass_fields__ if k in nd
            })
        for ed in data.get("edges", []):
            graph.edges.append(DependencyEdge(**ed))
        graph.entry_points = data.get("entry_points", [])
        graph.summary      = data.get("summary", "")
        graph.stats        = data.get("stats", {})
        return graph


# ── Language extractors ────────────────────────────────────────────────────────

class PythonExtractor:
    """
    Extracts dependency information from Python files using ast.parse().

    What it detects:
    - import X / from X import Y  → imports
    - def foo():                   → functions
    - class Foo(Bar):              → classes + inheritance
    - @app.route / @router.get    → route entry points
    - if __name__ == "__main__"    → main entry point
    - Flask/FastAPI app = App()   → framework instantiation
    """

    # Patterns that indicate a file is a Flask/FastAPI entry point
    FRAMEWORK_PATTERNS = [
        re.compile(r'\bapp\s*=\s*Flask\s*\('),
        re.compile(r'\bapp\s*=\s*FastAPI\s*\('),
        re.compile(r'\bapp\s*=\s*Starlette\s*\('),
        re.compile(r'\brouter\s*=\s*APIRouter\s*\('),
        re.compile(r'\bbp\s*=\s*Blueprint\s*\('),
    ]

    # Decorator patterns that indicate route handlers
    ROUTE_DECORATORS = {
        "route", "get", "post", "put", "delete", "patch",
        "head", "options", "websocket",
    }

    def extract(self, parsed_file: ParsedFile) -> ModuleNode:
        content = parsed_file.content
        lines   = content.splitlines()

        node = ModuleNode(
            file_path   = parsed_file.file_path,
            language    = "python",
            module_name = self._path_to_module(parsed_file.file_path),
            size_lines  = len(lines),
        )

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Can't parse — return what we know from raw text
            node.imports = self._extract_imports_regex(content)
            return node

        # ── Imports ───────────────────────────────────────────────────────
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    node.imports.append(alias.name)

            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    module = n.module
                    # Resolve relative imports
                    if n.level and n.level > 0:
                        parent = Path(parsed_file.file_path).parent
                        module = str(parent).replace("/", ".").replace("\\", ".") + "." + module
                    node.imports.append(module)
                    # Also add specific names for call tracking
                    for alias in n.names:
                        if alias.name != "*":
                            node.exports.append(f"{module}.{alias.name}")

        # ── Classes, functions, inheritance ───────────────────────────────
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                node.classes.append(n.name)
                node.exports.append(n.name)
                # Base classes → inheritance edges
                for base in n.bases:
                    base_name = self._get_name(base)
                    if base_name and base_name not in ("object", "ABC", "BaseModel"):
                        node.inherits_from.append(base_name)

            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.functions.append(n.name)
                if not self._is_method(n, tree):
                    node.exports.append(n.name)

                # Check decorators for route handlers
                for dec in n.decorator_list:
                    dec_name = self._get_decorator_name(dec)
                    if dec_name:
                        parts = dec_name.split(".")
                        if parts[-1] in self.ROUTE_DECORATORS:
                            node.entry_points.append(
                                f"{n.name} ({dec_name})"
                            )

        # ── Entry point detection ─────────────────────────────────────────
        if self._has_main_block(content):
            node.entry_points.append("__main__ block")
            node.is_entry_point = True

        for pattern in self.FRAMEWORK_PATTERNS:
            if pattern.search(content):
                node.is_entry_point = True
                break

        filename = Path(parsed_file.file_path).name
        if filename in ("main.py", "app.py", "server.py", "wsgi.py",
                        "asgi.py", "manage.py", "cli.py", "run.py"):
            node.is_entry_point = True

        # ── External function calls ───────────────────────────────────────
        node.calls = self._extract_calls(tree)

        return node

    def _path_to_module(self, file_path: str) -> str:
        """Converts file path to Python module notation."""
        path = file_path.replace("\\", "/")
        # Remove common source prefixes
        for prefix in ("src/", "app/", "lib/", "backend/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
        return path.replace("/", ".").removesuffix(".py")

    def _get_name(self, node: ast.AST) -> Optional[str]:
        """Extracts the name string from a Name or Attribute node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._get_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None

    def _get_decorator_name(self, node: ast.AST) -> Optional[str]:
        """Extracts decorator name from a decorator node."""
        if isinstance(node, ast.Attribute):
            parent = self._get_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return None

    def _is_method(self, func_node: ast.AST, tree: ast.AST) -> bool:
        """Returns True if the function is defined inside a class."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True
        return False

    def _has_main_block(self, content: str) -> bool:
        return '__name__' in content and '__main__' in content

    def _extract_calls(self, tree: ast.AST) -> list[str]:
        """Extracts names of called functions/methods (top 20)."""
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._get_name(node.func)
                if name and "." in name:
                    calls.add(name)
        return list(calls)[:20]

    def _extract_imports_regex(self, content: str) -> list[str]:
        """Fallback regex import extraction when AST fails."""
        imports = []
        for m in re.finditer(r'^(?:import|from)\s+([\w.]+)', content, re.MULTILINE):
            imports.append(m.group(1))
        return imports


class JavaScriptExtractor:
    """
    Extracts dependency information from JavaScript/TypeScript files.

    Detects:
    - import X from 'module'   → imports
    - require('module')        → imports (CommonJS)
    - export function/class    → exports
    - class Foo extends Bar    → inheritance
    - express.Router()         → framework entry points
    - app.use/get/post         → route handlers
    """

    IMPORT_PATTERNS = [
        # ES6: import X from 'module'
        re.compile(r"import\s+(?:.*?\s+from\s+)?['\"]([^'\"]+)['\"]"),
        # CommonJS: require('module')
        re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
    ]

    EXPORT_PATTERNS = [
        re.compile(r"export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)"),
        re.compile(r"export\s+(?:default\s+)?class\s+(\w+)"),
        re.compile(r"export\s+(?:const|let|var)\s+(\w+)"),
        re.compile(r"module\.exports\s*=\s*\{([^}]+)\}"),
    ]

    def extract(self, parsed_file: ParsedFile) -> ModuleNode:
        content  = parsed_file.content
        lines    = content.splitlines()
        language = parsed_file.language

        node = ModuleNode(
            file_path   = parsed_file.file_path,
            language    = language,
            module_name = self._path_to_module(parsed_file.file_path),
            size_lines  = len(lines),
        )

        # ── Imports ───────────────────────────────────────────────────────
        for pattern in self.IMPORT_PATTERNS:
            for m in pattern.finditer(content):
                module = m.group(1)
                # Skip node_modules (external deps)
                if not module.startswith("."):
                    node.imports.append(module)
                else:
                    # Relative import — store as-is for graph resolution
                    node.imports.append(module)

        # ── Exports ───────────────────────────────────────────────────────
        for pattern in self.EXPORT_PATTERNS:
            for m in pattern.finditer(content):
                name = m.group(1).strip()
                if name:
                    node.exports.append(name)

        # ── Classes and inheritance ───────────────────────────────────────
        for m in re.finditer(
            r'class\s+(\w+)(?:\s+extends\s+(\w+))?', content
        ):
            class_name = m.group(1)
            base_name  = m.group(2)
            node.classes.append(class_name)
            if base_name:
                node.inherits_from.append(base_name)

        # ── Functions ─────────────────────────────────────────────────────
        for m in re.finditer(
            r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()',
            content,
        ):
            fname = m.group(1) or m.group(2)
            if fname:
                node.functions.append(fname)

        # ── Entry points ──────────────────────────────────────────────────
        if any(p in content for p in [
            "app.listen(", "server.listen(", "createServer(",
            "express()", "Fastify(", "Koa()",
        ]):
            node.is_entry_point = True

        for m in re.finditer(
            r'(?:app|router)\.(get|post|put|delete|patch|use)\s*\(\s*[\'"]([^\'"]+)[\'"]',
            content,
        ):
            node.entry_points.append(f"{m.group(1).upper()} {m.group(2)}")

        filename = Path(parsed_file.file_path).name
        if filename in ("index.js", "index.ts", "main.js", "main.ts",
                        "server.js", "server.ts", "app.js", "app.ts"):
            node.is_entry_point = True

        return node

    def _path_to_module(self, file_path: str) -> str:
        path = file_path.replace("\\", "/")
        for prefix in ("src/", "lib/", "app/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
        return path.removesuffix(".js").removesuffix(".ts")


class GenericExtractor:
    """
    Fallback extractor for Java, Go, Rust, and other languages.
    Uses regex patterns to detect imports and definitions.
    Good enough for architecture-level analysis even without AST.
    """

    PATTERNS = {
        "java": {
            "import":  re.compile(r'^import\s+([\w.]+);', re.MULTILINE),
            "class":   re.compile(r'\bclass\s+(\w+)(?:\s+extends\s+(\w+))?'),
            "iface":   re.compile(r'\binterface\s+(\w+)'),
        },
        "go": {
            "import":  re.compile(r'"([\w./]+)"'),
            "func":    re.compile(r'^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(', re.MULTILINE),
            "struct":  re.compile(r'^type\s+(\w+)\s+struct', re.MULTILINE),
        },
        "rust": {
            "import":  re.compile(r'^use\s+([\w:]+)', re.MULTILINE),
            "func":    re.compile(r'^(?:pub\s+)?fn\s+(\w+)', re.MULTILINE),
            "struct":  re.compile(r'^(?:pub\s+)?struct\s+(\w+)', re.MULTILINE),
        },
    }

    def extract(self, parsed_file: ParsedFile) -> ModuleNode:
        content  = parsed_file.content
        language = parsed_file.language
        patterns = self.PATTERNS.get(language, {})

        node = ModuleNode(
            file_path   = parsed_file.file_path,
            language    = language,
            module_name = Path(parsed_file.file_path).stem,
            size_lines  = len(content.splitlines()),
        )

        if "import" in patterns:
            for m in patterns["import"].finditer(content):
                node.imports.append(m.group(1))

        if "class" in patterns:
            for m in patterns["class"].finditer(content):
                node.classes.append(m.group(1))
                if m.lastindex >= 2 and m.group(2):
                    node.inherits_from.append(m.group(2))

        if "iface" in patterns:
            for m in patterns["iface"].finditer(content):
                node.classes.append(m.group(1))

        for key in ("func", "struct"):
            if key in patterns:
                for m in patterns[key].finditer(content):
                    node.functions.append(m.group(1))

        return node


# ── Main DependencyAnalyzer ────────────────────────────────────────────────────

class DependencyAnalyzer:
    """
    Orchestrates dependency analysis across all files in a project.

    Usage:
        analyzer = DependencyAnalyzer()
        graph    = analyzer.analyze(parsed_files, project_id="my_project")
        summary  = analyzer.generate_summary(graph)
        mermaid  = analyzer.generate_mermaid(graph)
    """

    def __init__(self):
        self._python_extractor = PythonExtractor()
        self._js_extractor     = JavaScriptExtractor()
        self._generic          = GenericExtractor()

    def analyze(
        self,
        parsed_files: list[ParsedFile],
        project_id:   str,
        max_files:    int = 300,   # Cap for large repos
    ) -> DependencyGraph:
        """
        Analyzes all files and builds the dependency graph.

        Steps:
        1. Extract module info from each file
        2. Resolve relative imports to file paths
        3. Build edges (import, inheritance, call)
        4. Detect entry points
        5. Generate summary stats
        """
        logger.info(
            f"🔍 Analyzing dependencies: {len(parsed_files)} files "
            f"[{project_id}]"
        )

        graph = DependencyGraph(project_id=project_id)

        # Limit analysis for very large repos
        files_to_analyze = parsed_files[:max_files]

        # Build a path → file map for import resolution
        path_map = {pf.file_path: pf for pf in files_to_analyze}
        # Also index by module name for Python resolution
        module_map: dict[str, str] = {}

        # ── Step 1: Extract module info from each file ─────────────────────
        for pf in files_to_analyze:
            try:
                node = self._extract_node(pf)
                graph.nodes[pf.file_path] = node
                module_map[node.module_name] = pf.file_path
            except Exception as e:
                logger.debug(f"  Skipping {pf.file_path}: {e}")

        logger.info(f"  Extracted {len(graph.nodes)} module nodes")

        # ── Step 2: Build edges ────────────────────────────────────────────
        for file_path, node in graph.nodes.items():
            # Import edges
            for imp in node.imports:
                target = self._resolve_import(imp, file_path, module_map)
                if target and target != file_path:
                    graph.edges.append(DependencyEdge(
                        source    = file_path,
                        target    = target,
                        edge_type = "import",
                        label     = f"imports {Path(target).stem}",
                    ))

            # Inheritance edges
            for base in node.inherits_from:
                target = self._resolve_name(base, graph.nodes)
                if target:
                    graph.edges.append(DependencyEdge(
                        source    = file_path,
                        target    = target,
                        edge_type = "inheritance",
                        label     = f"extends {base}",
                    ))

        # ── Step 3: Identify entry points ──────────────────────────────────
        for file_path, node in graph.nodes.items():
            if node.is_entry_point or node.entry_points:
                graph.entry_points.append(file_path)

        # ── Step 4: Compute stats ──────────────────────────────────────────
        lang_counts: dict[str, int] = {}
        for node in graph.nodes.values():
            lang_counts[node.language] = lang_counts.get(node.language, 0) + 1

        graph.stats = {
            "total_modules":     len(graph.nodes),
            "total_edges":       len(graph.edges),
            "entry_points":      len(graph.entry_points),
            "languages":         lang_counts,
            "total_classes":     sum(len(n.classes)   for n in graph.nodes.values()),
            "total_functions":   sum(len(n.functions) for n in graph.nodes.values()),
            "most_imported":     self._most_imported(graph),
            "most_complex":      self._most_complex(graph),
        }

        logger.info(
            f"✅ Dependency analysis complete [{project_id}]: "
            f"{len(graph.nodes)} modules, "
            f"{len(graph.edges)} edges, "
            f"{len(graph.entry_points)} entry points"
        )

        return graph

    def generate_summary(self, graph: DependencyGraph) -> str:
        """
        Generates a human-readable architecture summary.
        Stored as a project memory so Gemini can use it when answering
        architecture questions.
        """
        lines = []
        s     = graph.stats

        lines.append(f"**Repository Architecture Summary**")
        lines.append("")
        lines.append(
            f"This project contains **{s['total_modules']} modules** "
            f"({', '.join(f'{v} {k}' for k, v in s.get('languages', {}).items())}) "
            f"with **{s['total_classes']} classes** and "
            f"**{s['total_functions']} functions**."
        )

        if graph.entry_points:
            ep_names = [Path(ep).name for ep in graph.entry_points[:5]]
            lines.append(
                f"\n**Entry Points:** {', '.join(ep_names)}"
            )

        if s.get("most_imported"):
            top = s["most_imported"][:3]
            lines.append(
                f"\n**Most depended-on modules:** "
                f"{', '.join(f'`{Path(m).name}`' for m in top)}"
            )

        if s.get("most_complex"):
            top = s["most_complex"][:3]
            lines.append(
                f"\n**Most complex modules (by definitions):** "
                f"{', '.join(f'`{Path(m).name}`' for m in top)}"
            )

        # Component groupings
        groups = self._detect_component_groups(graph)
        if groups:
            lines.append("\n**Component groups detected:**")
            for group_name, files in list(groups.items())[:6]:
                lines.append(f"- `{group_name}/`: {len(files)} files")

        return "\n".join(lines)

    def generate_mermaid(
        self,
        graph:     DependencyGraph,
        max_nodes: int = 20,
        max_edges: int = 30,
    ) -> str:
        """
        Generates a Mermaid diagram of the dependency graph.

        Limits nodes/edges for readability — only shows the most
        important modules (entry points + most imported).
        """
        # Select the most important nodes
        important = set(graph.entry_points[:5])
        important.update(graph.stats.get("most_imported", [])[:5])
        important.update(graph.stats.get("most_complex",  [])[:5])

        # Fill up to max_nodes from remaining nodes
        remaining = [fp for fp in graph.nodes if fp not in important]
        selected  = important | set(remaining[:max(0, max_nodes - len(important))])

        # Build node labels (short names)
        def node_id(fp: str) -> str:
            return re.sub(r'[^a-zA-Z0-9]', '_', Path(fp).stem)[:20]

        lines = ["graph TD"]

        # Add nodes with styling
        for fp in selected:
            node  = graph.nodes.get(fp)
            if not node:
                continue
            nid   = node_id(fp)
            label = Path(fp).stem
            if fp in graph.entry_points:
                lines.append(f'    {nid}["{label}"] :::entry')
            elif node.classes:
                lines.append(f'    {nid}["{label}"] :::service')
            else:
                lines.append(f'    {nid}("{label}")')

        # Add edges (only between selected nodes)
        edge_count = 0
        for edge in graph.edges:
            if edge.source not in selected or edge.target not in selected:
                continue
            if edge_count >= max_edges:
                break
            src = node_id(edge.source)
            tgt = node_id(edge.target)
            if edge.edge_type == "inheritance":
                lines.append(f"    {tgt} <|-- {src}")
            else:
                lines.append(f"    {src} --> {tgt}")
            edge_count += 1

        # Styling
        lines.append("    classDef entry fill:#6366f1,color:#fff,stroke:#4f46e5")
        lines.append("    classDef service fill:#1e293b,color:#94a3b8,stroke:#334155")

        return "\n".join(lines)

    def _extract_node(self, pf: ParsedFile) -> ModuleNode:
        """Routes to the correct extractor for the file's language."""
        if pf.language == "python":
            return self._python_extractor.extract(pf)
        if pf.language in ("javascript", "typescript"):
            return self._js_extractor.extract(pf)
        return self._generic.extract(pf)

    def _resolve_import(
        self,
        import_name: str,
        source_file: str,
        module_map:  dict[str, str],
    ) -> Optional[str]:
        """
        Tries to resolve an import name to a file path.
        Returns None if it's an external library (not in this project).
        """
        # Direct module map lookup
        if import_name in module_map:
            return module_map[import_name]

        # Partial match (e.g. "app.auth" matches "app/auth/__init__.py")
        for module_name, file_path in module_map.items():
            if module_name.startswith(import_name) or import_name.startswith(module_name):
                return file_path

        # Relative JS/TS imports (e.g. "./utils" → "src/utils.js")
        if import_name.startswith("."):
            source_dir = str(Path(source_file).parent)
            candidates = [
                f"{source_dir}/{import_name.lstrip('./')}.py",
                f"{source_dir}/{import_name.lstrip('./')}.js",
                f"{source_dir}/{import_name.lstrip('./')}.ts",
                f"{source_dir}/{import_name.lstrip('./')}/index.js",
            ]
            for c in candidates:
                normalized = c.replace("\\", "/").replace("//", "/")
                if normalized in module_map.values():
                    return normalized

        return None

    def _resolve_name(
        self,
        name:    str,
        nodes:   dict[str, ModuleNode],
    ) -> Optional[str]:
        """Tries to find which file defines a class by name."""
        for fp, node in nodes.items():
            if name in node.classes:
                return fp
        return None

    def _most_imported(self, graph: DependencyGraph) -> list[str]:
        """Returns file paths sorted by how many other files import them."""
        import_counts: dict[str, int] = {}
        for edge in graph.edges:
            if edge.edge_type == "import":
                import_counts[edge.target] = import_counts.get(edge.target, 0) + 1
        return sorted(import_counts, key=import_counts.get, reverse=True)[:5]

    def _most_complex(self, graph: DependencyGraph) -> list[str]:
        """Returns file paths sorted by number of definitions (classes + functions)."""
        def complexity(fp: str) -> int:
            n = graph.nodes.get(fp)
            return len(n.classes) + len(n.functions) if n else 0
        return sorted(graph.nodes, key=complexity, reverse=True)[:5]

    def _detect_component_groups(
        self,
        graph: DependencyGraph,
    ) -> dict[str, list[str]]:
        """Groups files by their directory for component detection."""
        groups: dict[str, list[str]] = {}
        for fp in graph.nodes:
            parts = fp.replace("\\", "/").split("/")
            if len(parts) >= 2:
                group = parts[-2]   # Parent directory
                groups.setdefault(group, []).append(fp)
        # Only return groups with 2+ files
        return {k: v for k, v in groups.items() if len(v) >= 2}


# Module-level singleton
dependency_analyzer = DependencyAnalyzer()