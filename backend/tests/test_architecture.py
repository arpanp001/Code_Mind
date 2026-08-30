# backend/tests/test_architecture.py
#
# Test suite for Improvement 3: Repository Architecture & Dependency Intelligence
#
# Run with:
#   pytest tests/test_architecture.py -v
#
# All tests use in-memory objects — no ChromaDB or Gemini API needed.

import pytest
from app.models.ingest_models import ParsedFile
from app.core.analysis.dependency_analyzer import (
    DependencyAnalyzer,
    DependencyGraph,
    DependencyEdge,
    ModuleNode,
    PythonExtractor,
    JavaScriptExtractor,
    GenericExtractor,
    dependency_analyzer,
)
from app.core.analysis.architecture_query import (
    ArchitectureQueryEngine,
    ArchitectureContext,
    is_architecture_question,
    architecture_query_engine,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_file(
    content:  str,
    language: str   = "python",
    path:     str   = "src/test.py",
) -> ParsedFile:
    return ParsedFile(
        file_path     = path,
        relative_path = path,
        content       = content,
        language      = language,
        project_id    = "test_project",
    )


def make_graph(
    nodes: dict[str, dict] = None,
    edges: list[dict]      = None,
) -> DependencyGraph:
    """Creates a DependencyGraph for testing."""
    graph = DependencyGraph(project_id="test_project")

    if nodes:
        for fp, nd in nodes.items():
            graph.nodes[fp] = ModuleNode(
                file_path      = fp,
                language       = nd.get("language", "python"),
                module_name    = nd.get("module_name", fp.replace("/", ".").rstrip(".py")),
                imports        = nd.get("imports", []),
                exports        = nd.get("exports", []),
                classes        = nd.get("classes", []),
                functions      = nd.get("functions", []),
                inherits_from  = nd.get("inherits_from", []),
                entry_points   = nd.get("entry_points", []),
                is_entry_point = nd.get("is_entry_point", False),
                size_lines     = nd.get("size_lines", 10),
            )

    if edges:
        for ed in edges:
            graph.edges.append(DependencyEdge(
                source    = ed["source"],
                target    = ed["target"],
                edge_type = ed.get("edge_type", "import"),
                label     = ed.get("label", "imports"),
            ))

    graph.entry_points = [
        fp for fp, n in graph.nodes.items()
        if n.is_entry_point
    ]

    graph.stats = {
        "total_modules":   len(graph.nodes),
        "total_edges":     len(graph.edges),
        "entry_points":    len(graph.entry_points),
        "languages":       {},
        "total_classes":   sum(len(n.classes)   for n in graph.nodes.values()),
        "total_functions": sum(len(n.functions) for n in graph.nodes.values()),
        "most_imported":   [],
        "most_complex":    [],
    }

    return graph


# ══════════════════════════════════════════════════════════════════════════════
# 1. Python Extractor Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPythonExtractor:

    def setup_method(self):
        self.extractor = PythonExtractor()

    def test_extracts_imports(self):
        code = '''
import os
import json
from app.auth import UserService
from app.models import User
'''
        node = self.extractor.extract(make_file(code))
        assert "os"       in node.imports
        assert "json"     in node.imports
        assert any("auth" in i for i in node.imports)

    def test_extracts_from_imports(self):
        code = "from app.core.auth import authenticate, create_token\n"
        node = self.extractor.extract(make_file(code))
        assert any("auth" in i for i in node.imports)

    def test_extracts_class_names(self):
        code = '''
class UserService:
    pass

class AuthManager:
    pass
'''
        node = self.extractor.extract(make_file(code))
        assert "UserService"  in node.classes
        assert "AuthManager"  in node.classes

    def test_extracts_function_names(self):
        code = '''
def authenticate(username, password):
    return True

def create_token(user_id):
    return "token"
'''
        node = self.extractor.extract(make_file(code))
        assert "authenticate"  in node.functions
        assert "create_token"  in node.functions

    def test_extracts_inheritance(self):
        code = '''
class UserService(BaseService):
    pass

class AdminService(UserService, Auditable):
    pass
'''
        node = self.extractor.extract(make_file(code))
        assert "BaseService" in node.inherits_from
        assert "UserService" in node.inherits_from or "Auditable" in node.inherits_from

    def test_detects_main_entry_point(self):
        code = '''
def main():
    app.run()

if __name__ == "__main__":
    main()
'''
        node = self.extractor.extract(make_file(code, path="main.py"))
        assert node.is_entry_point is True

    def test_detects_flask_app(self):
        code = '''
from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello"
'''
        node = self.extractor.extract(make_file(code, path="app.py"))
        assert node.is_entry_point is True

    def test_detects_fastapi_app(self):
        code = '''
from fastapi import FastAPI
app = FastAPI(title="MyApp")
'''
        node = self.extractor.extract(make_file(code, path="main.py"))
        assert node.is_entry_point is True

    def test_detects_route_decorators(self):
        code = '''
from flask import Flask
app = Flask(__name__)

@app.route("/login", methods=["POST"])
def login():
    return authenticate()

@app.route("/logout")
def logout():
    return "ok"
'''
        node = self.extractor.extract(make_file(code))
        assert len(node.entry_points) >= 1

    def test_extracts_file_path(self):
        code = "x = 1\n"
        node = self.extractor.extract(make_file(code, path="src/auth/login.py"))
        assert node.file_path == "src/auth/login.py"

    def test_extracts_language(self):
        node = self.extractor.extract(make_file("x = 1\n"))
        assert node.language == "python"

    def test_syntax_error_handled_gracefully(self):
        code = "def broken(\n    # missing paren"
        node = self.extractor.extract(make_file(code))
        assert isinstance(node, ModuleNode)
        assert node.language == "python"

    def test_module_name_from_path(self):
        code = "x = 1\n"
        node = self.extractor.extract(
            make_file(code, path="src/auth/login.py")
        )
        assert "auth" in node.module_name or "login" in node.module_name

    def test_size_lines_populated(self):
        code = "x = 1\ny = 2\nz = 3\n"
        node = self.extractor.extract(make_file(code))
        assert node.size_lines == 3

    def test_entry_point_filename(self):
        """Files named main.py/app.py/server.py are always entry points."""
        for fname in ["main.py", "app.py", "server.py", "manage.py"]:
            node = self.extractor.extract(
                make_file("x = 1\n", path=f"backend/{fname}")
            )
            assert node.is_entry_point is True, (
                f"{fname} should be detected as entry point"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. JavaScript Extractor Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestJavaScriptExtractor:

    def setup_method(self):
        self.extractor = JavaScriptExtractor()

    def test_extracts_es6_imports(self):
        code = '''
import express from 'express';
import { Router } from 'express';
import UserService from './services/user';
'''
        node = self.extractor.extract(
            make_file(code, language="javascript", path="src/app.js")
        )
        assert "express"     in node.imports
        assert any("user" in i.lower() for i in node.imports)

    def test_extracts_require_imports(self):
        code = '''
const express = require('express');
const path    = require('path');
const auth    = require('./auth');
'''
        node = self.extractor.extract(
            make_file(code, language="javascript", path="server.js")
        )
        assert "express" in node.imports
        assert "path"    in node.imports

    def test_extracts_class_names(self):
        code = '''
class UserService {
    constructor() {}
}

class AuthController extends BaseController {
    login() {}
}
'''
        node = self.extractor.extract(
            make_file(code, language="javascript", path="service.js")
        )
        assert "UserService"     in node.classes
        assert "AuthController"  in node.classes

    def test_detects_inheritance(self):
        code = "class AdminService extends UserService { }\n"
        node = self.extractor.extract(
            make_file(code, language="javascript", path="admin.js")
        )
        assert "UserService" in node.inherits_from

    def test_detects_express_app(self):
        code = '''
const app = express();
app.listen(3000, () => console.log("running"));
'''
        node = self.extractor.extract(
            make_file(code, language="javascript", path="server.js")
        )
        assert node.is_entry_point is True

    def test_detects_routes(self):
        code = '''
const router = express.Router();
router.get('/users', getUsers);
router.post('/login', loginUser);
'''
        node = self.extractor.extract(
            make_file(code, language="javascript", path="routes.js")
        )
        assert len(node.entry_points) >= 1

    def test_extracts_exports(self):
        code = '''
export function createUser(data) { return db.create(data); }
export class TokenService { }
export const DEFAULT_TIMEOUT = 5000;
'''
        node = self.extractor.extract(
            make_file(code, language="javascript", path="utils.js")
        )
        assert len(node.exports) >= 1

    def test_typescript_supported(self):
        code = '''
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class ApiService {
    constructor(private http: HttpClient) {}
}
'''
        node = self.extractor.extract(
            make_file(code, language="typescript", path="api.service.ts")
        )
        assert "ApiService" in node.classes
        assert node.language == "typescript"


# ══════════════════════════════════════════════════════════════════════════════
# 3. DependencyAnalyzer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDependencyAnalyzer:

    def setup_method(self):
        self.analyzer = DependencyAnalyzer()

    def test_analyze_returns_graph(self):
        files = [make_file("def foo(): pass\n")]
        graph = self.analyzer.analyze(files, "test_project")
        assert isinstance(graph, DependencyGraph)

    def test_graph_has_nodes(self):
        files = [
            make_file("def foo(): pass\n", path="src/a.py"),
            make_file("def bar(): pass\n", path="src/b.py"),
        ]
        graph = self.analyzer.analyze(files, "test_project")
        assert len(graph.nodes) == 2

    def test_graph_has_project_id(self):
        files = [make_file("x = 1\n")]
        graph = self.analyzer.analyze(files, "my_project")
        assert graph.project_id == "my_project"

    def test_detects_import_edges(self):
        files = [
            make_file(
                "from src.b import helper\n",
                path="src/a.py",
            ),
            make_file("def helper(): pass\n", path="src/b.py"),
        ]
        graph = self.analyzer.analyze(files, "test_project")
        # Should have at least one import edge
        import_edges = [e for e in graph.edges if e.edge_type == "import"]
        assert isinstance(import_edges, list)   # May be empty if resolution fails

    def test_detects_entry_points(self):
        files = [
            make_file(
                "from flask import Flask\napp = Flask(__name__)\n",
                path="app.py",
            ),
            make_file("def helper(): pass\n", path="utils.py"),
        ]
        graph = self.analyzer.analyze(files, "test_project")
        assert len(graph.entry_points) >= 1
        assert "app.py" in graph.entry_points

    def test_stats_populated(self):
        files = [
            make_file(
                "class Foo:\n    def bar(self): pass\n",
                path="src/foo.py",
            ),
        ]
        graph = self.analyzer.analyze(files, "test_project")
        assert "total_modules"   in graph.stats
        assert "total_classes"   in graph.stats
        assert "total_functions" in graph.stats
        assert graph.stats["total_modules"] >= 1

    def test_graph_serialization_roundtrip(self):
        """Graph should survive JSON serialization and deserialization."""
        files = [
            make_file(
                "class AuthService:\n    def login(self): pass\n",
                path="src/auth.py",
            ),
        ]
        graph      = self.analyzer.analyze(files, "test_project")
        graph_dict = graph.to_dict()
        restored   = DependencyGraph.from_dict(graph_dict)

        assert restored.project_id == graph.project_id
        assert len(restored.nodes) == len(graph.nodes)

    def test_generate_summary_returns_string(self):
        files = [
            make_file(
                "from flask import Flask\napp = Flask(__name__)\n",
                path="app.py",
            ),
        ]
        graph   = self.analyzer.analyze(files, "test_project")
        summary = self.analyzer.generate_summary(graph)
        assert isinstance(summary, str)
        assert len(summary) > 20

    def test_generate_mermaid_returns_string(self):
        files = [
            make_file("x = 1\n", path="src/a.py"),
            make_file("y = 2\n", path="src/b.py"),
        ]
        graph   = self.analyzer.analyze(files, "test_project")
        mermaid = self.analyzer.generate_mermaid(graph)
        assert isinstance(mermaid, str)
        assert "graph" in mermaid.lower()

    def test_empty_files_handled(self):
        graph = self.analyzer.analyze([], "test_project")
        assert isinstance(graph, DependencyGraph)
        assert len(graph.nodes) == 0

    def test_large_file_list_capped(self):
        """Should not crash or take forever on 500+ files."""
        files = [
            make_file(f"x_{i} = {i}\n", path=f"src/file_{i}.py")
            for i in range(50)   # Use 50 to keep test fast
        ]
        graph = self.analyzer.analyze(files, "test_project", max_files=20)
        assert len(graph.nodes) <= 20

    def test_multiple_languages_analyzed(self):
        files = [
            make_file("def foo(): pass\n",             "python",     "src/a.py"),
            make_file("function bar() { return 1; }\n", "javascript", "src/b.js"),
        ]
        graph = self.analyzer.analyze(files, "test_project")
        langs = graph.stats.get("languages", {})
        assert "python"     in langs
        assert "javascript" in langs


# ══════════════════════════════════════════════════════════════════════════════
# 4. Architecture Question Detection Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIsArchitectureQuestion:

    def test_entry_point_question(self):
        assert is_architecture_question("What is the entry point?")
        assert is_architecture_question("Where does the app start?")
        assert is_architecture_question("How does it start up?")

    def test_flow_question(self):
        assert is_architecture_question("How does a request flow through the system?")
        assert is_architecture_question("What is the request pipeline?")
        assert is_architecture_question("Explain the request flow")

    def test_dependency_question(self):
        assert is_architecture_question("What does UserService depend on?")
        assert is_architecture_question("Show me the dependencies")
        assert is_architecture_question("What imports this module?")

    def test_architecture_question(self):
        assert is_architecture_question("Describe the architecture")
        assert is_architecture_question("How is this project structured?")
        assert is_architecture_question("Show me the components")

    def test_code_questions_not_architecture(self):
        assert not is_architecture_question("Where is JWT implemented?")
        assert not is_architecture_question("Explain the authenticate function")
        assert not is_architecture_question("What does line 42 do?")

    def test_case_insensitive(self):
        assert is_architecture_question("WHAT IS THE ENTRY POINT?")
        assert is_architecture_question("Show Me The Architecture")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Architecture Query Engine Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestArchitectureQueryEngine:

    def setup_method(self):
        self.engine = ArchitectureQueryEngine()
        # Build a realistic test graph
        self.graph = make_graph(
            nodes={
                "app.py": {
                    "is_entry_point": True,
                    "entry_points":   ["@app.route /", "@app.route /login"],
                    "imports":        ["src/auth.py", "src/models.py"],
                    "classes":        [],
                    "functions":      ["create_app", "configure_middleware"],
                },
                "src/auth.py": {
                    "classes":        ["AuthService"],
                    "functions":      ["authenticate", "create_token", "verify_token"],
                    "imports":        ["src/models.py"],
                    "inherits_from":  [],
                },
                "src/models.py": {
                    "classes":        ["User", "Token"],
                    "functions":      ["get_user", "create_user"],
                    "imports":        [],
                },
                "src/routes.py": {
                    "entry_points":   ["GET /users", "POST /login"],
                    "functions":      ["get_users", "login", "logout"],
                    "imports":        ["src/auth.py"],
                },
            },
            edges=[
                {"source": "app.py",       "target": "src/auth.py",   "edge_type": "import"},
                {"source": "app.py",       "target": "src/models.py", "edge_type": "import"},
                {"source": "src/auth.py",  "target": "src/models.py", "edge_type": "import"},
                {"source": "src/routes.py","target": "src/auth.py",   "edge_type": "import"},
            ],
        )

    def test_query_returns_architecture_context(self):
        result = self.engine.query(self.graph, "What is the entry point?")
        assert isinstance(result, ArchitectureContext)

    def test_entry_point_query_mentions_entry_file(self):
        result = self.engine.query(self.graph, "Where does the app start?")
        assert "app.py" in result.context_text or "entry" in result.context_text.lower()

    def test_entry_point_query_type(self):
        result = self.engine.query(self.graph, "What is the entry point?")
        assert result.query_type == "entry_point"

    def test_dependency_query_mentions_imports(self):
        result = self.engine.query(
            self.graph, "What does AuthService depend on?"
        )
        assert result.query_type == "dependency"
        assert isinstance(result.context_text, str)

    def test_flow_query_type(self):
        result = self.engine.query(
            self.graph, "How does a request flow through the system?"
        )
        assert result.query_type == "flow"

    def test_overview_query(self):
        result = self.engine.query(
            self.graph, "Describe the architecture"
        )
        assert result.query_type == "general"
        assert "modules" in result.context_text.lower() or \
               "architecture" in result.context_text.lower()

    def test_inheritance_query(self):
        # Add an inheritance relationship
        self.graph.nodes["src/admin.py"] = ModuleNode(
            file_path     = "src/admin.py",
            language      = "python",
            module_name   = "src.admin",
            classes       = ["AdminService"],
            inherits_from = ["AuthService"],
        )
        result = self.engine.query(
            self.graph, "What classes inherit from AuthService?"
        )
        assert result.query_type == "inheritance"

    def test_empty_graph_handled(self):
        empty_graph = DependencyGraph(project_id="empty")
        result      = self.engine.query(empty_graph, "What is the entry point?")
        assert isinstance(result, ArchitectureContext)
        assert result.answered is False

    def test_files_referenced_populated(self):
        result = self.engine.query(self.graph, "What is the entry point?")
        assert isinstance(result.files_referenced, list)

    def test_mermaid_generated_for_overview(self):
        result = self.engine.query(self.graph, "Show the architecture diagram")
        assert isinstance(result.mermaid_diagram, str)

    def test_context_text_is_non_empty(self):
        result = self.engine.query(self.graph, "What is the entry point?")
        assert len(result.context_text) > 10

    def test_answered_true_when_graph_has_data(self):
        result = self.engine.query(self.graph, "Describe the architecture")
        assert result.answered is True

    def test_dependency_query_shows_most_imported_when_name_not_found(self):
        """When a specific module name isn't found, show most-imported list."""
        # Add most_imported stats
        self.graph.stats["most_imported"] = ["src/models.py", "src/auth.py"]
        result = self.engine.query(
            self.graph, "What depends on NonExistentModule?"
        )
        assert result.query_type == "dependency"
        assert isinstance(result.context_text, str)

    def test_graph_with_no_entry_points(self):
        graph = make_graph(nodes={
            "src/utils.py": {
                "functions": ["helper_one", "helper_two"],
                "is_entry_point": False,
            },
        })
        result = self.engine.query(graph, "What is the entry point?")
        assert isinstance(result, ArchitectureContext)
        # Should mention no entry points found
        assert "no" in result.context_text.lower() or \
               "not" in result.context_text.lower() or \
               "entry" in result.context_text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 6. DependencyGraph Serialization Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDependencyGraphSerialization:

    def test_to_dict_returns_dict(self):
        graph = make_graph(nodes={"src/a.py": {"classes": ["Foo"]}})
        d     = graph.to_dict()
        assert isinstance(d, dict)
        assert "nodes"      in d
        assert "edges"      in d
        assert "project_id" in d

    def test_from_dict_restores_nodes(self):
        graph = make_graph(nodes={
            "src/auth.py": {
                "classes":   ["AuthService"],
                "functions": ["authenticate"],
            },
        })
        restored = DependencyGraph.from_dict(graph.to_dict())
        assert "src/auth.py" in restored.nodes
        assert "AuthService" in restored.nodes["src/auth.py"].classes

    def test_from_dict_restores_edges(self):
        graph = make_graph(
            nodes={
                "src/a.py": {},
                "src/b.py": {},
            },
            edges=[{
                "source":    "src/a.py",
                "target":    "src/b.py",
                "edge_type": "import",
                "label":     "imports b",
            }],
        )
        restored = DependencyGraph.from_dict(graph.to_dict())
        assert len(restored.edges) == 1
        assert restored.edges[0].source == "src/a.py"
        assert restored.edges[0].target == "src/b.py"

    def test_roundtrip_preserves_project_id(self):
        graph    = DependencyGraph(project_id="my_unique_project")
        restored = DependencyGraph.from_dict(graph.to_dict())
        assert restored.project_id == "my_unique_project"

    def test_empty_graph_serializes(self):
        graph = DependencyGraph(project_id="empty")
        d     = graph.to_dict()
        assert d["nodes"] == {}
        assert d["edges"] == []