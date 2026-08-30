# backend/app/core/processing/ast_chunker.py
#
# IMPROVEMENT 2: AST-Based Function/Class Chunking
#
# Why AST over regex?
#   Regex can't reliably detect where a function ENDS — it finds the
#   opening line but not the closing brace/dedent. AST parsing gives
#   us exact start line, end line, decorators, docstrings, and nested
#   classes/methods — producing complete, syntactically valid chunks.
#
# What "complete logical unit" means:
#   BEFORE (regex/token chunker):
#     Chunk 1: lines 1-50  (half of class UserService)
#     Chunk 2: lines 51-100 (rest of class + start of AuthService)
#
#   AFTER (AST chunker):
#     Chunk 1: full UserService class (lines 1-48)
#     Chunk 2: full AuthService class (lines 50-99)
#     Chunk 3: full authenticate() function (lines 101-118)
#
# Language support:
#   Python     → ast (stdlib, always available)
#   JavaScript → custom regex-AST hybrid (no native Python JS parser)
#   TypeScript → same as JS (TypeScript is a superset)
#   Java       → custom regex-AST hybrid
#   Go         → custom regex-AST hybrid
#   All others → FunctionAwareChunker fallback (Phase 7 original)
#
# Metadata stored per chunk:
#   function_name, class_name, start_line, end_line,
#   language, file_path, chunk_type, decorators, is_method,
#   parent_class (for methods), return_type (when detectable)

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing      import Optional

from app.models.ingest_models import ParsedFile, CodeChunk
from app.utils.logger          import get_logger

logger = get_logger(__name__)


# ── Chunk size limits ──────────────────────────────────────────────────────────

MIN_AST_CHUNK_CHARS = 10     # Shorter than regex chunker — AST is more precise
MAX_AST_CHUNK_CHARS = 4000   # Allow larger chunks (complete class bodies)
FALLBACK_CHUNK_SIZE = 1500   # Used when splitting oversized AST chunks


# ── Rich metadata dataclass ────────────────────────────────────────────────────

@dataclass
class ASTChunkMetadata:
    """
    Rich metadata extracted from the AST for a single code chunk.
    Stored alongside the chunk text in ChromaDB metadata fields.
    """
    chunk_type:    str           # "function", "class", "method", "module"
    name:          str           # Function/class/method name
    parent_class:  str  = ""     # Set when chunk_type == "method"
    decorators:    list = field(default_factory=list)  # e.g. ["staticmethod"]
    is_async:      bool = False  # True for async def
    is_method:     bool = False  # True if inside a class
    docstring:     str  = ""     # First docstring if present
    return_type:   str  = ""     # Return type annotation (when visible)
    start_line:    int  = 0
    end_line:      int  = 0


# ── Python AST Parser ──────────────────────────────────────────────────────────

class PythonASTChunker:
    """
    Extracts function and class definitions from Python source using
    the built-in `ast` module.

    Gives us:
    - Exact start/end lines for every function, class, method
    - Decorator names
    - Async detection
    - Docstring extraction
    - Nested class/method flattening (each method becomes its own chunk)

    Fallback: if ast.parse() fails (syntax error), falls back to
    the Phase 7 FunctionAwareChunker.
    """

    def chunk(self, parsed_file: ParsedFile) -> list[CodeChunk]:
        """
        Parses Python source with ast.parse() and extracts all
        top-level and nested functions/classes as individual chunks.
        """
        content = parsed_file.content
        lines   = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.debug(
                f"ast.parse() failed for {parsed_file.file_path}: {e} "
                f"— falling back to regex chunker"
            )
            return []   # Signal to caller to use fallback

        metadata_list = self._extract_definitions(tree, lines, parent_class="")
        return self._metadata_to_chunks(metadata_list, lines, parsed_file)

    def _extract_definitions(
        self,
        tree:         ast.AST,
        lines:        list[str],
        parent_class: str = "",
    ) -> list[ASTChunkMetadata]:
        """
        Walks the AST and collects all function/class definitions.
        Recurses into class bodies to collect methods.
        """
        results = []

        for node in ast.walk(tree):

            # ── Class definition ───────────────────────────────────────────
            if isinstance(node, ast.ClassDef):
                # The class itself as one chunk (header + body if small)
                meta = ASTChunkMetadata(
                    chunk_type   = "class",
                    name         = node.name,
                    parent_class = parent_class,
                    decorators   = self._get_decorators(node),
                    is_method    = bool(parent_class),
                    docstring    = self._get_docstring(node),
                    start_line   = node.lineno,
                    end_line     = self._get_end_line(node, lines),
                )
                results.append(meta)

                # Recurse into the class body to extract methods
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_meta = ASTChunkMetadata(
                            chunk_type   = "method",
                            name         = child.name,
                            parent_class = node.name,
                            decorators   = self._get_decorators(child),
                            is_async     = isinstance(child, ast.AsyncFunctionDef),
                            is_method    = True,
                            docstring    = self._get_docstring(child),
                            return_type  = self._get_return_type(child),
                            start_line   = child.lineno,
                            end_line     = self._get_end_line(child, lines),
                        )
                        results.append(method_meta)

            # ── Top-level function definition ──────────────────────────────
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip if it's a method (we already collected those above)
                if parent_class:
                    continue

                # Check it's actually top-level (not nested inside a class)
                # by verifying no class is its parent in the tree
                # (We handle this via the class recursion above, so here
                # we only want functions that appear at module level)
                meta = ASTChunkMetadata(
                    chunk_type   = "function",
                    name         = node.name,
                    parent_class = "",
                    decorators   = self._get_decorators(node),
                    is_async     = isinstance(node, ast.AsyncFunctionDef),
                    is_method    = False,
                    docstring    = self._get_docstring(node),
                    return_type  = self._get_return_type(node),
                    start_line   = node.lineno,
                    end_line     = self._get_end_line(node, lines),
                )
                results.append(meta)

        # Remove duplicates (ast.walk visits all nodes including nested ones)
        # We de-duplicate by (name, start_line)
        seen  = set()
        dedup = []
        for m in results:
            key = (m.name, m.start_line)
            if key not in seen:
                seen.add(key)
                dedup.append(m)

        # Sort by start line
        dedup.sort(key=lambda m: m.start_line)
        return dedup

    def _get_end_line(self, node: ast.AST, lines: list[str]) -> int:
        """
        Gets the last line of an AST node.

        Python 3.8+: ast nodes have end_lineno attribute.
        Python 3.7: fall back to scanning forward for next dedent.
        """
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno

        # Fallback for Python 3.7: scan forward for next non-indented line
        start = node.lineno - 1   # 0-indexed
        if start >= len(lines):
            return node.lineno

        # Get the indentation level of the definition line
        indent = len(lines[start]) - len(lines[start].lstrip())

        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue    # Skip blank lines
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent <= indent:
                return i    # 1-indexed end line

        return len(lines)

    def _get_decorators(self, node: ast.AST) -> list[str]:
        """Extracts decorator names from a function or class node."""
        decorators = []
        for dec in getattr(node, 'decorator_list', []):
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{dec.value.id}.{dec.attr}"
                                  if isinstance(dec.value, ast.Name)
                                  else dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
        return decorators

    def _get_docstring(self, node: ast.AST) -> str:
        """Extracts the docstring of a function or class (first 200 chars)."""
        try:
            return ast.get_docstring(node)[:200] or ""
        except Exception:
            return ""

    def _get_return_type(self, node: ast.AST) -> str:
        """Extracts the return type annotation if present."""
        returns = getattr(node, 'returns', None)
        if returns is None:
            return ""
        try:
            return ast.unparse(returns)
        except Exception:
            return ""

    def _metadata_to_chunks(
        self,
        metadata_list: list[ASTChunkMetadata],
        lines:         list[str],
        parsed_file:   ParsedFile,
    ) -> list[CodeChunk]:
        """
        Converts ASTChunkMetadata objects into CodeChunk objects.

        For each definition:
        1. Extract lines[start_line-1 : end_line]
        2. If the chunk is too large, split it with sliding window
        3. Set chunk_type, function_name, class_name from metadata
        """
        from app.core.processing.chunker import (
            SlidingWindowChunker, make_chunk_id,
            MIN_CHUNK_SIZE, _token_counter,
        )
        sliding = SlidingWindowChunker()
        chunks  = []

        for idx, meta in enumerate(metadata_list):
            # Extract the exact lines for this definition
            start = max(0, meta.start_line - 1)   # Convert to 0-indexed
            end   = min(len(lines), meta.end_line) # Already inclusive
            chunk_lines = lines[start:end]
            text = textwrap.dedent("\n".join(chunk_lines)).strip()

            if len(text) < MIN_AST_CHUNK_CHARS:
                continue    # Skip trivially small chunks (__init__ with pass)

            # Determine function_name and class_name
            if meta.chunk_type in ("function", "method"):
                function_name = meta.name
                class_name    = meta.parent_class
            elif meta.chunk_type == "class":
                function_name = ""
                class_name    = meta.name
            else:
                function_name = ""
                class_name    = ""

            # Build the chunk
            chunk_id = make_chunk_id(
                parsed_file.project_id or "unknown",
                parsed_file.file_path,
                idx,
            )

            if len(text) <= MAX_AST_CHUNK_CHARS:
                # Normal case: chunk fits in one unit
                chunks.append(CodeChunk(
                    chunk_id      = chunk_id,
                    project_id    = parsed_file.project_id or "unknown",
                    chunk_index   = idx,
                    text          = text,
                    file_path     = parsed_file.file_path,
                    language      = parsed_file.language,
                    start_line    = meta.start_line,
                    end_line      = meta.end_line,
                    chunk_type    = meta.chunk_type,
                    function_name = function_name,
                    class_name    = class_name,
                    char_count    = len(text),
                    token_count   = _token_counter.count(text),
                ))
            else:
                # Large function/class: split with sliding window
                # but preserve type metadata on every sub-chunk
                logger.debug(
                    f"Large AST chunk ({len(text)} chars): "
                    f"{meta.name} in {parsed_file.file_path} — splitting"
                )
                sub_file = ParsedFile(
                    file_path     = parsed_file.file_path,
                    relative_path = parsed_file.relative_path,
                    content       = text,
                    language      = parsed_file.language,
                    project_id    = parsed_file.project_id,
                )
                sub_chunks = sliding.chunk(
                    sub_file,
                    chunk_size    = FALLBACK_CHUNK_SIZE,
                    chunk_overlap = 150,
                )
                for i, sc in enumerate(sub_chunks):
                    sc.chunk_index   = idx * 100 + i
                    sc.chunk_id      = make_chunk_id(
                        parsed_file.project_id or "unknown",
                        parsed_file.file_path,
                        sc.chunk_index,
                    )
                    sc.chunk_type    = meta.chunk_type
                    sc.function_name = function_name
                    sc.class_name    = class_name
                    sc.start_line    = meta.start_line + sc.start_line - 1
                    sc.end_line      = meta.start_line + sc.end_line - 1
                    chunks.append(sc)

        return chunks


# ── JavaScript / TypeScript AST-Style Chunker ─────────────────────────────────

class JavaScriptASTChunker:
    """
    Extracts functions and classes from JavaScript/TypeScript source.

    Python has no native JS/TS AST parser, so we use a carefully
    designed regex-brace-counting approach that's more accurate than
    the Phase 7 regex approach:

    Phase 7 approach: finds the START of a function (regex match),
      then takes the next N lines — may cut off mid-function.

    This approach: finds the START, then counts braces { } until we
      reach the matching closing brace — giving us the EXACT end line.

    Patterns detected:
      function foo() { ... }
      const foo = () => { ... }
      const foo = async () => { ... }
      class Foo { ... }
      export default function foo() { ... }
      export const foo = () => { ... }
    """

    # Patterns that start a new JS/TS definition
    START_PATTERNS = [
        # Named function declaration
        re.compile(
            r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?'
            r'function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(',
            re.MULTILINE,
        ),
        # Arrow function / expression
        re.compile(
            r'^(?:export\s+)?(?:const|let|var)\s+'
            r'([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*'
            r'(?:async\s*)?\(?.*?\)?\s*=>',
            re.MULTILINE,
        ),
        # Class declaration
        re.compile(
            r'^(?:export\s+)?(?:default\s+)?class\s+'
            r'([A-Za-z_$][A-Za-z0-9_$]*)',
            re.MULTILINE,
        ),
        # Method inside a class (indented)
        re.compile(
            r'^\s+(?:static\s+)?(?:async\s+)?'
            r'(?:get\s+|set\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\s*\(',
            re.MULTILINE,
        ),
    ]

    def chunk(self, parsed_file: ParsedFile) -> list[CodeChunk]:
        """Extracts JS/TS definitions using brace-counting."""
        content = parsed_file.content
        lines   = content.splitlines()

        definitions = self._find_definitions(lines)
        if not definitions:
            return []   # Signal fallback

        return self._defs_to_chunks(definitions, lines, parsed_file)

    def _find_definitions(self, lines: list[str]) -> list[dict]:
        """
        Finds all function/class definitions and their start/end lines.
        Uses brace-counting to find the exact end line.
        """
        content     = "\n".join(lines)
        definitions = []
        used_starts = set()   # Avoid overlapping definitions

        for pattern in self.START_PATTERNS:
            for match in pattern.finditer(content):
                start_line = content[:match.start()].count("\n")

                if start_line in used_starts:
                    continue

                name       = match.group(1) if match.lastindex else ""
                chunk_type = self._detect_type(lines[start_line] if start_line < len(lines) else "")
                end_line   = self._find_closing_brace(lines, start_line)

                if end_line is None:
                    continue   # Couldn't find end — skip

                text = "\n".join(lines[start_line:end_line + 1]).strip()
                if len(text) < MIN_AST_CHUNK_CHARS:
                    continue

                definitions.append({
                    "name":       name,
                    "chunk_type": chunk_type,
                    "start_line": start_line + 1,   # 1-indexed
                    "end_line":   end_line + 1,
                    "text":       text,
                })
                used_starts.add(start_line)

        definitions.sort(key=lambda d: d["start_line"])
        return definitions

    def _find_closing_brace(
        self,
        lines:      list[str],
        start_idx:  int,
    ) -> Optional[int]:
        """
        Counts { and } to find the line containing the matching closing brace.
        Returns the 0-indexed line number of the closing brace.
        """
        depth    = 0
        in_string = False
        string_char = ""

        for i in range(start_idx, min(start_idx + 500, len(lines))):
            line = lines[i]
            j    = 0
            while j < len(line):
                ch = line[j]

                # Handle string literals (skip braces inside strings)
                if in_string:
                    if ch == string_char and (j == 0 or line[j-1] != '\\'):
                        in_string = False
                    j += 1
                    continue

                if ch in ('"', "'", '`'):
                    in_string   = True
                    string_char = ch
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return i

                j += 1

        return None   # No matching brace found

    def _detect_type(self, line: str) -> str:
        """Detects whether a line starts a function, class, or method."""
        stripped = line.strip()
        if re.search(r'\bclass\b', stripped):
            return "class"
        if re.search(r'\bfunction\b|\=>\s*\{|=\s*(?:async\s*)?\(', stripped):
            return "function"
        if re.match(r'\s+\w+\s*\(', line):
            return "method"
        return "function"

    def _defs_to_chunks(
        self,
        definitions: list[dict],
        lines:       list[str],
        parsed_file: ParsedFile,
    ) -> list[CodeChunk]:
        from app.core.processing.chunker import make_chunk_id, _token_counter
        chunks = []
        for idx, defn in enumerate(definitions):
            text      = defn["text"]
            chunk_id  = make_chunk_id(
                parsed_file.project_id or "unknown",
                parsed_file.file_path,
                idx,
            )
            chunk_type = defn["chunk_type"]
            chunks.append(CodeChunk(
                chunk_id      = chunk_id,
                project_id    = parsed_file.project_id or "unknown",
                chunk_index   = idx,
                text          = text,
                file_path     = parsed_file.file_path,
                language      = parsed_file.language,
                start_line    = defn["start_line"],
                end_line      = defn["end_line"],
                chunk_type    = chunk_type,
                function_name = defn["name"] if chunk_type in ("function", "method") else "",
                class_name    = defn["name"] if chunk_type == "class" else "",
                char_count    = len(text),
                token_count   = _token_counter.count(text),
            ))
        return chunks


# ── Java AST-Style Chunker ─────────────────────────────────────────────────────

class JavaASTChunker:
    """
    Extracts methods, classes, and interfaces from Java source.
    Uses brace-counting for exact end-line detection.

    Patterns:
      public class Foo { ... }
      public interface Foo { ... }
      public void authenticate(String username) { ... }
      private static String createToken(User user) { ... }
    """

    # Java definition patterns (class/interface/method declarations)
    JAVA_PATTERNS = [
        # Class / interface / enum
        re.compile(
            r'^(?:(?:public|private|protected|abstract|final|static)\s+)*'
            r'(?:class|interface|enum|@interface)\s+'
            r'([A-Z][A-Za-z0-9_]*)',
            re.MULTILINE,
        ),
        # Method declaration (returns a type, has parens)
        re.compile(
            r'^\s+(?:(?:public|private|protected|static|final|abstract|'
            r'synchronized|native|default)\s+)*'
            r'(?:<[^>]+>\s+)?'                         # generic return type
            r'(?:void|int|String|boolean|long|double|float|byte|char|'
            r'[A-Z][A-Za-z0-9_<>\[\]]*)\s+'
            r'([a-z][A-Za-z0-9_]*)\s*\(',
            re.MULTILINE,
        ),
    ]

    def chunk(self, parsed_file: ParsedFile) -> list[CodeChunk]:
        content = parsed_file.content
        lines   = content.splitlines()

        definitions = self._find_java_definitions(content, lines)
        if not definitions:
            return []

        return self._defs_to_chunks(definitions, lines, parsed_file)

    def _find_java_definitions(
        self,
        content: str,
        lines:   list[str],
    ) -> list[dict]:
        definitions = []
        used_starts = set()

        for pattern in self.JAVA_PATTERNS:
            for match in pattern.finditer(content):
                start_line = content[:match.start()].count("\n")
                if start_line in used_starts:
                    continue

                name       = match.group(1) if match.lastindex else ""
                line_text  = lines[start_line] if start_line < len(lines) else ""
                chunk_type = "class" if re.search(r'\b(?:class|interface|enum)\b', line_text) else "method"
                end_line   = self._find_closing_brace(lines, start_line)

                if end_line is None:
                    continue

                text = "\n".join(lines[start_line:end_line + 1]).strip()
                if len(text) < MIN_AST_CHUNK_CHARS:
                    continue

                definitions.append({
                    "name":       name,
                    "chunk_type": chunk_type,
                    "start_line": start_line + 1,
                    "end_line":   end_line + 1,
                    "text":       text,
                })
                used_starts.add(start_line)

        definitions.sort(key=lambda d: d["start_line"])
        return definitions

    def _find_closing_brace(self, lines: list[str], start_idx: int) -> Optional[int]:
        """Same brace-counting logic as the JS chunker."""
        depth = 0
        for i in range(start_idx, min(start_idx + 600, len(lines))):
            for ch in lines[i]:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return i
        return None

    def _defs_to_chunks(
        self,
        definitions: list[dict],
        lines:       list[str],
        parsed_file: ParsedFile,
    ) -> list[CodeChunk]:
        from app.core.processing.chunker import make_chunk_id, _token_counter
        chunks = []
        for idx, defn in enumerate(definitions):
            text      = defn["text"]
            chunk_type = defn["chunk_type"]
            chunk_id  = make_chunk_id(
                parsed_file.project_id or "unknown",
                parsed_file.file_path,
                idx,
            )
            chunks.append(CodeChunk(
                chunk_id      = chunk_id,
                project_id    = parsed_file.project_id or "unknown",
                chunk_index   = idx,
                text          = text,
                file_path     = parsed_file.file_path,
                language      = parsed_file.language,
                start_line    = defn["start_line"],
                end_line      = defn["end_line"],
                chunk_type    = chunk_type,
                function_name = defn["name"] if chunk_type == "method" else "",
                class_name    = defn["name"] if chunk_type == "class"  else "",
                char_count    = len(text),
                token_count   = _token_counter.count(text),
            ))
        return chunks


# ── Go AST-Style Chunker ───────────────────────────────────────────────────────

class GoASTChunker:
    """
    Extracts functions, methods, and structs from Go source.
    Uses brace-counting for exact end-line detection.

    Patterns:
      func Foo() string { ... }
      func (r *Receiver) Foo() { ... }
      type Foo struct { ... }
      type Foo interface { ... }
    """

    GO_PATTERNS = [
        re.compile(
            r'^func\s+(?:\([^)]+\)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(',
            re.MULTILINE,
        ),
        re.compile(
            r'^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)',
            re.MULTILINE,
        ),
    ]

    def chunk(self, parsed_file: ParsedFile) -> list[CodeChunk]:
        content = parsed_file.content
        lines   = content.splitlines()
        defs    = self._find_go_definitions(content, lines)
        if not defs:
            return []
        return self._defs_to_chunks(defs, lines, parsed_file)

    def _find_go_definitions(self, content: str, lines: list[str]) -> list[dict]:
        definitions = []
        used_starts = set()

        for pattern in self.GO_PATTERNS:
            for match in pattern.finditer(content):
                start_line = content[:match.start()].count("\n")
                if start_line in used_starts:
                    continue

                name      = match.group(1)
                line_text = lines[start_line] if start_line < len(lines) else ""
                chunk_type = "class" if "struct" in line_text or "interface" in line_text else "function"
                end_line   = self._find_closing_brace(lines, start_line)

                if end_line is None:
                    continue

                text = "\n".join(lines[start_line:end_line + 1]).strip()
                if not text.strip():
                    continue

                definitions.append({
                    "name":       name,
                    "chunk_type": chunk_type,
                    "start_line": start_line + 1,
                    "end_line":   end_line + 1,
                    "text":       text,
                })
                used_starts.add(start_line)

        definitions.sort(key=lambda d: d["start_line"])
        return definitions

    def _find_closing_brace(self, lines: list[str], start_idx: int) -> Optional[int]:
        depth = 0
        for i in range(start_idx, min(start_idx + 600, len(lines))):
            for ch in lines[i]:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return i
        return None

    def _defs_to_chunks(
        self,
        definitions: list[dict],
        lines:       list[str],
        parsed_file: ParsedFile,
    ) -> list[CodeChunk]:
        from app.core.processing.chunker import make_chunk_id, _token_counter
        chunks = []
        for idx, defn in enumerate(definitions):
            text       = defn["text"]
            chunk_type = defn["chunk_type"]
            chunk_id   = make_chunk_id(
                parsed_file.project_id or "unknown",
                parsed_file.file_path,
                idx,
            )
            chunks.append(CodeChunk(
                chunk_id      = chunk_id,
                project_id    = parsed_file.project_id or "unknown",
                chunk_index   = idx,
                text          = text,
                file_path     = parsed_file.file_path,
                language      = parsed_file.language,
                start_line    = defn["start_line"],
                end_line      = defn["end_line"],
                chunk_type    = chunk_type,
                function_name = defn["name"] if chunk_type == "function" else "",
                class_name    = defn["name"] if chunk_type == "class"    else "",
                char_count    = len(text),
                token_count   = _token_counter.count(text),
            ))
        return chunks


# ── AST-Aware Chunking Engine ──────────────────────────────────────────────────

class ASTChunkingEngine:
    """
    Orchestrates AST-based chunking across all supported languages.

    Priority:
    1. Try AST parser for the specific language
    2. If AST returns empty (parse error / no definitions), fall back
       to Phase 7 FunctionAwareChunker

    This means:
    - Python: always uses ast.parse() (exact lines, decorators, docstrings)
    - JS/TS:  uses brace-counting (exact end lines)
    - Java:   uses brace-counting
    - Go:     uses brace-counting
    - Others: Phase 7 FunctionAwareChunker (regex approach)
    - Markdown: Phase 7 MarkdownChunker
    - Config/SQL: Phase 7 SlidingWindowChunker

    The API is identical to ChunkingEngine — you can swap it in
    directly without changing any calling code.
    """

    def __init__(
        self,
        chunk_size:    int  = 1500,
        chunk_overlap: int  = 150,
        prefer_ast:    bool = True,  # Set False to disable AST and use Phase 7
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prefer_ast    = prefer_ast

        # AST parsers (per language)
        self._python_chunker = PythonASTChunker()
        self._js_chunker     = JavaScriptASTChunker()
        self._java_chunker   = JavaASTChunker()
        self._go_chunker     = GoASTChunker()

        # Phase 7 fallback chunkers
        from app.core.processing.chunker import (
            FunctionAwareChunker,
            MarkdownChunker,
            SlidingWindowChunker,
        )
        self._regex_chunker    = FunctionAwareChunker()
        self._markdown_chunker = MarkdownChunker()
        self._sliding_chunker  = SlidingWindowChunker()

    def chunk_file(self, parsed_file: ParsedFile) -> list[CodeChunk]:
        """
        Chunks a single ParsedFile using the best available strategy.

        Returns:
            List of CodeChunk objects with complete metadata.
        """
        language = parsed_file.language
        chunks   = []

        if self.prefer_ast:
            chunks = self._try_ast_chunk(parsed_file, language)

        if not chunks:
            # Fallback to Phase 7 chunkers
            chunks = self._fallback_chunk(parsed_file, language)

        # Final size validation
        from app.core.processing.chunker import MIN_CHUNK_SIZE
        return [c for c in chunks if c.text.strip()]

    def _try_ast_chunk(
        self,
        parsed_file: ParsedFile,
        language:    str,
    ) -> list[CodeChunk]:
        """Attempts AST-based chunking; returns [] to signal fallback."""
        try:
            if language == "python":
                return self._python_chunker.chunk(parsed_file)

            if language in ("javascript", "typescript"):
                return self._js_chunker.chunk(parsed_file)

            if language == "java":
                return self._java_chunker.chunk(parsed_file)

            if language == "go":
                return self._go_chunker.chunk(parsed_file)

        except Exception as e:
            logger.warning(
                f"AST chunking failed for {parsed_file.file_path}: {e} "
                f"— using fallback"
            )

        return []   # Signal: use fallback

    def _fallback_chunk(
        self,
        parsed_file: ParsedFile,
        language:    str,
    ) -> list[CodeChunk]:
        """Delegates to Phase 7 chunkers."""
        if language in ("markdown", "rst"):
            return self._markdown_chunker.chunk(
                parsed_file, self.chunk_size, self.chunk_overlap
            )

        # Try the Phase 7 FunctionAwareChunker first
        # (it supports many languages via regex)
        from app.core.processing.chunker import FUNCTION_AWARE_LANGUAGES
        if language in FUNCTION_AWARE_LANGUAGES:
            chunks = self._regex_chunker.chunk(
                parsed_file, self.chunk_size, self.chunk_overlap
            )
            if chunks:
                return chunks

        # Final fallback: sliding window
        return self._sliding_chunker.chunk(
            parsed_file, self.chunk_size, self.chunk_overlap
        )

    def chunk_all(
        self,
        parsed_files: list[ParsedFile],
        project_id:   str,
    ):
        """
        Chunks all files. Returns a ChunkingResult (same type as Phase 7).
        Fully compatible with the existing pipeline.
        """
        from app.core.processing.chunker import ChunkingResult

        result = ChunkingResult(project_id=project_id)
        chunks_by_lang: dict[str, int] = {}

        logger.info(
            f"🔪 AST-chunking {len(parsed_files)} files for {project_id}"
        )

        for parsed_file in parsed_files:
            parsed_file.project_id = project_id
            file_chunks = self.chunk_file(parsed_file)

            if not file_chunks:
                result.skipped_files += 1
                continue

            result.chunks.extend(file_chunks)
            result.total_files += 1

            lang = parsed_file.language
            chunks_by_lang[lang] = chunks_by_lang.get(lang, 0) + len(file_chunks)

            logger.debug(
                f"  {parsed_file.file_path}: "
                f"{len(file_chunks)} AST chunks ({language_strategy(lang)})"
            )

        result.total_chunks       = len(result.chunks)
        result.chunks_by_language = chunks_by_lang

        logger.info(
            f"✅ AST chunking complete: {result.total_chunks} chunks "
            f"from {result.total_files} files"
        )
        for lang, count in sorted(
            chunks_by_lang.items(), key=lambda x: x[1], reverse=True
        ):
            strategy = language_strategy(lang)
            logger.info(f"   {lang:15s}: {count} chunks ({strategy})")

        return result


def language_strategy(language: str) -> str:
    """Returns which strategy will be used for a given language."""
    if language == "python":
        return "Python AST (ast.parse)"
    if language in ("javascript", "typescript"):
        return "JS brace-counting"
    if language == "java":
        return "Java brace-counting"
    if language == "go":
        return "Go brace-counting"
    if language in ("markdown", "rst"):
        return "MarkdownChunker"
    return "FunctionAwareChunker / SlidingWindow"


# ── Module-level singleton ─────────────────────────────────────────────────────

ast_chunking_engine = ASTChunkingEngine(
    chunk_size    = 1500,
    chunk_overlap = 150,
    prefer_ast    = True,
)