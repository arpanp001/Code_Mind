# backend/app/core/processing/chunker.py
#
# IMPROVEMENT 2 INTEGRATION PATCH
#
# What changed vs Phase 7 original:
#   1. Added module-level exports that ast_chunker.py depends on:
#        - make_chunk_id()        → generates stable chunk IDs
#        - MIN_CHUNK_SIZE         → minimum chars for a valid chunk
#        - _token_counter         → shared token counting utility
#        - FUNCTION_AWARE_LANGUAGES → set of languages the regex chunker supports
#   2. ChunkingResult now also carries a `skipped_files` counter
#   3. The module-level `chunking_engine` singleton is unchanged —
#      existing pipeline code (ingest.py) still imports it the same way
#   4. A new `ast_chunking_engine` is exported from this module so
#      ingest.py can swap to it with a one-line change
#
# MIGRATION:
#   To switch the pipeline to AST chunking, in ingest.py change:
#
#     # OLD (Phase 7 regex chunker):
#     from app.core.processing.chunker import chunking_engine
#     chunking_result = chunking_engine.chunk_all(parsed_files, project_id)
#
#     # NEW (AST chunker with Phase 7 fallback):
#     from app.core.processing.chunker import ast_chunking_engine
#     chunking_result = ast_chunking_engine.chunk_all(parsed_files, project_id)
#
#   Everything else (the ChunkingResult structure, chunk fields,
#   how chunks are embedded and stored) is identical.

import hashlib
import re
import tiktoken
from dataclasses import dataclass, field
from typing      import Optional

from app.models.ingest_models import ParsedFile, CodeChunk
from app.config               import settings
from app.utils.logger         import get_logger

logger = get_logger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum character count for a chunk to be indexed.
# Shorter chunks (e.g. __init__(self): pass) add noise without value.
MIN_CHUNK_SIZE = 30

# Languages where FunctionAwareChunker (regex) can detect function boundaries.
# All others fall through to SlidingWindowChunker.
FUNCTION_AWARE_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "cpp",
    "c",
    "csharp",
    "kotlin",
    "swift",
    "ruby",
    "php",
    "scala",
}

# Languages that use indentation instead of braces
INDENTATION_LANGUAGES = {"python", "yaml", "coffeescript"}

# Languages that use markdown-style heading chunking
MARKDOWN_LANGUAGES = {"markdown", "rst", "mdx"}


# ── Token Counter ──────────────────────────────────────────────────────────────

class TokenCounter:
    """
    Shared utility for counting tokens in text.
    Uses tiktoken (cl100k_base) with char/4 fallback.
    Exposed as module-level `_token_counter` for ast_chunker.py to import.
    """

    def __init__(self):
        self._enc = None
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    def count(self, text: str) -> int:
        """Returns approximate token count for the given text."""
        if self._enc:
            try:
                return len(self._enc.encode(text, disallowed_special=()))
            except Exception:
                pass
        return max(1, len(text) // 4)


# Module-level singleton — imported by ast_chunker.py
_token_counter = TokenCounter()


# ── Chunk ID generation ────────────────────────────────────────────────────────

def make_chunk_id(project_id: str, file_path: str, chunk_index: int) -> str:
    """
    Generates a stable, unique chunk ID from project+file+index.

    Stable means the same function in the same file always gets the
    same ID — this lets us detect unchanged chunks on re-index without
    recomputing embeddings.

    Format: first 12 hex chars of SHA-256(project_id:file_path:index)
    """
    raw = f"{project_id}:{file_path}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class ChunkingResult:
    """
    Output from a chunking run over all files in a project.

    Compatible with both ChunkingEngine (Phase 7) and
    ASTChunkingEngine (Improvement 2) — both produce this structure.
    """
    project_id:         str
    chunks:             list[CodeChunk]  = field(default_factory=list)
    total_chunks:       int              = 0
    total_files:        int              = 0
    skipped_files:      int              = 0     # NEW: files with no valid chunks
    chunks_by_language: dict[str, int]   = field(default_factory=dict)
    errors:             list[str]        = field(default_factory=list)


# ── Sliding Window Chunker ─────────────────────────────────────────────────────

class SlidingWindowChunker:
    """
    Generic fallback chunker.
    Splits any text file into overlapping windows of approximately
    `chunk_size` characters. Used for SQL, YAML, config files, and
    any language where AST/regex chunking isn't available.
    """

    def chunk(
        self,
        parsed_file:  ParsedFile,
        chunk_size:   int = 1500,
        chunk_overlap: int = 150,
    ) -> list[CodeChunk]:
        """Splits content into overlapping character windows."""
        content = parsed_file.content
        if not content or not content.strip():
            return []

        lines    = content.splitlines()
        chunks   = []
        idx      = 0
        char_pos = 0

        while char_pos < len(content):
            chunk_text = content[char_pos:char_pos + chunk_size]
            if not chunk_text.strip():
                break

            # Calculate line numbers for this window
            start_char = char_pos
            end_char   = char_pos + len(chunk_text)
            start_line = content[:start_char].count("\n") + 1
            end_line   = content[:end_char].count("\n") + 1

            chunk_id = make_chunk_id(
                parsed_file.project_id or "unknown",
                parsed_file.file_path,
                idx,
            )
            chunks.append(CodeChunk(
                chunk_id      = chunk_id,
                project_id    = parsed_file.project_id or "unknown",
                chunk_index   = idx,
                text          = chunk_text.strip(),
                file_path     = parsed_file.file_path,
                language      = parsed_file.language,
                start_line    = start_line,
                end_line      = end_line,
                chunk_type    = "block",
                function_name = "",
                class_name    = "",
                char_count    = len(chunk_text),
                token_count   = _token_counter.count(chunk_text),
            ))
            idx      += 1
            char_pos += max(1, chunk_size - chunk_overlap)

        return chunks


# ── Markdown Chunker ───────────────────────────────────────────────────────────

class MarkdownChunker:
    """
    Splits Markdown/RST documents at heading boundaries.
    Each heading + its content becomes one chunk.

    Examples:
      # Authentication      → one chunk
      ## JWT Tokens         → one chunk
      ### Refresh Tokens    → one chunk
    """

    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def chunk(
        self,
        parsed_file:  ParsedFile,
        chunk_size:   int = 1500,
        chunk_overlap: int = 150,
    ) -> list[CodeChunk]:
        content = parsed_file.content
        if not content.strip():
            return []

        lines    = content.splitlines()
        chunks   = []
        sections = self._split_by_heading(content)

        for idx, (heading, section_content, start_line) in enumerate(sections):
            if not section_content.strip():
                continue

            chunk_id = make_chunk_id(
                parsed_file.project_id or "unknown",
                parsed_file.file_path,
                idx,
            )
            end_line = start_line + section_content.count("\n")
            chunks.append(CodeChunk(
                chunk_id      = chunk_id,
                project_id    = parsed_file.project_id or "unknown",
                chunk_index   = idx,
                text          = section_content.strip(),
                file_path     = parsed_file.file_path,
                language      = parsed_file.language,
                start_line    = start_line,
                end_line      = end_line,
                chunk_type    = "heading",
                function_name = "",
                class_name    = "",
                char_count    = len(section_content),
                token_count   = _token_counter.count(section_content),
            ))

        # If no headings found, fall back to sliding window
        if not chunks:
            return SlidingWindowChunker().chunk(
                parsed_file, chunk_size, chunk_overlap
            )

        return chunks

    def _split_by_heading(
        self,
        content: str,
    ) -> list[tuple[str, str, int]]:
        """
        Returns list of (heading_text, section_content, start_line_1indexed).
        """
        sections  = []
        matches   = list(self.HEADING_PATTERN.finditer(content))

        if not matches:
            return [("", content, 1)]

        for i, match in enumerate(matches):
            heading_text = match.group(0)
            start_pos    = match.start()
            end_pos      = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section      = content[start_pos:end_pos]
            start_line   = content[:start_pos].count("\n") + 1
            sections.append((heading_text, section, start_line))

        # Content before first heading (if any)
        if matches and matches[0].start() > 0:
            preamble = content[:matches[0].start()]
            if preamble.strip():
                sections.insert(0, ("", preamble, 1))

        return sections


# ── Function-Aware Chunker (Phase 7 regex approach) ───────────────────────────

class FunctionAwareChunker:
    """
    Phase 7 regex-based function chunker.

    Detects function/class definition lines with language-specific
    patterns, then uses a heuristic to find the end (next same-level
    definition or EOF).

    This is the fallback for:
    - Languages not supported by AST chunkers (Rust, C/C++, etc.)
    - When AST parsing fails (syntax errors, malformed files)
    - When ASTChunkingEngine.prefer_ast=False

    The AST chunker is strictly better for Python/JS/TS/Java/Go —
    this chunker exists only as a robust fallback.
    """

    # Language → regex that matches the START of a function/class
    PATTERNS: dict[str, re.Pattern] = {
        "python": re.compile(
            r'^[ \t]*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\('
            r'|^[ \t]*class\s+[A-Za-z_]\w*',
            re.MULTILINE,
        ),
        "javascript": re.compile(
            r'^(?:(?:export|async|function)\s+)*function\s+\w+'
            r'|^(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(?',
            re.MULTILINE,
        ),
        "typescript": re.compile(
            r'^(?:(?:export|async|function)\s+)*function\s+\w+'
            r'|^(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(?'
            r'|^(?:export\s+)?(?:abstract\s+)?class\s+\w+',
            re.MULTILINE,
        ),
        "java": re.compile(
            r'^\s*(?:public|private|protected|static|final|abstract|'
            r'synchronized)(?:\s+\w+){1,4}\s+\w+\s*\(',
            re.MULTILINE,
        ),
        "go": re.compile(
            r'^func\s+(?:\([^)]+\)\s+)?\w+\s*\('
            r'|^type\s+\w+\s+(?:struct|interface)',
            re.MULTILINE,
        ),
        "rust": re.compile(
            r'^(?:pub\s+)?(?:async\s+)?fn\s+\w+'
            r'|^(?:pub\s+)?(?:struct|enum|trait|impl)\s+\w+',
            re.MULTILINE,
        ),
        "cpp": re.compile(
            r'^(?:\w[\w\s\*&:<>]*)\s+\w+\s*\([^)]*\)\s*(?:const\s*)?\{?',
            re.MULTILINE,
        ),
        "csharp": re.compile(
            r'^\s*(?:public|private|protected|internal|static|virtual|'
            r'override|abstract|async)(?:\s+\w+){1,4}\s+\w+\s*[\(\{]',
            re.MULTILINE,
        ),
        "kotlin": re.compile(
            r'^(?:fun\s+\w+|class\s+\w+|object\s+\w+|interface\s+\w+)',
            re.MULTILINE,
        ),
        "swift": re.compile(
            r'^(?:func\s+\w+|class\s+\w+|struct\s+\w+|protocol\s+\w+|'
            r'extension\s+\w+)',
            re.MULTILINE,
        ),
        "ruby": re.compile(
            r'^[ \t]*def\s+\w+|^[ \t]*class\s+\w+|^[ \t]*module\s+\w+',
            re.MULTILINE,
        ),
        "php": re.compile(
            r'^[ \t]*(?:public|private|protected|static)?\s*function\s+\w+'
            r'|^[ \t]*class\s+\w+',
            re.MULTILINE,
        ),
        "scala": re.compile(
            r'^[ \t]*(?:def|class|object|trait|case class)\s+\w+',
            re.MULTILINE,
        ),
    }

    def chunk(
        self,
        parsed_file:  ParsedFile,
        chunk_size:   int = 1500,
        chunk_overlap: int = 150,
    ) -> list[CodeChunk]:
        language = parsed_file.language
        content  = parsed_file.content

        if not content or not content.strip():
            return []

        pattern = self.PATTERNS.get(language)
        if not pattern:
            # Language not supported — use sliding window
            return SlidingWindowChunker().chunk(
                parsed_file, chunk_size, chunk_overlap
            )

        lines   = content.splitlines()
        matches = list(pattern.finditer(content))

        if not matches:
            return SlidingWindowChunker().chunk(
                parsed_file, chunk_size, chunk_overlap
            )

        chunks = []
        for idx, (match, next_match) in enumerate(
            zip(matches, matches[1:] + [None])
        ):
            start_pos = match.start()
            end_pos   = next_match.start() if next_match else len(content)

            chunk_text = content[start_pos:end_pos].strip()
            if len(chunk_text) < MIN_CHUNK_SIZE:
                continue

            # Trim oversized chunks
            if len(chunk_text) > chunk_size * 3:
                chunk_text = chunk_text[:chunk_size * 3]

            start_line = content[:start_pos].count("\n") + 1
            end_line   = content[:end_pos].count("\n") + 1

            # Try to detect function/class name from first line
            first_line = lines[start_line - 1] if start_line <= len(lines) else ""
            func_name, class_name, chunk_type = self._extract_name(
                first_line, language
            )

            chunk_id = make_chunk_id(
                parsed_file.project_id or "unknown",
                parsed_file.file_path,
                idx,
            )
            chunks.append(CodeChunk(
                chunk_id      = chunk_id,
                project_id    = parsed_file.project_id or "unknown",
                chunk_index   = idx,
                text          = chunk_text,
                file_path     = parsed_file.file_path,
                language      = language,
                start_line    = start_line,
                end_line      = end_line,
                chunk_type    = chunk_type,
                function_name = func_name,
                class_name    = class_name,
                char_count    = len(chunk_text),
                token_count   = _token_counter.count(chunk_text),
            ))

        return chunks

    def _extract_name(
        self,
        line:     str,
        language: str,
    ) -> tuple[str, str, str]:
        """
        Extracts (function_name, class_name, chunk_type) from a
        definition line using language-specific patterns.
        Returns ("", "", "block") if detection fails.
        """
        stripped = line.strip()

        # Class detection
        class_match = re.search(
            r'\b(?:class|interface|struct|enum|trait|protocol|object)\s+'
            r'([A-Za-z_]\w*)',
            stripped,
        )
        if class_match:
            return "", class_match.group(1), "class"

        # Function/method detection
        func_match = re.search(
            r'\b(?:def|func|function|fn)\s+([A-Za-z_]\w*)',
            stripped,
        )
        if func_match:
            return func_match.group(1), "", "function"

        # Generic detection (const foo = / public void foo)
        generic_match = re.search(
            r'(?:const|let|var|public|private|protected)\s+'
            r'([A-Za-z_]\w*)',
            stripped,
        )
        if generic_match:
            return generic_match.group(1), "", "function"

        return "", "", "block"


# ── Phase 7 ChunkingEngine (original — unchanged) ─────────────────────────────

class ChunkingEngine:
    """
    Phase 7 chunking engine — unchanged from original.
    Kept for backward compatibility and as the fallback engine
    inside ASTChunkingEngine.

    Dispatches to:
      MarkdownChunker        → markdown, rst
      FunctionAwareChunker   → python, js, ts, java, go, rust, etc.
      SlidingWindowChunker   → everything else (sql, yaml, config)
    """

    def __init__(
        self,
        chunk_size:    int = 1500,
        chunk_overlap: int = 150,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._markdown     = MarkdownChunker()
        self._function     = FunctionAwareChunker()
        self._sliding      = SlidingWindowChunker()

    def chunk_file(self, parsed_file: ParsedFile) -> list[CodeChunk]:
        language = parsed_file.language

        if language in MARKDOWN_LANGUAGES:
            return self._markdown.chunk(
                parsed_file, self.chunk_size, self.chunk_overlap
            )

        if language in FUNCTION_AWARE_LANGUAGES:
            chunks = self._function.chunk(
                parsed_file, self.chunk_size, self.chunk_overlap
            )
            if chunks:
                return chunks

        return self._sliding.chunk(
            parsed_file, self.chunk_size, self.chunk_overlap
        )

    def chunk_all(
        self,
        parsed_files: list[ParsedFile],
        project_id:   str,
    ) -> ChunkingResult:
        result          = ChunkingResult(project_id=project_id)
        chunks_by_lang  : dict[str, int] = {}

        logger.info(
            f"🔪 Chunking {len(parsed_files)} files for {project_id}"
        )

        for pf in parsed_files:
            pf.project_id = project_id
            file_chunks   = self.chunk_file(pf)

            if not file_chunks:
                result.skipped_files += 1
                continue

            result.chunks.extend(file_chunks)
            result.total_files += 1

            lang = pf.language
            chunks_by_lang[lang] = chunks_by_lang.get(lang, 0) + len(file_chunks)

            logger.debug(
                f"  {pf.file_path}: {len(file_chunks)} chunks ({lang})"
            )

        result.total_chunks       = len(result.chunks)
        result.chunks_by_language = chunks_by_lang

        logger.info(
            f"✅ Chunking complete: {result.total_chunks} chunks "
            f"from {result.total_files} files "
            f"({result.skipped_files} skipped)"
        )
        for lang, count in sorted(
            chunks_by_lang.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"   {lang:15s}: {count} chunks")

        return result


# ── Module-level singletons ────────────────────────────────────────────────────

# Original Phase 7 engine (unchanged)
chunking_engine = ChunkingEngine(
    chunk_size    = getattr(settings, 'chunk_size',    1500),
    chunk_overlap = getattr(settings, 'chunk_overlap', 150),
)

# Improvement 2: AST engine with Phase 7 fallback
# Import here (after ChunkingEngine is defined) to avoid circular imports
try:
    from app.core.processing.ast_chunker import ASTChunkingEngine
    ast_chunking_engine = ASTChunkingEngine(
        chunk_size    = getattr(settings, 'chunk_size',    1500),
        chunk_overlap = getattr(settings, 'chunk_overlap', 150),
        prefer_ast    = True,
    )
    logger.debug("✅ AST chunking engine loaded")
except ImportError as e:
    # If ast_chunker.py is not yet deployed, fall back to Phase 7
    ast_chunking_engine = chunking_engine   # type: ignore
    logger.warning(f"AST chunker not available, using Phase 7: {e}")