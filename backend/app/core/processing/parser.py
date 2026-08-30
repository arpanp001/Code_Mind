# backend/app/core/processing/parser.py
#
# Reads source code files from disk and produces ParsedFile objects.
# Two responsibilities:
#   1. Walk a directory tree, applying all filters
#   2. Read each valid file's content with correct encoding

import os
from pathlib import Path
from typing import Generator
import chardet
import pathspec

from app.core.processing.language_detector import detector as lang_detector
from app.models.ingest_models import ParsedFile
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration Constants ───────────────────────────────────────────────────

# Files larger than this are skipped — they're usually generated or minified.
# A 500KB file is a minified bundle or a massive auto-generated file,
# not something a developer wrote and wants to query.
MAX_FILE_SIZE_BYTES = 500 * 1024   # 500 KB

# Files with fewer than this many characters are skipped.
# A 3-character file (e.g. empty __init__.py) adds no RAG value.
MIN_CONTENT_LENGTH = 10


class CodeParser:
    """
    Walks a project directory and produces ParsedFile objects
    for every valid source code file.

    Usage:
        parser = CodeParser()
        files = parser.parse_directory("/tmp/uploads/proj_abc123/")
        # files is a list of ParsedFile objects, ready for chunking
    """

    def __init__(self):
        self._gitignore_spec = None   # Will hold parsed .gitignore rules

    def parse_directory(
        self,
        root_path: str,
        project_id: str | None = None
    ) -> list[ParsedFile]:
        """
        Main entry point.
        Walks root_path, filters files, reads content, returns ParsedFile list.

        Args:
            root_path:  Absolute path to the extracted project folder
            project_id: Optional project ID to attach to each ParsedFile

        Returns:
            List of ParsedFile objects sorted by file path
        """
        root = Path(root_path).resolve()

        if not root.exists():
            raise FileNotFoundError(f"Project directory not found: {root_path}")

        logger.info(f"🔍 Parsing directory: {root}")

        # Load .gitignore if the project has one
        self._load_gitignore(root)

        parsed_files: list[ParsedFile] = []
        skipped_count = 0

        # Walk the directory tree
        for file_path in self._walk_files(root):
            try:
                parsed = self._read_file(file_path, root, project_id)
                if parsed:
                    parsed_files.append(parsed)
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Could not read {file_path}: {e}")
                skipped_count += 1

        logger.info(
            f"✅ Parsed {len(parsed_files)} files "
            f"({skipped_count} skipped) from {root.name}"
        )

        # Sort by path so related files are processed together
        # (all auth/ files together, all models/ together, etc.)
        return sorted(parsed_files, key=lambda f: f.relative_path)

    def _walk_files(self, root: Path) -> Generator[Path, None, None]:
        """
        Generator that yields every processable file path under root.

        Using os.walk() instead of Path.rglob() because os.walk() lets us
        PRUNE entire directories before descending into them.
        With rglob(), you'd still visit all 10,000 files in node_modules
        just to filter them — os.walk() with dirnames modification skips them entirely.
        """
        for dirpath, dirnames, filenames in os.walk(root):
            current_dir = Path(dirpath)

            # ── Prune directories IN PLACE ────────────────────────────────
            # Modifying dirnames[:] (slice assignment) tells os.walk()
            # NOT to descend into those directories.
            # This is the critical optimization — node_modules is never entered.
            dirnames[:] = [
                d for d in dirnames
                if not lang_detector.should_skip_directory(d)
                and not d.startswith('.')   # Skip hidden dirs like .git
            ]

            for filename in filenames:
                file_path = current_dir / filename
                yield file_path

    def _read_file(
        self,
        file_path: Path,
        root: Path,
        project_id: str | None
    ) -> ParsedFile | None:
        """
        Reads a single file and returns a ParsedFile, or None if it should be skipped.

        Steps:
        1. Check file size (skip if too large or too small)
        2. Detect language from extension (skip if unknown)
        3. Check .gitignore rules
        4. Read raw bytes
        5. Check for binary content
        6. Detect encoding and decode to string
        7. Return ParsedFile
        """
        # ── Size check (fast — uses OS metadata, doesn't read file) ──────
        try:
            file_size = file_path.stat().st_size
        except OSError:
            return None

        if file_size == 0:
            return None   # Empty file

        if file_size > MAX_FILE_SIZE_BYTES:
            logger.debug(
                f"⏭️  Skipping large file: "
                f"{file_path.name} ({file_size // 1024}KB)"
            )
            return None

        # ── Language detection ────────────────────────────────────────────
        language = lang_detector.detect(str(file_path))
        if language is None:
            return None   # Unknown or unwanted file type

        # ── .gitignore check ─────────────────────────────────────────────
        if self._gitignore_spec:
            # pathspec needs a RELATIVE path (relative to where .gitignore lives)
            relative = file_path.relative_to(root)
            if self._gitignore_spec.match_file(str(relative)):
                logger.debug(f"⏭️  Gitignore match: {relative}")
                return None

        # ── Read raw bytes ────────────────────────────────────────────────
        try:
            raw_bytes = file_path.read_bytes()
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return None

        # ── Binary content check ─────────────────────────────────────────
        # Even if the extension looks good, check the actual bytes.
        # Avoids crashes when reading corrupted or misnamed files.
        if lang_detector.is_likely_binary(raw_bytes):
            logger.debug(f"⏭️  Binary content detected: {file_path.name}")
            return None

        # ── Encoding detection and decode ─────────────────────────────────
        content, encoding = self._decode_content(raw_bytes, file_path)
        if content is None:
            return None

        # ── Minimum content check ─────────────────────────────────────────
        stripped = content.strip()
        if len(stripped) < MIN_CONTENT_LENGTH:
            return None   # File is effectively empty

        # ── Build relative path ───────────────────────────────────────────
        # Store paths relative to project root for display in the frontend.
        # We never want to show "/tmp/uploads/proj_abc123/src/auth/login.py"
        # — we show "src/auth/login.py"
        try:
            relative_path = str(file_path.relative_to(root))
        except ValueError:
            relative_path = file_path.name

        # Normalize path separators to forward slashes (Windows compatibility)
        relative_path = relative_path.replace("\\", "/")

        line_count = len(content.splitlines())

        return ParsedFile(
            file_path=relative_path,
            relative_path=relative_path,
            content=content,
            language=language,
            file_size_bytes=file_size,
            line_count=line_count,
            encoding=encoding,
            project_id=project_id,
        )

    def _decode_content(
        self,
        raw_bytes: bytes,
        file_path: Path
    ) -> tuple[str | None, str]:
        """
        Decodes raw bytes to a string, trying multiple encodings.

        Strategy:
        1. Try UTF-8 (most modern code)
        2. Use chardet to detect encoding from byte patterns
        3. Try UTF-8 with error replacement as last resort
        4. Return None if all attempts fail
        """
        # ── Try UTF-8 first (fastest, covers 95%+ of code) ───────────────
        try:
            return raw_bytes.decode('utf-8'), 'utf-8'
        except UnicodeDecodeError:
            pass

        # ── Use chardet to detect encoding ────────────────────────────────
        # chardet analyzes byte frequency patterns to guess the encoding
        detected = chardet.detect(raw_bytes)
        encoding = detected.get('encoding')
        confidence = detected.get('confidence', 0)

        if encoding and confidence > 0.7:
            try:
                return raw_bytes.decode(encoding), encoding
            except (UnicodeDecodeError, LookupError):
                pass

        # ── Last resort: UTF-8 with replacement characters ────────────────
        # This never fails — invalid bytes become the replacement character (?)
        # Slightly garbled output is better than skipping the file entirely
        try:
            content = raw_bytes.decode('utf-8', errors='replace')
            logger.debug(
                f"⚠️  Used fallback encoding for: {file_path.name}"
            )
            return content, 'utf-8-fallback'
        except Exception:
            logger.warning(f"Cannot decode: {file_path.name}")
            return None, 'unknown'

    def _load_gitignore(self, root: Path) -> None:
        """
        Reads the project's .gitignore file (if it exists) and compiles
        the patterns into a pathspec matcher.

        This means if a project ignores "*.log" or "secret_config.py",
        we respect that and don't include those files in the index.
        """
        gitignore_path = root / '.gitignore'
        if not gitignore_path.exists():
            self._gitignore_spec = None
            return

        try:
            gitignore_content = gitignore_path.read_text(encoding='utf-8')
            # pathspec.GitWildMatchPattern understands .gitignore syntax
            self._gitignore_spec = pathspec.PathSpec.from_lines(
                'gitwildmatch', gitignore_content.splitlines()
            )
            logger.info("📋 Loaded .gitignore rules")
        except Exception as e:
            logger.warning(f"Could not parse .gitignore: {e}")
            self._gitignore_spec = None


# Module-level singleton
parser = CodeParser()