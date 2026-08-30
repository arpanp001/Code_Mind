# backend/app/core/processing/language_detector.py
#
# Detects the programming language of a file based on its extension.
# Also determines whether a file is worth processing at all.

from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Extension → Language Mapping ─────────────────────────────────────────────
# Every extension supported by CodeMind maps to a canonical language name.
# This name is stored in ChromaDB metadata and used by the frontend
# to pick the right syntax highlighter.

EXTENSION_MAP: dict[str, str] = {
    # Python
    ".py":    "python",
    ".pyw":   "python",
    ".pyx":   "python",    # Cython

    # JavaScript / TypeScript
    ".js":    "javascript",
    ".jsx":   "javascript",
    ".mjs":   "javascript",
    ".cjs":   "javascript",
    ".ts":    "typescript",
    ".tsx":   "typescript",

    # Java
    ".java":  "java",

    # C / C++
    ".c":     "c",
    ".h":     "c",
    ".cpp":   "cpp",
    ".cc":    "cpp",
    ".cxx":   "cpp",
    ".hpp":   "cpp",
    ".hxx":   "cpp",

    # C#
    ".cs":    "csharp",

    # Web
    ".html":  "html",
    ".htm":   "html",
    ".css":   "css",
    ".scss":  "scss",
    ".sass":  "scss",
    ".less":  "css",

    # Data / Config
    ".json":  "json",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".toml":  "toml",
    ".xml":   "xml",
    ".env":   "bash",       # .env files look like bash assignments

    # Shell / Scripts
    ".sh":    "bash",
    ".bash":  "bash",
    ".zsh":   "bash",
    ".fish":  "bash",
    ".ps1":   "powershell",
    ".bat":   "batch",

    # Documentation
    ".md":    "markdown",
    ".mdx":   "markdown",
    ".rst":   "rst",
    ".txt":   "text",

    # Database
    ".sql":   "sql",

    # Go
    ".go":    "go",

    # Rust
    ".rs":    "rust",

    # Ruby
    ".rb":    "ruby",

    # PHP
    ".php":   "php",

    # Swift / Kotlin
    ".swift": "swift",
    ".kt":    "kotlin",
    ".kts":   "kotlin",

    # Dart / Flutter
    ".dart":  "dart",

    # Build files
    ".gradle": "groovy",
    ".cmake":  "cmake",
    "makefile": "makefile",   # No extension — matched by filename below
}

# ── Files to Process by Exact Filename (no extension) ────────────────────────
# Some important files have no extension but should always be included.
FILENAME_MAP: dict[str, str] = {
    "makefile":       "makefile",
    "dockerfile":     "dockerfile",
    "docker-compose": "yaml",
    "jenkinsfile":    "groovy",
    "rakefile":       "ruby",
    "gemfile":        "ruby",
    "procfile":       "text",
    "readme":         "markdown",
    ".env.example":   "bash",
    ".env.sample":    "bash",
}

# ── Files/Patterns to ALWAYS Skip ─────────────────────────────────────────────
# These are binary files, generated files, or files with no learning value.
# Storing them would waste vector space and confuse the RAG retrieval.

SKIP_EXTENSIONS: set[str] = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".bmp", ".tiff", ".psd", ".ai", ".eps",

    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",

    # Compiled / Binary
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear",
    ".exe", ".dll", ".so", ".dylib", ".lib", ".a",
    ".o", ".obj",

    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",

    # Audio / Video
    ".mp3", ".mp4", ".wav", ".ogg", ".avi", ".mov", ".mkv",
    ".flac", ".aac",

    # Documents (binary)
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",

    # Database files (binary)
    ".db", ".sqlite", ".sqlite3",

    # Lock files (auto-generated, enormous, no RAG value)
    ".lock",

    # Map files (minified source maps)
    ".map",

    # Package files
    ".whl", ".egg",
}

SKIP_FILENAMES: set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
}

# ── Directory Names to Always Skip ────────────────────────────────────────────
# These directories are never useful for understanding the codebase.
# Skipping them is the single biggest performance win — node_modules
# alone can contain 10,000+ files.

SKIP_DIRECTORIES: set[str] = {
    # Package managers
    "node_modules",
    "vendor",
    "bower_components",

    # Python virtual environments
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".pytest_cache",
    "*.egg-info",

    # Version control
    ".git",
    ".svn",
    ".hg",

    # Build output
    "dist",
    "build",
    "out",
    "output",
    "target",          # Java Maven/Gradle
    "bin",
    "obj",             # C#
    ".next",           # Next.js
    ".nuxt",           # Nuxt.js
    "coverage",

    # IDE / Editor
    ".idea",
    ".vscode",
    ".vs",
    "__MACOSX",        # macOS ZIP artifacts
    ".DS_Store",

    # Logs / Temp
    "logs",
    "tmp",
    "temp",
    ".cache",
}


class LanguageDetector:
    """
    Detects whether a file should be processed and what language it is.

    Usage:
        detector = LanguageDetector()
        lang = detector.detect("src/auth/login.py")     # → "python"
        skip = detector.should_skip_directory("node_modules")  # → True
    """

    def detect(self, file_path: str) -> str | None:
        """
        Returns the language string for a file, or None if it should be skipped.

        Returns None (skip) when:
        - The file extension is in SKIP_EXTENSIONS
        - The file has no extension and isn't in FILENAME_MAP
        - The extension is unknown (no mapping found)

        Returns a language string when the file should be processed.
        """
        path = Path(file_path)
        ext = path.suffix.lower()           # e.g. ".py"
        name = path.name.lower()            # e.g. "login.py"
        stem = path.stem.lower()  

        if name in SKIP_FILENAMES:
            return None          # e.g. "login" (no extension)

        # 1. Always skip these extensions
        if ext in SKIP_EXTENSIONS:
            return None

        # SKIP_FILENAMES = {
        #     "package-lock.json",
        #     "yarn.lock",
        #     "poetry.lock",
        #     "pnpm-lock.yaml",
        # }

        # 2. Check extension map first (most common case)
        if ext and ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]

        # 3. Check exact filename match (e.g. "Dockerfile", "Makefile")
        if name in FILENAME_MAP:
            return FILENAME_MAP[name]

        # 4. Check stem (filename without extension) for files like ".env.example"
        if stem in FILENAME_MAP:
            return FILENAME_MAP[stem]

        # 5. Unknown extension — skip it
        # We don't process files we can't identify — they're likely binary
        # or generated files that would confuse the RAG retrieval
        logger.debug(f"Skipping unknown file type: {file_path}")
        return None

    def should_skip_directory(self, dir_name: str) -> bool:
        """
        Returns True if an entire directory should be skipped.
        Called during directory walking — if True, we never descend into it.

        This is a critical optimization: checking the directory name
        before recursing prevents scanning thousands of files in node_modules.
        """
        return dir_name.lower() in SKIP_DIRECTORIES

    def is_likely_binary(self, content_bytes: bytes) -> bool:
        """
        Checks if a file's raw bytes look like binary data.
        Even if a file has a .txt extension, it might be binary.

        Method: count null bytes in the first 8KB.
        Binary files almost always contain null bytes; text files never do.
        """
        sample = content_bytes[:8192]        # Check first 8KB only (fast)
        null_count = sample.count(b'\x00')
        # If more than 10% of the sample is null bytes → binary
        return null_count > len(sample) * 0.10


# Module-level singleton — import this everywhere instead of creating new instances
detector = LanguageDetector()