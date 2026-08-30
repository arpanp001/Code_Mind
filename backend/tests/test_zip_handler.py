# backend/tests/test_zip_handler.py
#
# Tests for the ZIP extraction and file parsing pipeline.
# Run with: pytest tests/test_zip_handler.py -v

import os
import zipfile
import tempfile
import pytest
from pathlib import Path

from app.core.processing.language_detector import LanguageDetector
from app.core.processing.parser import CodeParser


# ── Language Detector Tests ───────────────────────────────────────────────────

class TestLanguageDetector:

    def setup_method(self):
        self.detector = LanguageDetector()

    def test_detects_python(self):
        assert self.detector.detect("src/auth/login.py") == "python"

    def test_detects_javascript(self):
        assert self.detector.detect("app/components/Button.jsx") == "javascript"

    def test_detects_typescript(self):
        assert self.detector.detect("src/types/index.ts") == "typescript"

    def test_detects_java(self):
        assert self.detector.detect("src/main/App.java") == "java"

    def test_skips_images(self):
        assert self.detector.detect("assets/logo.png") is None

    def test_skips_lock_files(self):
        assert self.detector.detect("package-lock.json") is None
        # .lock extension is skipped
        assert self.detector.detect("poetry.lock") is None

    def test_skips_compiled_files(self):
        assert self.detector.detect("Main.class") is None
        assert self.detector.detect("app.pyc") is None

    def test_detects_dockerfile(self):
        assert self.detector.detect("Dockerfile") == "dockerfile"

    def test_detects_markdown(self):
        assert self.detector.detect("README.md") == "markdown"

    def test_skips_node_modules_directory(self):
        assert self.detector.should_skip_directory("node_modules") is True

    def test_skips_git_directory(self):
        assert self.detector.should_skip_directory(".git") is True

    def test_does_not_skip_src(self):
        assert self.detector.should_skip_directory("src") is False

    def test_binary_detection(self):
        # File with lots of null bytes → binary
        binary_data = b'\x00' * 100 + b'some text'
        assert self.detector.is_likely_binary(binary_data) is True

        # Normal text file → not binary
        text_data = b'def hello():\n    print("Hello, World!")\n'
        assert self.detector.is_likely_binary(text_data) is False


# ── Code Parser Tests ─────────────────────────────────────────────────────────

class TestCodeParser:

    def setup_method(self):
        self.parser = CodeParser()

    def _make_temp_project(self, files: dict[str, str]) -> str:
        """
        Helper: creates a temporary directory with the given files.
        files = { "relative/path.ext": "file content" }
        Returns the temp directory path.
        """
        tmp = tempfile.mkdtemp()
        for rel_path, content in files.items():
            full_path = Path(tmp) / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
        return tmp

    def test_parses_python_file(self):
        tmp = self._make_temp_project({
            "src/main.py": "def hello():\n    return 'Hello, World!'\n"
        })
        files = self.parser.parse_directory(tmp)
        assert len(files) == 1
        assert files[0].language == "python"
        assert files[0].relative_path == "src/main.py"
        assert "def hello" in files[0].content

    def test_skips_node_modules(self):
        tmp = self._make_temp_project({
            "src/index.js":               "console.log('hello')",
            "node_modules/lodash/main.js": "// lodash code"
        })
        files = self.parser.parse_directory(tmp)
        # Only src/index.js should be parsed
        paths = [f.relative_path for f in files]
        assert "src/index.js" in paths
        assert not any("node_modules" in p for p in paths)

    def test_skips_empty_files(self):
        tmp = self._make_temp_project({
            "src/empty.py":   "",
            "src/main.py":    "def hello():\n    pass\n"
        })
        files = self.parser.parse_directory(tmp)
        # Empty file should be skipped
        assert len(files) == 1
        assert files[0].relative_path == "src/main.py"

    def test_multiple_languages(self):
        tmp = self._make_temp_project({
            "backend/app.py":      "from flask import Flask",
            "frontend/App.jsx":    "export default function App() {}",
            "README.md":           "# My Project\nThis is a README.",
        })
        files = self.parser.parse_directory(tmp)
        languages = {f.language for f in files}
        assert "python"     in languages
        assert "javascript" in languages
        assert "markdown"   in languages

    def test_provides_line_count(self):
        content = "line 1\nline 2\nline 3\n"
        tmp = self._make_temp_project({"test.py": content})
        files = self.parser.parse_directory(tmp)
        assert files[0].line_count == 3

    def test_skips_dist_directory(self):
        tmp = self._make_temp_project({
            "src/app.js":    "const x = 1;",
            "dist/app.js":   "const x=1;"   # minified
        })
        files = self.parser.parse_directory(tmp)
        paths = [f.relative_path for f in files]
        assert "src/app.js" in paths
        assert not any("dist" in p for p in paths)