# backend/tests/test_github_ingestion.py
#
# Tests for GitHub URL parsing and ingestion logic.
# Run with: pytest tests/test_github_ingestion.py -v
#
# Note: Tests that actually hit the GitHub API / clone repos
# are marked with @pytest.mark.integration and skipped by default.
# Run them with: pytest -m integration

import pytest
from app.core.ingestion.github_url_parser import GithubUrlParser


class TestGithubUrlParser:

    def setup_method(self):
        self.parser = GithubUrlParser()

    # ── Valid URL formats ─────────────────────────────────────────────────

    def test_parses_standard_https_url(self):
        result = self.parser.parse("https://github.com/tiangolo/fastapi")
        assert result.owner == "tiangolo"
        assert result.repo  == "fastapi"
        assert result.branch is None

    def test_parses_url_with_git_suffix(self):
        result = self.parser.parse("https://github.com/tiangolo/fastapi.git")
        assert result.repo == "fastapi"   # .git removed

    def test_parses_url_with_branch(self):
        result = self.parser.parse(
            "https://github.com/tiangolo/fastapi/tree/master"
        )
        assert result.owner  == "tiangolo"
        assert result.repo   == "fastapi"
        assert result.branch == "master"

    def test_parses_url_without_protocol(self):
        result = self.parser.parse("github.com/django/django")
        assert result.owner == "django"
        assert result.repo  == "django"

    def test_parses_url_with_trailing_slash(self):
        result = self.parser.parse("https://github.com/django/django/")
        assert result.repo == "django"

    def test_parses_url_with_nested_branch(self):
        result = self.parser.parse(
            "https://github.com/org/repo/tree/feature/my-new-feature"
        )
        assert result.owner  == "org"
        assert result.repo   == "repo"
        assert result.branch == "feature"   # Only first segment captured

    def test_builds_correct_clone_url(self):
        result = self.parser.parse("https://github.com/tiangolo/fastapi")
        assert result.clone_url == "https://github.com/tiangolo/fastapi.git"

    def test_owner_is_lowercased(self):
        result = self.parser.parse("https://github.com/TianGolo/FastAPI")
        assert result.owner == "tiangolo"   # Lowercased

    # ── Invalid URLs ──────────────────────────────────────────────────────

    def test_rejects_non_github_url(self):
        with pytest.raises(ValueError, match="Could not parse"):
            self.parser.parse("https://gitlab.com/owner/repo")

    def test_rejects_plain_string(self):
        with pytest.raises(ValueError):
            self.parser.parse("not a url at all")

    def test_rejects_github_homepage(self):
        with pytest.raises(ValueError):
            self.parser.parse("https://github.com")

    # ── Component validation ──────────────────────────────────────────────

    def test_validates_valid_owner(self):
        # Should not raise
        self.parser.validate_components("tiangolo", "fastapi")

    def test_rejects_owner_with_special_chars(self):
        with pytest.raises(ValueError, match="Invalid GitHub username"):
            self.parser.validate_components("owner@name!", "repo")

    def test_rejects_repo_with_spaces(self):
        with pytest.raises(ValueError, match="Invalid repository name"):
            self.parser.validate_components("owner", "my repo name")

    # ── Clone URL format ──────────────────────────────────────────────────

    def test_clone_url_always_https(self):
        result = self.parser.parse("http://github.com/owner/repo")
        assert result.clone_url.startswith("https://")

    def test_clone_url_has_git_suffix(self):
        result = self.parser.parse("https://github.com/owner/repo")
        assert result.clone_url.endswith(".git")


class TestGithubUrlParserEdgeCases:

    def setup_method(self):
        self.parser = GithubUrlParser()

    def test_handles_org_with_hyphens(self):
        result = self.parser.parse(
            "https://github.com/my-awesome-org/my-repo"
        )
        assert result.owner == "my-awesome-org"
        assert result.repo  == "my-repo"

    def test_handles_repo_with_dots(self):
        result = self.parser.parse(
            "https://github.com/owner/my.awesome.repo"
        )
        assert result.repo == "my.awesome.repo"

    def test_handles_numeric_owner(self):
        result = self.parser.parse("https://github.com/user123/repo456")
        assert result.owner == "user123"
        assert result.repo  == "repo456"