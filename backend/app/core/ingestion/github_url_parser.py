# backend/app/core/ingestion/github_url_parser.py
#
# Parses and normalises GitHub repository URLs into their components.
#
# Handles all common GitHub URL formats:
#   https://github.com/owner/repo
#   https://github.com/owner/repo.git
#   https://github.com/owner/repo/tree/main
#   https://github.com/owner/repo/tree/feature/my-branch
#   http://github.com/owner/repo          (http instead of https)
#   github.com/owner/repo                 (no protocol)

import re
from dataclasses import dataclass
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedGithubUrl:
    """
    Structured representation of a GitHub repository URL.
    All fields are extracted from the raw URL string the user pasted.
    """
    owner:       str            # GitHub username or org: "tiangolo"
    repo:        str            # Repository name: "fastapi"
    branch:      Optional[str]  # Branch from URL if present: "main", None
    clone_url:   str            # Clean HTTPS URL for git clone
    display_url: str            # Human-readable URL for logging/display


class GithubUrlParser:
    """
    Parses GitHub URLs into structured components.

    Usage:
        parser = GithubUrlParser()
        parsed = parser.parse("https://github.com/tiangolo/fastapi/tree/master")
        # parsed.owner  = "tiangolo"
        # parsed.repo   = "fastapi"
        # parsed.branch = "master"
    """

    # Regex that matches all common GitHub URL formats.
    # Named groups make the match result self-documenting.
    #
    # Pattern breakdown:
    # (?:https?://)?         → optional http:// or https://
    # (?:www\.)?             → optional www.
    # github\.com/           → literal github.com/
    # (?P<owner>[^/]+)/      → capture owner (anything up to /)
    # (?P<repo>[^/\.]+)      → capture repo name (no / or .)
    # (?:\.git)?             → optional .git suffix
    # (?:/tree/              → optional /tree/ path
    #   (?P<branch>[^/?#]+)  → capture branch name
    # )?
    # GITHUB_URL_PATTERN = re.compile(
    #     r"(?:https?://)?"
    #     r"(?:www\.)?"
    #     r"github\.com/"
    #     r"(?P<owner>[^/\s]+)/"
    #     r"(?P<repo>[^/\.\s]+)"
    #     r"(?:\.git)?"
    #     r"(?:/tree/(?P<branch>[^/?#\s]+))?",
    #     re.IGNORECASE
    # )

    GITHUB_URL_PATTERN = re.compile(
        r"(?:https?://)?"
        r"(?:www\.)?"
        r"github\.com/"
        r"(?P<owner>[^/\s]+)/"
        r"(?P<repo>[^/\s]+)"
        r"(?:\.git)?"
        r"(?:/tree/(?P<branch>.*))?"
        r"/?$",
        re.IGNORECASE
    )

    def parse(self, url: str) -> ParsedGithubUrl:
        """
        Parses a GitHub URL string into a ParsedGithubUrl.

        Raises:
            ValueError: if the URL doesn't match the expected GitHub format
        """
        url = url.strip()

        # Add protocol if missing (e.g. "github.com/owner/repo")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        match = self.GITHUB_URL_PATTERN.search(url)
        if not match:
            raise ValueError(
                f"Could not parse GitHub URL: '{url}'. "
                f"Expected format: https://github.com/owner/repository"
            )

        owner  = match.group("owner").lower()
        repo   = match.group("repo")
        branch = match.group("branch")  # None if not in URL
        if branch:
            branch = branch.split("/")[0]

        # Remove .git suffix from repo name if present
        if repo.endswith(".git"):
            repo = repo[:-4]

        # Build the canonical clone URL
        clone_url = f"https://github.com/{owner}/{repo}.git"
        display_url = f"https://github.com/{owner}/{repo}"

        parsed = ParsedGithubUrl(
            owner=owner,
            repo=repo,
            branch=branch,
            clone_url=clone_url,
            display_url=display_url,
        )

        logger.debug(
            f"Parsed GitHub URL: {owner}/{repo} "
            f"(branch: {branch or 'default'})"
        )
        return parsed

    def validate_components(self, owner: str, repo: str) -> None:
        """
        Validates owner and repo name against GitHub's naming rules.
        GitHub usernames: alphanumeric + hyphens, max 39 chars.
        Repo names: alphanumeric + hyphens + underscores + dots, max 100 chars.

        Raises:
            ValueError: if names contain invalid characters
        """
        owner_pattern = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,37}[a-zA-Z0-9])?$")
        repo_pattern  = re.compile(r"^[a-zA-Z0-9._\-]{1,100}$")

        if not owner_pattern.match(owner):
            raise ValueError(
                f"Invalid GitHub username: '{owner}'. "
                f"Must be alphanumeric with hyphens, max 39 characters."
            )

        if not repo_pattern.match(repo):
            raise ValueError(
                f"Invalid repository name: '{repo}'. "
                f"Must be alphanumeric with hyphens, underscores, or dots."
            )


# Module-level singleton
url_parser = GithubUrlParser()