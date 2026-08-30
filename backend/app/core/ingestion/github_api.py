# backend/app/core/ingestion/github_api.py
#
# Lightweight GitHub REST API client.
# Used to validate repos, get default branch, and check repo size
# BEFORE we start cloning — fast fail if the repo doesn't exist.

from urllib import response

import httpx
from dataclasses import dataclass
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"

# Request timeout — GitHub API should respond in under 10 seconds
API_TIMEOUT = 10.0

# GitHub's size limit we warn about (in KB — GitHub uses KB for repo sizes)
# Repos over 500MB take a very long time to clone
LARGE_REPO_THRESHOLD_KB = 500_000  # 500 MB


@dataclass
class RepoMetadata:
    """
    Key metadata about a GitHub repository fetched from the API.
    Used to make decisions before cloning starts.
    """
    owner:          str
    name:           str
    full_name:      str           # "owner/repo"
    default_branch: str           # "main" or "master" etc.
    description:    Optional[str]
    size_kb:        int           # Repo size in kilobytes
    is_private:     bool
    stars:          int
    language:       Optional[str] # GitHub's detected primary language
    clone_url:      str


class GithubApiClient:
    """
    Calls GitHub REST API endpoints to fetch repo information.

    No authentication required for public repos.
    If GITHUB_TOKEN env var is set, uses it to avoid rate limiting.
    GitHub rate limit: 60 req/hour unauthenticated, 5000/hour authenticated.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: Optional GitHub personal access token.
                   Set GITHUB_TOKEN in .env to avoid rate limits.
        """
        self.token = token
        self.headers = {
            "Accept":     "application/vnd.github.v3+json",
            "User-Agent": "CodeMind/1.0",
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

    async def get_repo_metadata(
        self,
        owner: str,
        repo: str
    ) -> RepoMetadata:
        """
        Fetches repository metadata from GitHub API.

        Raises:
            ValueError:  Repo not found or is private
            RuntimeError: GitHub API rate limit hit or network error
        """
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"


        async with httpx.AsyncClient(timeout=API_TIMEOUT,follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=self.headers)
            except httpx.TimeoutException:
                raise RuntimeError(
                    "GitHub API request timed out. "
                    "Check your internet connection and try again."
                )
            except httpx.NetworkError as e:
                raise RuntimeError(f"Network error contacting GitHub API: {e}")

        # Handle different HTTP status codes explicitly
        if response.status_code == 404:
            raise ValueError(
                f"Repository '{owner}/{repo}' not found. "
                f"Make sure the URL is correct and the repository is public."
            )

        if response.status_code == 403:
            # Could be rate limiting or private repo
            remaining = response.headers.get("X-RateLimit-Remaining", "?")
            if remaining == "0":
                reset_time = response.headers.get("X-RateLimit-Reset", "")
                raise RuntimeError(
                    f"GitHub API rate limit exceeded. "
                    f"Add GITHUB_TOKEN to your .env file for higher limits. "
                    f"Resets at: {reset_time}"
                )
            raise ValueError(
                f"Access denied to '{owner}/{repo}'. "
                f"The repository may be private."
            )

        if response.status_code == 451:
            raise ValueError(
                f"Repository '{owner}/{repo}' is unavailable for legal reasons."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub API returned unexpected status {response.status_code}"
            )

        data = response.json()

        # Warn about large repos but don't block them
        size_kb = data.get("size", 0)
        if size_kb > LARGE_REPO_THRESHOLD_KB:
            logger.warning(
                f"⚠️  Large repository detected: {size_kb // 1024}MB. "
                f"Cloning may take several minutes."
            )

        return RepoMetadata(
            owner=data["owner"]["login"],
            name=data["name"],
            full_name=data["full_name"],
            default_branch=data.get("default_branch", "main"),
            description=data.get("description"),
            size_kb=size_kb,
            is_private=data.get("private", False),
            stars=data.get("stargazers_count", 0),
            language=data.get("language"),
            clone_url=data["clone_url"],
        )

    async def check_branch_exists(
        self,
        owner: str,
        repo:  str,
        branch: str
    ) -> bool:
        """
        Checks if a specific branch exists in the repository.
        Returns False instead of raising an error — caller decides what to do.
        """
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/branches/{branch}"

        async with httpx.AsyncClient(timeout=API_TIMEOUT,follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=self.headers)
                return response.status_code == 200
            except Exception:
                return False    # Network error — assume branch might exist

    async def get_rate_limit_status(self) -> dict:
        """
        Returns current GitHub API rate limit status.
        Useful for debugging and monitoring.
        """
        url = f"{GITHUB_API_BASE}/rate_limit"
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                core = data["resources"]["core"]
                return {
                    "limit":     core["limit"],
                    "remaining": core["remaining"],
                    "reset_at":  core["reset"],
                }
        return {}


# Module-level singleton
# Reads GITHUB_TOKEN from environment if available
def create_github_api_client() -> GithubApiClient:
    import os
    token = os.getenv("GITHUB_TOKEN")
    return GithubApiClient(token=token)