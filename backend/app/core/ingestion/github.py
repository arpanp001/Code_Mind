# backend/app/core/ingestion/github.py
#
# Main GitHub repository ingestion engine.
# Coordinates URL parsing, API validation, cloning, and file parsing
# into a single process() call that returns an IngestionResult.

import os
import shutil
from pathlib import Path
from unittest import result

import git
from git import Repo, GitCommandError

from app.core.ingestion.github_url_parser import url_parser
from app.core.ingestion.github_api        import create_github_api_client
from app.core.ingestion.clone_progress    import CloneProgressTracker
from app.core.processing.parser           import parser as code_parser
from app.models.ingest_models             import ParsedFile, IngestionResult
from app.config                           import settings
from app.utils.logger                     import get_logger

logger = get_logger(__name__)

# Where cloned repos are stored temporarily
# Structure: {upload_dir}/{project_id}/repo/
CLONE_SUBDIR = "repo"

# Git clone timeout in seconds
# Large repos (Linux kernel, etc.) can take a while
CLONE_TIMEOUT = 600  # 5 minutes
MAX_FILES_INDEX = 300 


class GithubIngestionEngine:
    """
    Ingests a public GitHub repository into ParsedFile objects.

    Usage:
        engine = GithubIngestionEngine()
        result = await engine.process(
            project_id="proj_abc",
            url="https://github.com/tiangolo/fastapi",
            branch="master"
        )
    """

    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.api_client = create_github_api_client()

    async def process(
        self,
        project_id: str,
        url:        str,
        branch:     str | None = None,
    ) -> IngestionResult:
        """
        Full pipeline: validate → clone → parse → return result.

        Args:
            project_id: The project ID (used for directory naming)
            url:        Raw GitHub URL from the user
            branch:     Branch to clone (None = use repo's default branch)

        Returns:
            IngestionResult with parsed files and stats
        """
        result = IngestionResult(project_id=project_id)
        clone_path = self.upload_dir / project_id / CLONE_SUBDIR

        try:
            # ── Step 1: Parse the URL ─────────────────────────────────────
            logger.info(f"🔗 Parsing GitHub URL: {url}")
            try:
                parsed_url = url_parser.parse(url)
                url_parser.validate_components(parsed_url.owner, parsed_url.repo)
            except ValueError as e:
                result.errors.append(str(e))
                return result

            logger.info(
                f"📦 Repository: {parsed_url.owner}/{parsed_url.repo}"
            )

            # ── Step 2: Validate via GitHub API ───────────────────────────
            logger.info("🌐 Validating repository via GitHub API...")
            try:
                metadata = await self.api_client.get_repo_metadata(
                    parsed_url.owner,
                    parsed_url.repo
                )
            except ValueError as e:
                # Repo not found, private, etc. — user error
                result.errors.append(str(e))
                return result
            except RuntimeError as e:
                # Network error, rate limit — transient error
                result.errors.append(str(e))
                return result

            logger.info(
                f"✅ Repository validated: {metadata.full_name} "
                f"({metadata.size_kb // 1024}MB, "
                f"⭐ {metadata.stars})"
            )

            # ── Step 3: Resolve the branch to clone ───────────────────────
            # Priority order:
            # 1. Branch specified in the URL path (/tree/branch-name)
            # 2. Branch passed as a parameter to this function
            # 3. Repository's default branch from API metadata
            target_branch = (
                parsed_url.branch  # From URL: /tree/feature-x
                or branch          # From request body
                or metadata.default_branch   # From GitHub API
            )

            # Verify the branch exists (skip if using default branch)
            if target_branch != metadata.default_branch:
                branch_exists = await self.api_client.check_branch_exists(
                    parsed_url.owner,
                    parsed_url.repo,
                    target_branch
                )
                if not branch_exists:
                    logger.warning(
                        f"Branch '{target_branch}' not found, "
                        f"falling back to '{metadata.default_branch}'"
                    )
                    target_branch = metadata.default_branch

            logger.info(f"🌿 Target branch: {target_branch}")

            # ── Step 4: Clone the repository ──────────────────────────────
            logger.info(f"⬇️  Cloning into: {clone_path}")
            clone_path.mkdir(parents=True, exist_ok=True)

            progress_tracker = CloneProgressTracker(project_id)

            try:
                self._clone_repository(
                    clone_url=parsed_url.clone_url,
                    clone_path=clone_path,
                    branch=target_branch,
                    progress=progress_tracker,
                )
                progress_tracker.complete()

            except GitCommandError as e:
                error_msg = self._parse_git_error(e)
                result.errors.append(error_msg)
                logger.error(f"❌ Clone failed: {error_msg}")
                return result
            except Exception as e:
                result.errors.append(f"Clone failed: {str(e)}")
                return result

            # ── Step 5: Parse all source files ────────────────────────────
            logger.info("🔍 Parsing cloned source files...")

            parsed_files = code_parser.parse_directory(
                str(clone_path),
                project_id=project_id
            )

            parsed_files = self._filter_large_repo(parsed_files)

            result.total_files_found = self._count_all_files(clone_path)
            result.files_processed   = len(parsed_files)
            result.files_skipped     = (
                result.total_files_found - result.files_processed
            )
            result.parsed_files = parsed_files

            logger.info(
                f"✅ GitHub ingestion complete: "
                f"{result.files_processed} files parsed, "
                f"{result.files_skipped} skipped"
            )

        except Exception as e:
            error_msg = f"Unexpected error during ingestion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)

        finally:
            # Always clean up progress tracker
            progress_tracker.cleanup() if 'progress_tracker' in locals() else None

        return result

    def _clone_repository(
        self,
        clone_url:  str,
        clone_path: Path,
        branch:     str,
        progress:   CloneProgressTracker,
    ) -> None:
        """
        Performs the actual git clone operation.

        We use --depth=1 (shallow clone) which only fetches the latest
        commit on the target branch, not the entire git history.

        Why shallow?
        - A repo with 10 years of history might have 500MB of commits
        - We only need the CURRENT source code, not history
        - Shallow clone of a large repo: 30 seconds
        - Full clone of the same repo: 10+ minutes
        """
        logger.info(
            f"Running: git clone --depth 1 --branch {branch} "
            f"{clone_url} {clone_path}"
        )

        Repo.clone_from(
            url=clone_url,
            to_path=str(clone_path),
            branch=branch,
            depth=1,
            progress=progress,
        )
    
    # backend/app/core/ingestion/github.py
# Replace _filter_large_repo with a comprehensive version:

    def _filter_large_repo(
        self,
        parsed_files: list,
        max_files:    int = 300,
    ) -> list:
        """
        Smart filtering for large repos.
        Skips: tests, docs, examples, migrations, generated code.
        Keeps: core source, config, entry points.
        """
        if len(parsed_files) <= max_files:
            return parsed_files

        logger.warning(
            f"Large repo: {len(parsed_files)} files → filtering to {max_files}"
        )

        # Directories to always skip (even if not in language_detector)
        SKIP_DIRS = {
            'test', 'tests', 'spec', 'specs', '__tests__',
            'docs', 'doc', 'documentation',
            'examples', 'example', 'samples', 'demo', 'demos',
            'migrations', 'migration',
            'node_modules', 'dist', 'build', 'out', '__pycache__',
            '.git', 'vendor', 'third_party', 'thirdparty',
            'fixtures', 'mocks', 'stubs', 'assets', 'static',
        }

        # Priority score for each file (lower = higher priority)
        def score(f: object) -> int:
            p    = f.file_path.lower()
            parts = p.replace('\\', '/').split('/')

            # Skip anything in low-value directories
            for part in parts[:-1]:
                if part in SKIP_DIRS:
                    return 99

            # Filename-based scoring
            name = parts[-1]

            # Highest priority: entry points and core files
            if name in ('main.py', 'app.py', 'index.js', 'index.ts',
                        'main.go', 'main.rs', 'server.py', 'server.js',
                        'manage.py', '__init__.py', 'cli.py'):
                return 1

            # High priority: source files in core directories
            if any(p.startswith(d + '/') for d in
                   ('src', 'lib', 'app', 'core', 'pkg', 'internal')):
                return 2

            # Medium: config and setup files
            if name in ('setup.py', 'setup.cfg', 'pyproject.toml',
                        'package.json', 'cargo.toml', 'go.mod',
                        'requirements.txt', 'Makefile', 'dockerfile',
                        'docker-compose.yml'):
                return 3

            # General source files
            if f.language not in ('markdown', 'rst', 'text'):
                return 4

            # Documentation
            if f.language in ('markdown', 'rst'):
                return 7

            return 5

        sorted_files = sorted(parsed_files, key=score)
        kept         = sorted_files[:max_files]

        skipped = len(parsed_files) - len(kept)
        logger.info(
            f"File filter: kept {len(kept)}, skipped {skipped} "
            f"(tests/docs/examples)"
        )
        return kept


    def _count_all_files(self, path: Path) -> int:
        """Counts all files under a path (before filtering)."""
        try:
            return sum(1 for _ in path.rglob("*") if _.is_file())
        except Exception:
            return 0

    def _parse_git_error(self, error: GitCommandError) -> str:
        """
        Converts cryptic GitPython error messages into user-friendly text.

        GitCommandError messages often contain the raw git stderr output
        which is unhelpful ("remote: Repository not found" etc.)
        """
        stderr = str(error.stderr).lower() if error.stderr else ""
        stdout = str(error.stdout).lower() if error.stdout else ""
        combined = stderr + stdout

        if "repository not found" in combined:
            return (
                "Repository not found. "
                "Confirm the URL is correct and the repository is public."
            )
        if "could not read username" in combined:
            return (
                "Repository requires authentication. "
                "CodeMind only supports public repositories."
            )
        if "invalid branch" in combined or "not found" in combined:
            return (
                f"Branch not found in repository. "
                f"Check the branch name and try again."
            )
        if "timed out" in combined or "timeout" in combined:
            return (
                "Clone timed out. The repository may be too large. "
                "Try a smaller repository or a specific branch."
            )
        if "rate limit" in combined:
            return (
                "GitHub rate limit exceeded. "
                "Add GITHUB_TOKEN to your .env file."
            )

        # Fall back to truncated raw error
        return f"Git clone failed: {str(error)[:200]}"

    def cleanup(self, project_id: str) -> None:
        """
        Removes the cloned repository from disk after processing.
        Call this after chunking and embedding are complete.
        """
        project_path = self.upload_dir / project_id
        if project_path.exists():
            shutil.rmtree(project_path)
            logger.info(f"🗑️  Cleaned up clone: {project_id}")


# Module-level singleton
github_engine = GithubIngestionEngine()