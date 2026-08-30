# backend/app/core/ingestion/zip_handler.py
#
# Handles secure extraction of ZIP files and coordinates
# the full ZIP ingestion pipeline.
#
# SECURITY: Protects against zip-slip attacks where a malicious
# ZIP contains entries like "../../../etc/crontab" that would
# write files outside the target directory.

import os
import zipfile
import shutil
from pathlib import Path

from app.core.processing.parser import parser as code_parser
from app.models.ingest_models import ParsedFile, IngestionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of files we'll extract from a single ZIP.
# Prevents ZIP bomb attacks (tiny ZIP that expands to millions of files).
MAX_FILES_IN_ZIP = 5000

# Maximum total uncompressed size (500MB)
MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024


class ZipHandler:
    """
    Manages ZIP file ingestion from upload to parsed file list.

    Usage:
        handler = ZipHandler(upload_dir="/path/to/uploads")
        result = await handler.process(project_id="proj_abc", zip_path="/path/to/file.zip")
    """

    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)

    async def process(
        self,
        project_id: str,
        zip_path: str
    ) -> IngestionResult:
        """
        Full pipeline: extract → validate → parse → return result.

        This is an async method so FastAPI can await it in a background task.
        The heavy CPU work (parsing) runs synchronously inside it, which is
        acceptable for a background task (it doesn't block request handling).
        For a production system you'd use FastAPI's run_in_threadpool().
        """
        result = IngestionResult(project_id=project_id)
        extract_path = self.upload_dir / project_id

        try:
            # ── Step 1: Validate the ZIP ──────────────────────────────────
            logger.info(f"📦 Starting ZIP ingestion: {zip_path}")
            self._validate_zip(zip_path, result)
            if result.errors:
                return result

            # ── Step 2: Extract safely ────────────────────────────────────
            logger.info(f"📂 Extracting to: {extract_path}")
            self._extract_safely(zip_path, extract_path, result)
            if result.errors:
                return result

            # ── Step 3: Find the real project root ────────────────────────
            # Many ZIPs wrap everything in a single top-level folder:
            # myproject.zip
            #   └── myproject/        ← we want THIS as the root
            #       ├── src/
            #       └── README.md
            project_root = self._find_project_root(extract_path)
            logger.info(f"🌳 Project root: {project_root}")

            # ── Step 4: Parse all source files ────────────────────────────
            logger.info("🔍 Parsing source files...")
            parsed_files = code_parser.parse_directory(
                str(project_root),
                project_id=project_id
            )

            result.total_files_found = result.total_files_found  # already set
            result.files_processed   = len(parsed_files)
            result.files_skipped     = (
                result.total_files_found - result.files_processed
            )
            result.parsed_files = parsed_files

            logger.info(
                f"✅ ZIP ingestion complete: "
                f"{result.files_processed} files parsed, "
                f"{result.files_skipped} skipped"
            )

        except Exception as e:
            error_msg = f"Ingestion failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)

        return result

    def _validate_zip(self, zip_path: str, result: IngestionResult) -> None:
        """
        Validates the ZIP file before extraction.
        Checks for corruption, ZIP bombs, and empty archives.
        """
        if not os.path.exists(zip_path):
            result.errors.append(f"ZIP file not found: {zip_path}")
            return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Check for corruption
                bad_file = zf.testzip()
                if bad_file:
                    result.errors.append(
                        f"Corrupted file in ZIP: {bad_file}"
                    )
                    return

                members = zf.infolist()

                # Check for empty archive
                if not members:
                    result.errors.append("ZIP file is empty")
                    return

                # Check file count (ZIP bomb protection)
                if len(members) > MAX_FILES_IN_ZIP:
                    result.errors.append(
                        f"ZIP contains {len(members)} files "
                        f"(max {MAX_FILES_IN_ZIP}). "
                        f"Please split into smaller archives."
                    )
                    return

                # Check uncompressed size (ZIP bomb protection)
                total_size = sum(m.file_size for m in members)
                if total_size > MAX_UNCOMPRESSED_BYTES:
                    mb = total_size // (1024 * 1024)
                    result.errors.append(
                        f"ZIP uncompressed size is {mb}MB "
                        f"(max {MAX_UNCOMPRESSED_BYTES // 1024 // 1024}MB)"
                    )
                    return

                result.total_files_found = len(members)
                logger.info(
                    f"✅ ZIP validated: {len(members)} files, "
                    f"{total_size // 1024}KB uncompressed"
                )

        except zipfile.BadZipFile:
            result.errors.append(
                "Invalid ZIP file. Make sure the file is a valid .zip archive."
            )

    def _extract_safely(
        self,
        zip_path: str,
        extract_path: Path,
        result: IngestionResult
    ) -> None:
        """
        Extracts ZIP contents with zip-slip attack protection.

        Zip-slip: a malicious ZIP can contain entries with paths like:
        ../../etc/crontab
        ../../../home/user/.ssh/authorized_keys

        If you naively call extractall(), these write to dangerous locations.
        We validate EVERY path before extracting.
        """
        # Clean up any previous extraction for this project
        if extract_path.exists():
            shutil.rmtree(extract_path)
        extract_path.mkdir(parents=True, exist_ok=True)

        # Resolve the canonical extraction target path
        # resolve() gives us the real absolute path with no ".." components
        safe_root = extract_path.resolve()

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.infolist():

                    # Build the full output path for this member
                    member_path = (safe_root / member.filename).resolve()

                    # ── ZIP-SLIP CHECK ────────────────────────────────────
                    # If the resolved path does NOT start with our safe root,
                    # this entry is trying to escape the target directory.
                    # Example:
                    #   safe_root:   /tmp/uploads/proj_abc
                    #   member_path: /tmp/uploads/proj_abc/../../etc/passwd
                    #   resolved:    /etc/passwd   ← does NOT start with safe_root!
                    try:
                        member_path.relative_to(safe_root)
                    except ValueError:
                        logger.warning(
                            f"🚨 ZIP-slip attempt blocked: {member.filename}"
                        )
                        result.errors.append(
                            f"Security: Blocked malicious path in ZIP: "
                            f"{member.filename}"
                        )
                        continue   # Skip this file, continue with others

                    # Extract this member safely
                    try:
                        zf.extract(member, safe_root)
                    except Exception as e:
                        logger.warning(
                            f"Could not extract {member.filename}: {e}"
                        )

        except Exception as e:
            result.errors.append(f"Extraction failed: {str(e)}")
            raise

    def _find_project_root(self, extract_path: Path) -> Path:
        """
        Finds the actual project root inside the extracted directory.

        Many ZIP archives look like this:
        extraction_folder/
            my-project-main/      ← wrapper folder (GitHub adds this)
                src/
                package.json
                README.md

        We want src/, package.json, README.md — not the wrapper.

        Strategy: if the extraction folder contains exactly ONE subdirectory
        and NO files at the top level, that subdirectory is the real root.
        """
        contents = list(extract_path.iterdir())

        # Filter out macOS metadata folders that appear in ZIPs made on Mac
        contents = [
            c for c in contents
            if c.name not in ('__MACOSX', '.DS_Store')
        ]

        # If there's exactly one item and it's a directory → unwrap it
        if len(contents) == 1 and contents[0].is_dir():
            logger.info(
                f"📂 Unwrapping single top-level folder: {contents[0].name}"
            )
            return contents[0]

        # Otherwise the extraction root is the project root
        return extract_path

    def cleanup(self, project_id: str) -> None:
        """
        Removes the extracted files after they've been processed.
        Call this after chunking and embedding are complete to save disk space.
        The ZIP file itself is also removed.
        """
        extract_path = self.upload_dir / project_id
        zip_path     = self.upload_dir / f"{project_id}.zip"

        if extract_path.exists():
            shutil.rmtree(extract_path)
            logger.info(f"🗑️  Cleaned up extracted files: {project_id}")

        if zip_path.exists():
            zip_path.unlink()
            logger.info(f"🗑️  Cleaned up ZIP file: {project_id}")


# Module-level singleton
def create_zip_handler(upload_dir: str) -> ZipHandler:
    return ZipHandler(upload_dir=upload_dir)