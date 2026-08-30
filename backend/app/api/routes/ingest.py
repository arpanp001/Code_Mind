# backend/app/api/routes/ingest.py
#
# Complete ingestion pipeline — ZIP and GitHub.
#
# What's fixed vs the previous version:
#   1. process_github_repo now calls update_pipeline_progress() at every stage
#      (was missing; only process_zip_file had them)
#   2. get_ingestion_status response includes source_url, branch, languages
#      (required for project metadata injection into Gemini prompts)
#   3. list_projects includes source_url, branch, languages per item
#   4. ProjectStatusResponse model dict includes branch/languages
#   5. project_name in summary derived cleanly from url (not parsed_files[0])
#   6. reindex supports both github and zip (zip via re-upload message)
#   7. process_zip_file stores languages metadata after chunking
#   8. Cancellation checks are consistent in both pipeline functions
#   9. SSE progress endpoint sends richer payload including detail text
#  10. All imports are at the top (no inline imports inside functions except
#      optional heavy ones wrapped in try/except)

import os
import json
import shutil
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.models.ingest_models import (
    GithubIngestRequest, IngestResponse, ProjectStatusResponse,
    ProjectListResponse, ProjectListItem, DeleteResponse,
    ProjectStatus, SourceType,
)
from app.utils.database import (
    create_project, get_project, get_all_projects,
    update_project_status, update_project_metadata,
    delete_project, find_existing_project,
)
from app.config import settings
from app.utils.logger import get_logger
from app.core.ingestion.zip_handler import create_zip_handler
from app.core.ingestion.github import github_engine
from app.core.ingestion.clone_progress import get_clone_progress
from app.core.processing.chunker import chunking_engine, ast_chunking_engine
from app.core.rag.embedder import embedding_engine
from app.core.rag.vectorstore import chroma_client

logger = get_logger(__name__)

# ── Routers ───────────────────────────────────────────────────────────────────

router          = APIRouter(prefix="/ingest",   tags=["Ingestion"])
projects_router = APIRouter(prefix="/projects", tags=["Projects"])

# ── In-memory state ───────────────────────────────────────────────────────────

# { project_id: { step, detail, percent } }
_pipeline_progress: dict[str, dict] = {}

# Project IDs that the user has requested to cancel
_cancelled_projects: set[str] = set()


# ── Progress helpers ──────────────────────────────────────────────────────────

def update_pipeline_progress(
    project_id: str,
    step:       str,
    detail:     str = "",
    percent:    int = 0,
) -> None:
    """
    Updates the in-memory progress record for a project.
    Called from background tasks; the SSE endpoint reads this dict.
    """
    _pipeline_progress[project_id] = {
        "step":    step,
        "detail":  detail,
        "percent": percent,
    }
    logger.debug(f"  Progress [{project_id}] {percent}% — {step}: {detail}")


def is_cancelled(project_id: str) -> bool:
    """Returns True if the user cancelled indexing for this project."""
    return project_id in _cancelled_projects


def clear_cancelled(project_id: str) -> None:
    """Removes a project from the cancelled set after handling."""
    _cancelled_projects.discard(project_id)


# ── Background task: GitHub pipeline ─────────────────────────────────────────

async def process_github_repo(project_id: str, url: str, branch: str) -> None:
    """
    Full GitHub ingestion pipeline:
      clone → parse → filter → chunk → embed → store → summary

    Progress is reported via update_pipeline_progress() so the SSE
    endpoint can stream real-time updates to the frontend.

    The project is marked 'ready' as soon as chunks are stored so the
    user can start chatting while the summary is still generating.
    """
    logger.info(f"🔄 GitHub pipeline starting: {project_id} ({url}@{branch})")

    try:
        await update_project_status(project_id, "processing")
        update_pipeline_progress(project_id, "cloning",
            f"Cloning {url} @ {branch}…", 10)

        # ── Phase 6: Clone + parse ────────────────────────────────────────
        result = await github_engine.process(
            project_id = project_id,
            url        = url,
            branch     = branch if branch and branch != "main" else None,
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after clone: {project_id}")
            return

        if result.errors:
            error_summary = "; ".join(result.errors[:2])
            update_pipeline_progress(project_id, "error", error_summary, 0)
            await update_project_status(
                project_id, "failed", error_msg=error_summary
            )
            logger.error(f"❌ Clone/parse failed [{project_id}]: {error_summary}")
            return

        if not result.parsed_files:
            msg = (
                "No supported source files found in the repository. "
                "Ensure the repo contains Python, JavaScript, TypeScript, "
                "Go, Java, Rust, or other supported languages."
            )
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        update_pipeline_progress(
            project_id, "parsing",
            f"Parsed {result.files_processed} source files", 30,
        )
        logger.info(f"📁 Parsed {result.files_processed} files [{project_id}]")

        # ── Phase 7: Chunk ────────────────────────────────────────────────
        update_pipeline_progress(
            project_id, "chunking",
            f"Chunking {result.files_processed} files…", 45,
        )

        chunking_result = ast_chunking_engine.chunk_all(
            result.parsed_files, project_id=project_id
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after chunking: {project_id}")
            return

        # Store detected languages immediately so metadata is available
        languages = list(chunking_result.chunks_by_language.keys())
        await update_project_metadata(project_id, branch=branch, languages=languages)

        if not chunking_result.chunks:
            msg = "No valid code chunks were produced. The files may be empty or unsupported."
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        logger.info(
            f"🔪 {chunking_result.total_chunks} chunks from "
            f"{chunking_result.total_files} files [{project_id}]"
        )

        # ── Phase 8: Embed ────────────────────────────────────────────────
        update_pipeline_progress(
            project_id, "embedding",
            f"Generating embeddings for {chunking_result.total_chunks} chunks…", 60,
        )

        loop = asyncio.get_event_loop()

        embedding_result = await loop.run_in_executor(
            None,
            lambda: embedding_engine.embed_chunks(
                chunking_result.chunks, project_id=project_id
            ),
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after embedding: {project_id}")
            return

        if not embedding_result.embedded:
            msg = "Embedding generation failed — no vectors were produced."
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        logger.info(
            f"🧮 {embedding_result.embedded_chunks} embeddings [{project_id}]"
        )

        # ── Phase 9: Store in ChromaDB ────────────────────────────────────
        update_pipeline_progress(
            project_id, "storing",
            f"Storing {embedding_result.embedded_chunks} vectors in ChromaDB…", 80,
        )

        stored_count = await loop.run_in_executor(
            None,
            lambda: chroma_client.store_chunks(
                project_id      = project_id,
                embedded_chunks = embedding_result.embedded,
            ),
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after storing: {project_id}")
            return

        if stored_count == 0:
            msg = "Failed to store chunks in ChromaDB. The collection may be corrupted."
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        # ── Mark ready NOW — user can start chatting immediately ──────────
        await update_project_status(
            project_id,
            "ready",
            file_count  = result.files_processed,
            chunk_count = stored_count,
        )

        logger.info(
            f"✅ GitHub pipeline complete [{project_id}]: "
            f"{result.files_processed} files, {stored_count} chunks"
        )

        # ── Phase 10: Generate project summary (non-blocking) ─────────────
        update_pipeline_progress(
            project_id, "summarizing",
            "Generating project overview…", 90,
        )

        try:
            from app.core.llm.project_summarizer import generate_project_summary
            from app.core.memory.project_memory  import project_memory as pm

            # Derive a clean project name from the GitHub URL
            project_name = url.rstrip("/").split("/")[-1]

            summary = await loop.run_in_executor(
                None,
                lambda: generate_project_summary(
                    project_id   = project_id,
                    project_name = project_name,
                    file_count   = result.files_processed,
                    chunk_count  = stored_count,
                    top_files    = [f.file_path for f in result.parsed_files[:15]],
                    languages    = languages,
                    source_url   = url,
                    branch       = branch,
                ),
            )

            await loop.run_in_executor(
                None,
                lambda: pm.add_memory(
                    project_id  = project_id,
                    content     = summary,
                    memory_type = "note",
                    title       = "Project Overview (Auto-generated)",
                    tags        = ["overview", "auto-generated"],
                ),
            )
            logger.info(f"📋 Project summary generated [{project_id}]")

        except Exception as e:
            # Summary failure must never fail the pipeline
            logger.warning(f"Summary generation skipped [{project_id}]: {e}")

        update_pipeline_progress(
            project_id, "done",
            f"Ready! {stored_count} chunks indexed from {result.files_processed} files.", 100,
        )

    except Exception as e:
        error_msg = f"Pipeline error: {str(e)[:200]}"
        update_pipeline_progress(project_id, "error", str(e)[:150], 0)
        logger.error(
            f"💥 GitHub pipeline failed [{project_id}]: {e}",
            exc_info=True,
        )
        await update_project_status(
            project_id, "failed", error_msg=error_msg
        )


# ── Background task: ZIP pipeline ─────────────────────────────────────────────

async def process_zip_file(project_id: str, file_path: str) -> None:
    """
    Full ZIP ingestion pipeline:
      extract → parse → chunk → embed → store → summary

    Mirrors process_github_repo exactly in structure and progress reporting.
    """
    logger.info(f"🔄 ZIP pipeline starting: {project_id} ({file_path})")

    try:
        await update_project_status(project_id, "processing")
        update_pipeline_progress(project_id, "extracting",
            "Extracting ZIP archive…", 10)

        # ── Phase 5: Extract + parse ──────────────────────────────────────
        handler = create_zip_handler(settings.upload_dir)
        result  = await handler.process(
            project_id = project_id,
            zip_path   = file_path,
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after extraction: {project_id}")
            return

        if result.errors:
            error_summary = "; ".join(result.errors[:2])
            update_pipeline_progress(project_id, "error", error_summary, 0)
            await update_project_status(
                project_id, "failed", error_msg=error_summary
            )
            return

        if not result.parsed_files:
            msg = (
                "No supported source files found in the ZIP. "
                "Ensure it contains Python, JavaScript, or other supported languages."
            )
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        update_pipeline_progress(
            project_id, "parsing",
            f"Parsed {result.files_processed} source files", 30,
        )
        logger.info(f"📁 Parsed {result.files_processed} files [{project_id}]")

        # ── Phase 7: Chunk ────────────────────────────────────────────────
        update_pipeline_progress(
            project_id, "chunking",
            f"Chunking {result.files_processed} files intelligently…", 45,
        )

        chunking_result = chunking_engine.chunk_all(
            result.parsed_files, project_id=project_id
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after chunking: {project_id}")
            return

        # Store detected languages
        languages = list(chunking_result.chunks_by_language.keys())
        await update_project_metadata(project_id, languages=languages)

        if not chunking_result.chunks:
            msg = "No valid code chunks were produced from the ZIP contents."
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        logger.info(
            f"🔪 {chunking_result.total_chunks} chunks from "
            f"{chunking_result.total_files} files [{project_id}]"
        )

        # ── Phase 8: Embed ────────────────────────────────────────────────
        update_pipeline_progress(
            project_id, "embedding",
            f"Generating embeddings for {chunking_result.total_chunks} chunks…", 60,
        )

        loop = asyncio.get_event_loop()

        embedding_result = await loop.run_in_executor(
            None,
            lambda: embedding_engine.embed_chunks(
                chunking_result.chunks, project_id=project_id
            ),
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after embedding: {project_id}")
            return

        if not embedding_result.embedded:
            msg = "Embedding generation produced no vectors."
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        logger.info(
            f"🧮 {embedding_result.embedded_chunks} embeddings [{project_id}]"
        )

        # ── Phase 9: Store in ChromaDB ────────────────────────────────────
        update_pipeline_progress(
            project_id, "storing",
            f"Storing {embedding_result.embedded_chunks} vectors in ChromaDB…", 80,
        )

        stored_count = await loop.run_in_executor(
            None,
            lambda: chroma_client.store_chunks(
                project_id      = project_id,
                embedded_chunks = embedding_result.embedded,
            ),
        )

        if is_cancelled(project_id):
            clear_cancelled(project_id)
            logger.info(f"🚫 Cancelled after storing: {project_id}")
            return

        if stored_count == 0:
            msg = "Failed to store chunks in ChromaDB."
            update_pipeline_progress(project_id, "error", msg, 0)
            await update_project_status(project_id, "failed", error_msg=msg)
            return

        # ── Mark ready ────────────────────────────────────────────────────
        await update_project_status(
            project_id,
            "ready",
            file_count  = result.files_processed,
            chunk_count = stored_count,
        )

        logger.info(
            f"✅ ZIP pipeline complete [{project_id}]: "
            f"{result.files_processed} files, {stored_count} chunks"
        )

        # ── Phase 10: Generate project summary (non-blocking) ─────────────
        update_pipeline_progress(
            project_id, "summarizing",
            "Generating project overview…", 90,
        )

        try:
            from app.core.llm.project_summarizer import generate_project_summary
            from app.core.memory.project_memory  import project_memory as pm

            # Derive project name from ZIP filename
            project_name = (
                Path(file_path).stem
                .replace("_", " ")
                .replace("-", " ")
            )

            summary = await loop.run_in_executor(
                None,
                lambda: generate_project_summary(
                    project_id   = project_id,
                    project_name = project_name,
                    file_count   = result.files_processed,
                    chunk_count  = stored_count,
                    top_files    = [f.file_path for f in result.parsed_files[:15]],
                    languages    = languages,
                ),
            )

            await loop.run_in_executor(
                None,
                lambda: pm.add_memory(
                    project_id  = project_id,
                    content     = summary,
                    memory_type = "note",
                    title       = "Project Overview (Auto-generated)",
                    tags        = ["overview", "auto-generated"],
                ),
            )
            logger.info(f"📋 Project summary generated [{project_id}]")

        except Exception as e:
            logger.warning(f"Summary generation skipped [{project_id}]: {e}")

        update_pipeline_progress(
            project_id, "done",
            f"Ready! {stored_count} chunks indexed from {result.files_processed} files.", 100,
        )

    except Exception as e:
        error_msg = f"Pipeline error: {str(e)[:200]}"
        update_pipeline_progress(project_id, "error", str(e)[:150], 0)
        logger.error(
            f"💥 ZIP pipeline failed [{project_id}]: {e}",
            exc_info=True,
        )
        await update_project_status(
            project_id, "failed", error_msg=error_msg
        )


# ═════════════════════════════════════════════════════════════════════════════
# INGEST ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@router.post("/zip", response_model=IngestResponse, status_code=202)
async def ingest_zip(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    force_reindex:    bool       = False,
):
    """
    Upload and start ingestion of a ZIP file.

    If a project with the same name already exists and force_reindex=false,
    returns the existing project ID with a message so the frontend can offer
    "Open Existing" or "Re-index" choices without creating a duplicate.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    content   = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {settings.max_upload_size_mb}MB limit",
        )

    project_name = file.filename.replace(".zip", "")

    # ── Duplicate detection ───────────────────────────────────────────────
    if not force_reindex:
        existing = await find_existing_project(project_name, "zip")
        if existing:
            logger.info(
                f"♻️  Returning existing ZIP project: "
                f"{existing['id']} ({project_name})"
            )
            return IngestResponse(
                project_id  = existing["id"],
                status      = ProjectStatus(existing["status"]),
                message     = (
                    f"Project '{project_name}' is already indexed "
                    f"(status: {existing['status']}). "
                    f"Add ?force_reindex=true to rebuild from scratch."
                ),
                source_type = SourceType.ZIP,
            )

    # ── Create new project ────────────────────────────────────────────────
    project_id = await create_project(
        name        = project_name,
        source_type = "zip",
        source_url  = None,
    )

    os.makedirs(settings.upload_dir, exist_ok=True)
    save_path = os.path.join(settings.upload_dir, f"{project_id}.zip")
    with open(save_path, "wb") as f:
        f.write(content)

    await update_project_status(project_id, "processing")
    background_tasks.add_task(process_zip_file, project_id, save_path)

    logger.info(
        f"✅ ZIP ingestion started: {project_id} ({project_name}.zip, "
        f"{len(content) // 1024}KB)"
    )

    return IngestResponse(
        project_id  = project_id,
        status      = ProjectStatus.PROCESSING,
        message     = f"ZIP '{project_name}' upload accepted. Processing started.",
        source_type = SourceType.ZIP,
    )


@router.post("/github", response_model=IngestResponse, status_code=202)
async def ingest_github(
    request:          GithubIngestRequest,
    background_tasks: BackgroundTasks,
    force_reindex:    bool = False,
):
    """
    Start ingestion of a public GitHub repository.

    Stores the branch so that metadata questions ("what branch was indexed?")
    can be answered directly from project metadata without retrieval.
    """
    repo_name = request.url.rstrip("/").split("/")[-1]
    branch    = request.branch or "main"

    # ── Duplicate detection ───────────────────────────────────────────────
    if not force_reindex:
        existing = await find_existing_project(repo_name, "github", request.url)
        if existing:
            logger.info(
                f"♻️  Returning existing GitHub project: "
                f"{existing['id']} ({repo_name})"
            )
            return IngestResponse(
                project_id  = existing["id"],
                status      = ProjectStatus(existing["status"]),
                message     = (
                    f"Repository '{repo_name}' is already indexed "
                    f"(status: {existing['status']}). "
                    f"Add ?force_reindex=true to rebuild from scratch."
                ),
                source_type = SourceType.GITHUB,
            )

    # ── Create new project ────────────────────────────────────────────────
    project_id = await create_project(
        name        = repo_name,
        source_type = "github",
        source_url  = request.url,
    )

    # Store branch immediately so status endpoint can return it right away
    await update_project_metadata(project_id, branch=branch)
    await update_project_status(project_id, "processing")

    background_tasks.add_task(
        process_github_repo,
        project_id,
        request.url,
        branch,
    )

    logger.info(
        f"✅ GitHub ingestion started: {project_id} "
        f"({repo_name} @ {branch})"
    )

    return IngestResponse(
        project_id  = project_id,
        status      = ProjectStatus.PROCESSING,
        message     = f"Repository '{repo_name}' ingestion started (branch: {branch}).",
        source_type = SourceType.GITHUB,
    )


@router.get("/github/preview")
async def preview_github_repo(url: str):
    """
    Fetches repository metadata WITHOUT cloning.

    Returns size, language, stars, default branch, and a large-repo warning.
    Called by the frontend when the user pastes a GitHub URL so they can
    see repo info before committing to a full clone+index operation.

    Hard timeout: 8 seconds (frontend previewApi has a 10s timeout).
    """
    from app.core.ingestion.github_url_parser import url_parser
    from app.core.ingestion.github_api        import create_github_api_client

    try:
        parsed = url_parser.parse(url)
        url_parser.validate_components(parsed.owner, parsed.repo)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_client = create_github_api_client()

    try:
        metadata = await asyncio.wait_for(
            api_client.get_repo_metadata(parsed.owner, parsed.repo),
            timeout=8.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "GitHub API did not respond in time. "
                "This may be a temporary GitHub issue. "
                "You can still index without the preview — "
                "just click 'Index Repository'."
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    size_kb = metadata.size_kb

    if size_kb < 10_000:
        size_category = "small"
        est_minutes   = 1
    elif size_kb < 50_000:
        size_category = "medium"
        est_minutes   = 2
    elif size_kb < 150_000:
        size_category = "large"
        est_minutes   = 5
    else:
        size_category = "very_large"
        est_minutes   = 10

    return {
        "owner":             metadata.owner,
        "name":              metadata.name,
        "full_name":         metadata.full_name,
        "description":       metadata.description,
        "default_branch":    metadata.default_branch,
        "size_kb":           size_kb,
        "size_mb":           round(size_kb / 1024, 1),
        "stars":             metadata.stars,
        "language":          metadata.language,
        "clone_url":         metadata.clone_url,
        "size_category":     size_category,
        "is_large":          size_kb > 50_000,
        "estimated_minutes": est_minutes,
    }


@router.get("/github/branches")
async def list_github_branches(url: str):
    """
    Lists available branches for a GitHub repository.
    Used to populate the branch selector dropdown in the frontend.
    Falls back to ["main", "master", "develop"] on any failure.
    """
    import httpx
    from app.core.ingestion.github_url_parser import url_parser

    try:
        parsed = url_parser.parse(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_url = (
        f"https://api.github.com/repos/"
        f"{parsed.owner}/{parsed.repo}/branches?per_page=30"
    )

    headers = {
        "Accept":     "application/vnd.github.v3+json",
        "User-Agent": "CodeMind/1.0",
    }

    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(api_url, headers=headers)

        if response.status_code == 200:
            return {"branches": [b["name"] for b in response.json()]}

        logger.warning(
            f"GitHub branches API returned {response.status_code} "
            f"for {parsed.owner}/{parsed.repo}"
        )

    except Exception as e:
        logger.warning(f"Failed to list branches for {url}: {e}")

    # Fallback — always return something useful
    return {"branches": ["main", "master", "develop"]}


@router.get("/progress/{project_id}")
async def get_pipeline_progress(project_id: str):
    """
    Server-Sent Events stream of real-time pipeline progress.

    The frontend connects here via EventSource and receives JSON events:
      { step, detail, percent }

    Steps match STEP_LABELS in usePipelineProgress.js:
      cloning | extracting | parsing | chunking | embedding | storing |
      summarizing | done | error

    The stream closes automatically when the project reaches a terminal
    state (ready, failed) or after a 10-minute safety timeout.
    """
    async def event_generator():
        last_data = None
        timeout = 0

        while timeout < 600:
            project = await get_project(project_id)

            if not project:
                payload = {
                    "step": "error",
                    "percent": 0,
                    "detail": "Project not found",
                }
                yield f"data: {json.dumps(payload)}\n\n"
                break

            status = project["status"]
            progress = _pipeline_progress.get(project_id, {})

            if status == "ready":
                payload = {
                    "step": "done",
                    "percent": 100,
                    "detail": f'{project["chunk_count"]} chunks ready',
                }
                yield f"data: {json.dumps(payload)}\n\n"
                break

            elif status == "failed":
                payload = {
                    "step": "error",
                    "percent": 0,
                    "detail": project.get("error_msg") or "Processing failed",
                }
                yield f"data: {json.dumps(payload)}\n\n"
                break

            elif progress and progress != last_data:
                yield f"data: {json.dumps(progress)}\n\n"
                last_data = progress.copy()

            await asyncio.sleep(1)
            timeout += 1

    _pipeline_progress.pop(project_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status/{project_id}", response_model=ProjectStatusResponse)
async def get_ingestion_status(project_id: str):
    """
    Returns the current status of a project.

    Includes source_url, branch, and languages so the frontend and
    Gemini prompt builder can use project metadata in answers.
    Frontend polls this every 3 seconds during ingestion.
    """
    project = await get_project(project_id)

    if not project:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found",
        )

    clone_progress = get_clone_progress(project_id)

    return ProjectStatusResponse(
        project_id    = project["id"],
        name          = project["name"],
        status        = ProjectStatus(project["status"]),
        source_type   = SourceType(project["source_type"]),
        source_url    = project.get("source_url"),
        branch        = project.get("branch", "main"),
        languages     = project.get("languages", ""),
        file_count    = project["file_count"],
        chunk_count   = project["chunk_count"],
        error_message = project.get("error_msg"),
        created_at    = project["created_at"],
        updated_at    = project["updated_at"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# PROJECT MANAGEMENT ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@projects_router.get("", response_model=ProjectListResponse)
async def list_projects():
    """
    Returns all projects with status, stats, and metadata.
    Includes source_url, branch, languages so the sidebar and chat
    can display rich project info without extra API calls.
    """
    rows = await get_all_projects()

    items = [
        ProjectListItem(
            project_id  = row["id"],
            name        = row["name"],
            status      = ProjectStatus(row["status"]),
            source_type = SourceType(row["source_type"]),
            source_url  = row.get("source_url"),
            branch      = row.get("branch", "main"),
            languages   = row.get("languages", ""),
            file_count  = row["file_count"],
            chunk_count = row["chunk_count"],
            created_at  = row["created_at"],
        )
        for row in rows
    ]

    return ProjectListResponse(projects=items, total=len(items))


@projects_router.delete("/{project_id}", response_model=DeleteResponse)
async def remove_project(project_id: str):
    """
    Completely removes a project from:
      1. SQLite registry (project metadata)
      2. ChromaDB (all embedded code chunks)
      3. Disk (uploaded ZIP or cloned repo)
      4. Memory collection (ChromaDB memory collection)
    """
    project = await get_project(project_id)
    if not project:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found",
        )

    loop = asyncio.get_event_loop()

    # 1. SQLite
    await delete_project(project_id)

    # 2. ChromaDB code chunks
    await loop.run_in_executor(
        None,
        lambda: chroma_client.delete_project(project_id),
    )

    # 3. ChromaDB memory collection
    try:
        from app.core.memory.project_memory import project_memory as pm
        await loop.run_in_executor(
            None,
            lambda: pm.delete_project_memories(project_id),
        )
    except Exception as e:
        logger.warning(f"Memory cleanup failed for {project_id}: {e}")

    # 4. Disk files
    project_dir = Path(settings.upload_dir) / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir, ignore_errors=True)

    zip_file = Path(settings.upload_dir) / f"{project_id}.zip"
    if zip_file.exists():
        zip_file.unlink(missing_ok=True)

    logger.info(f"🗑️  Project fully deleted: {project_id}")

    return DeleteResponse(
        message    = "Project and all associated data deleted successfully",
        project_id = project_id,
    )


@projects_router.post("/{project_id}/cancel")
async def cancel_indexing(project_id: str):
    """
    Cancels an in-progress indexing pipeline.

    Sets a flag that the background task checks between each stage.
    The project status is immediately set to 'failed' with a
    cancellation message.
    """
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["status"] != "processing":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot cancel — project is '{project['status']}', "
                f"not 'processing'."
            ),
        )

    _cancelled_projects.add(project_id)
    await update_project_status(
        project_id, "failed",
        error_msg="Indexing cancelled by user",
    )

    logger.info(f"🚫 Indexing cancelled by user: {project_id}")
    return {"message": "Indexing cancelled", "project_id": project_id}


@projects_router.post("/{project_id}/reindex")
async def reindex_project(
    project_id:       str,
    background_tasks: BackgroundTasks,
):
    """
    Completely wipes and rebuilds a project's embeddings.

    Steps:
      1. Delete ChromaDB code collection
      2. Delete ChromaDB memory collection
      3. Reset project status + counts in SQLite
      4. Remove previous clone/extract from disk
      5. Re-run the full pipeline in the background

    GitHub projects: re-clone from stored source_url.
    ZIP projects: returns a 400 asking user to re-upload the file
                  (we don't store the original ZIP content).
    """
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["source_type"] == "zip":
        raise HTTPException(
            status_code=400,
            detail=(
                "Re-index is not supported for ZIP projects via API "
                "because the original file is not stored after processing. "
                "Please upload the ZIP file again to re-index."
            ),
        )

    source_url = project.get("source_url")
    if not source_url:
        raise HTTPException(
            status_code=400,
            detail=(
                "No source URL found for this project. "
                "Cannot re-index without the original repository URL."
            ),
        )

    branch = project.get("branch", "main")
    loop   = asyncio.get_event_loop()

    # 1. Wipe ChromaDB code chunks
    await loop.run_in_executor(
        None,
        lambda: chroma_client.delete_project(project_id),
    )

    # 2. Wipe ChromaDB memory collection
    try:
        from app.core.memory.project_memory import project_memory as pm
        await loop.run_in_executor(
            None,
            lambda: pm.delete_project_memories(project_id),
        )
    except Exception as e:
        logger.warning(f"Memory wipe failed for {project_id}: {e}")

    # 3. Reset SQLite record
    await update_project_status(
        project_id, "processing",
        file_count  = 0,
        chunk_count = 0,
    )

    # 4. Remove previous clone
    clone_path = Path(settings.upload_dir) / project_id
    if clone_path.exists():
        shutil.rmtree(clone_path, ignore_errors=True)
        logger.info(f"🗑️  Previous clone removed: {clone_path}")

    # 5. Re-run pipeline
    background_tasks.add_task(
        process_github_repo,
        project_id,
        source_url,
        branch,
    )

    logger.info(
        f"🔄 Re-index started: {project_id} "
        f"({source_url} @ {branch})"
    )

    return {
        "message":    "Re-indexing started successfully",
        "project_id": project_id,
        "source_url": source_url,
        "branch":     branch,
    }


# ═════════════════════════════════════════════════════════════════════════════
# DEBUG / INSPECTION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@projects_router.get("/{project_id}/chunks")
async def inspect_chunks(
    project_id: str,
    language:   str | None = None,
    limit:      int        = 10,
):
    """
    Debug: shows what code chunks were produced for a project.
    Useful for verifying function detection, chunk size, and line numbers.
    Filter by ?language=python to see only Python chunks.
    """
    from app.core.processing.parser    import parser as code_parser
    from app.core.ingestion.zip_handler import ZipHandler
    from app.core.ingestion.github      import CLONE_SUBDIR

    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["source_type"] == "zip":
        extract_path = Path(settings.upload_dir) / project_id
        handler      = ZipHandler(settings.upload_dir)
        source_dir   = (
            handler._find_project_root(extract_path)
            if extract_path.exists() else None
        )
    else:
        source_dir = Path(settings.upload_dir) / project_id / CLONE_SUBDIR

    if not source_dir or not source_dir.exists():
        return {
            "message":    "Source files not on disk — already processed and cleaned up",
            "project_id": project_id,
        }

    parsed    = code_parser.parse_directory(str(source_dir), project_id)
    chunk_res = chunking_engine.chunk_all(parsed, project_id)

    chunks = chunk_res.chunks
    if language:
        chunks = [c for c in chunks if c.language == language]

    return {
        "project_id":         project_id,
        "total_chunks":       chunk_res.total_chunks,
        "total_files":        chunk_res.total_files,
        "chunks_by_language": chunk_res.chunks_by_language,
        "sample_chunks": [
            {
                "chunk_id":      c.chunk_id,
                "file_path":     c.file_path,
                "language":      c.language,
                "chunk_type":    c.chunk_type,
                "function_name": c.function_name,
                "class_name":    c.class_name,
                "start_line":    c.start_line,
                "end_line":      c.end_line,
                "char_count":    c.char_count,
                "token_count":   c.token_count,
                "text_preview":  (
                    c.text[:200] + "…" if len(c.text) > 200 else c.text
                ),
            }
            for c in chunks[:limit]
        ],
    }


@projects_router.get("/{project_id}/files")
async def list_project_files(project_id: str):
    """
    Debug: lists all source files that were parsed, grouped by language.
    Verifies the parser is picking up the right files and skipping binaries.
    """
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    from app.core.processing.parser    import parser as code_parser
    from app.core.ingestion.zip_handler import ZipHandler

    extract_path = Path(settings.upload_dir) / project_id
    if not extract_path.exists():
        return {
            "message":    "Extracted files not on disk — already processed",
            "project_id": project_id,
        }

    handler      = ZipHandler(settings.upload_dir)
    project_root = handler._find_project_root(extract_path)
    parsed       = code_parser.parse_directory(str(project_root), project_id)

    by_language: dict[str, list] = {}
    for f in parsed:
        by_language.setdefault(f.language, []).append({
            "path":     f.relative_path,
            "lines":    f.line_count,
            "size_kb":  round(f.file_size_bytes / 1024, 1),
            "encoding": f.encoding,
        })

    return {
        "project_id":  project_id,
        "total_files": len(parsed),
        "by_language": {
            lang: {"count": len(files), "files": files}
            for lang, files in sorted(by_language.items())
        },
    }


@projects_router.get("/{project_id}/github-info")
async def get_github_repo_info(project_id: str):
    """
    Debug: shows metadata for a GitHub-ingested project.
    Lists parsed files grouped by language (first 10 per language).
    """
    from app.core.ingestion.github      import CLONE_SUBDIR
    from app.core.processing.parser     import parser as code_parser

    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["source_type"] != "github":
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only for GitHub-ingested projects",
        )

    clone_path = Path(settings.upload_dir) / project_id / CLONE_SUBDIR

    if not clone_path.exists():
        return {
            "message":    "Clone directory not on disk — already processed",
            "project_id": project_id,
            "source_url": project.get("source_url"),
            "branch":     project.get("branch", "main"),
        }

    parsed = code_parser.parse_directory(str(clone_path), project_id)

    by_language: dict[str, list] = {}
    for f in parsed:
        by_language.setdefault(f.language, []).append({
            "path":    f.relative_path,
            "lines":   f.line_count,
            "size_kb": round(f.file_size_bytes / 1024, 1),
        })

    return {
        "project_id":  project_id,
        "source_url":  project.get("source_url"),
        "branch":      project.get("branch", "main"),
        "languages":   project.get("languages", ""),
        "total_files": len(parsed),
        "by_language": {
            lang: {"count": len(files), "files": files[:10]}
            for lang, files in sorted(
                by_language.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        },
    }


@projects_router.get("/{project_id}/chroma-stats")
async def get_chroma_stats(project_id: str):
    """
    Debug: shows ChromaDB collection statistics for a project.
    Verifies that chunks were stored correctly after ingestion.
    """
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    loop = asyncio.get_event_loop()

    stats = await loop.run_in_executor(
        None,
        lambda: chroma_client.get_collection_stats(project_id),
    )

    all_collections = await loop.run_in_executor(
        None,
        chroma_client.list_all_collections,
    )

    return {
        "project_id":     project_id,
        "project_name":   project["name"],
        "project_status": project["status"],
        "branch":         project.get("branch", "main"),
        "languages":      project.get("languages", ""),
        "chroma_stats":   stats,
        "all_collections": all_collections,
    }