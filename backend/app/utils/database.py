# backend/app/utils/database.py

import aiosqlite
import uuid
from datetime import datetime, timezone
from typing import Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Path to our SQLite database file
DB_PATH = "./codemind.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                source_type  TEXT NOT NULL,
                source_url   TEXT,
                branch       TEXT DEFAULT 'main',
                status       TEXT NOT NULL DEFAULT 'pending',
                file_count   INTEGER DEFAULT 0,
                chunk_count  INTEGER DEFAULT 0,
                languages    TEXT DEFAULT '',
                error_msg    TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            )
        """)
        # Add branch and languages columns if upgrading existing DB
        try:
            await db.execute("ALTER TABLE projects ADD COLUMN branch TEXT DEFAULT 'main'")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE projects ADD COLUMN languages TEXT DEFAULT ''")
        except Exception:
            pass
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_projects_status
            ON projects(status)
        """)
        await db.commit()
        logger.info("✅ Database initialized")


def _now() -> str:
    """Returns current UTC time as ISO string. Always use UTC in databases."""
    return datetime.now(timezone.utc).isoformat()


async def create_project(
    name: str,
    source_type: str,
    source_url: Optional[str] = None
) -> str:
    """
    Inserts a new project row and returns its generated ID.
    UUIDs guarantee uniqueness even across multiple servers.
    """
    project_id = str(uuid.uuid4())
    now = _now()

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO projects
                (id, name, source_type, source_url, status, created_at, updated_at)
            VALUES
                (?, ?, ?, ?, 'pending', ?, ?)
        """, (project_id, name, source_type, source_url, now, now))
        await db.commit()

    logger.info(f"📝 Created project: {project_id} ({name})")
    return project_id

async def find_existing_project(
    name:        str,
    source_type: str,
    source_url:  str | None = None,
) -> dict | None:
    """
    Checks if a project with the same name and source already exists.
    Used to prevent duplicate projects from repeated uploads.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if source_url:
            # For GitHub: match by URL (most reliable)
            async with db.execute(
                """SELECT * FROM projects
                   WHERE source_url = ? AND status != 'failed'
                   ORDER BY created_at DESC LIMIT 1""",
                (source_url,)
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return dict(row)

        # For ZIP: match by name + source_type
        async with db.execute(
            """SELECT * FROM projects
               WHERE name = ? AND source_type = ? AND status != 'failed'
               ORDER BY created_at DESC LIMIT 1""",
            (name, source_type)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_project(project_id: str) -> Optional[dict]:
    """
    Fetches a single project by ID.
    Returns None if not found (caller handles the 404).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        # row_factory makes rows behave like dicts instead of tuples
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
            # dict(row) converts aiosqlite.Row → plain Python dict
            return dict(row) if row else None


async def get_all_projects() -> list[dict]:
    """Returns all projects, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def update_project_status(
    project_id: str,
    status: str,
    file_count: int = 0,
    chunk_count: int = 0,
    error_msg: Optional[str] = None
):
    """
    Updates a project's processing status and stats.
    Called by the ingestion pipeline as it progresses.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE projects
            SET status = ?, file_count = ?, chunk_count = ?,
                error_msg = ?, updated_at = ?
            WHERE id = ?
        """, (status, file_count, chunk_count, error_msg, _now(), project_id))
        await db.commit()

    logger.info(f"🔄 Project {project_id}: status → {status}")

async def update_project_metadata(
    project_id: str,
    branch:     str  = None,
    languages:  list = None,
) -> None:
    """Updates optional metadata fields."""
    async with aiosqlite.connect(DB_PATH) as db:
        if branch:
            await db.execute(
                "UPDATE projects SET branch = ? WHERE id = ?",
                (branch, project_id)
            )
        if languages:
            await db.execute(
                "UPDATE projects SET languages = ? WHERE id = ?",
                (','.join(languages), project_id)
            )
        await db.commit()

async def delete_project(project_id: str) -> bool:
    """
    Deletes a project from the registry.
    Returns True if something was deleted, False if project didn't exist.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM projects WHERE id = ?", (project_id,)
        )
        await db.commit()
        deleted = cursor.rowcount > 0

    if deleted:
        logger.info(f"🗑️  Deleted project: {project_id}")
    return deleted