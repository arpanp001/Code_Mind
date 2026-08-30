# backend/app/api/routes/memory.py

import asyncio
from fastapi import APIRouter, HTTPException

from app.models.memory_models import (
    MemoryAddRequest, MemoryAddResponse, MemoryItem,
    MemoryListResponse, MemorySearchRequest, MemoryDeleteResponse,
    MemoryType,
)
from app.core.memory.project_memory import project_memory
from app.utils.database             import get_project
from app.utils.logger               import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])


def _entry_to_item(entry) -> MemoryItem:
    """Converts MemoryEntry to MemoryItem API response."""
    return MemoryItem(
        memory_id   = entry.memory_id,
        project_id  = entry.project_id,
        content     = entry.content,
        memory_type = MemoryType(entry.memory_type),
        tags        = entry.tags,
        title       = entry.title or None,
        created_at  = entry.created_at,
    )


@router.post("/add", response_model=MemoryAddResponse, status_code=201)
async def add_memory(request: MemoryAddRequest):
    """
    Saves a new memory (architecture decision, bug fix, or note).
    Generates an embedding and stores in ChromaDB for semantic search.
    """
    project = await get_project(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    loop = asyncio.get_event_loop()
    try:
        entry = await loop.run_in_executor(
            None,
            lambda: project_memory.add_memory(
                project_id  = request.project_id,
                content     = request.content,
                memory_type = request.memory_type.value,
                title       = request.title or "",
                tags        = request.tags,
            )
        )
    except Exception as e:
        logger.error(f"Failed to add memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return MemoryAddResponse(
        memory_id = entry.memory_id,
        status    = "saved",
        message   = "Memory saved and embedded successfully",
    )


@router.get("/{project_id}", response_model=MemoryListResponse)
async def list_memories(
    project_id:  str,
    memory_type: str | None = None,
):
    """Returns all memories for a project, newest first."""
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    loop    = asyncio.get_event_loop()
    entries = await loop.run_in_executor(
        None,
        lambda: project_memory.list_memories(project_id, memory_type)
    )

    return MemoryListResponse(
        memories   = [_entry_to_item(e) for e in entries],
        total      = len(entries),
        project_id = project_id,
    )


@router.post("/search", response_model=MemoryListResponse)
async def search_memories(request: MemorySearchRequest):
    """Semantic search through project memories."""
    project = await get_project(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    loop    = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: project_memory.search_memories(
            project_id  = request.project_id,
            query       = request.query,
            memory_type = request.memory_type.value if request.memory_type else None,
            top_k       = request.top_k,
        )
    )

    items = []
    for r in results:
        item = _entry_to_item(r.memory)
        item.relevance_score = r.similarity
        items.append(item)

    return MemoryListResponse(
        memories   = items,
        total      = len(items),
        project_id = request.project_id,
    )


@router.delete("/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    memory_id:  str,
    project_id: str,           # ← query param (from ?project_id=...)
):
    """Deletes a specific memory by ID."""
    loop    = asyncio.get_event_loop()
    deleted = await loop.run_in_executor(
        None,
        lambda: project_memory.delete_memory(project_id, memory_id)
    )

    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")

    return MemoryDeleteResponse(
        message   = "Memory deleted",
        memory_id = memory_id,
    )