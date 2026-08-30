# backend/app/main.py
#
# CHANGE vs original:
#   warm_up() now runs as a background asyncio task AFTER the server
#   has already bound to the port and Render can detect it.
#
#   Previously: warm_up() blocked the lifespan coroutine, so the server
#   never reached "accepting requests" before hitting 512MB OOM.
#
#   Now: lifespan returns immediately → Render detects the port → server
#   is marked live → warm_up() loads the model in the background.
#   First request that needs embeddings will wait for the model if it
#   hasn't loaded yet (lazy load in _ensure_ready()).

import uuid
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import time

from app.config import settings
from app.utils.logger import get_logger
from app.utils.database import init_db

from app.api.routes.ingest import router as ingest_router, projects_router
from app.api.routes.query  import router as query_router
from app.api.routes.memory import router as memory_router

logger = get_logger(__name__)


async def _background_warmup():
    """
    Loads the embedding model in a background task.

    Runs AFTER the server has bound to its port so Render can detect
    the service is alive. The model loads on a thread pool executor
    so it doesn't block the async event loop.

    Any request that arrives before this finishes will trigger
    lazy loading in _ensure_ready() — slightly slower on that
    one request, but the service stays alive.
    """
    await asyncio.sleep(2)   # Give Render a moment to detect the port
    loop = asyncio.get_event_loop()
    try:
        from app.core.rag.embedder import embedding_engine
        logger.info("🔄 Background: loading embedding model...")
        await loop.run_in_executor(None, embedding_engine.warm_up)
        logger.info("✅ Background: embedding model ready")
    except Exception as e:
        logger.warning(
            f"Background warm-up failed (non-fatal — "
            f"model will load lazily on first request): {e}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise DB and directories, then schedule warm-up
    as a background task so the port binds BEFORE the heavy ML work.

    Shutdown: log goodbye.
    """
    logger.info("🚀 CodeMind API starting up...")

    # Lightweight setup — completes in milliseconds
    os.makedirs(settings.upload_dir,          exist_ok=True)
    os.makedirs(settings.chroma_persist_path, exist_ok=True)
    await init_db()

    logger.info(f"🌍 Environment: {settings.app_env}")
    logger.info("✅ All systems ready — accepting requests")

    # Schedule model warm-up to run AFTER we yield
    # (i.e. after the server is already live and port is bound)
    warmup_task = asyncio.create_task(_background_warmup())

    yield   # ← Server is live here. Render detects the port here.

    # Cleanup on shutdown
    warmup_task.cancel()
    try:
        await warmup_task
    except (asyncio.CancelledError, Exception):
        pass

    logger.info("👋 CodeMind API shutting down...")


app = FastAPI(
    title       = "CodeMind API",
    description = "RAG-Based AI-Powered Codebase Memory System",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [settings.frontend_url, "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request Timing Middleware ────────────────────────────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    request.state.request_id = request_id

    response   = await call_next(request)
    process_ms = (time.time() - start_time) * 1000

    response.headers["X-Request-ID"]   = request_id
    response.headers["X-Process-Time"] = f"{process_ms:.1f}ms"

    return response


# ── Global Error Handler ─────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled error [{getattr(request.state, 'request_id', '?')}] "
        f"on {request.url.path}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":      "Internal server error",
            "detail":     str(exc) if settings.app_env == "development"
                          else "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown"),
        }
    )


# ── Register All Routers ─────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(ingest_router,   prefix=API_PREFIX)
app.include_router(projects_router, prefix=API_PREFIX)
app.include_router(query_router,    prefix=API_PREFIX)
app.include_router(memory_router,   prefix=API_PREFIX)


# ── Root Endpoints ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "CodeMind API is running",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    from app.core.rag.embedder            import embedding_engine
    from app.core.llm.gemini              import gemini_client
    from app.core.llm.conversation_memory import conversation_memory

    return {
        "status":      "healthy",
        "environment": settings.app_env,
        "embedding":   embedding_engine.get_stats(),
        "llm":         gemini_client.get_stats(),
        "memory":      conversation_memory.get_stats(),
    }