# backend/app/core/rag/embedder.py
#
# CHANGE vs original: Lazy loading — model loads on FIRST USE, not at import.
# This fixes "Out of memory (512Mi)" on Render free tier because the server
# can start, bind to a port, and accept requests before the heavy model loads.

import os
import json
import hashlib
import time
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.models.ingest_models import CodeChunk
from app.config               import settings
from app.utils.logger         import get_logger

logger = get_logger(__name__)


EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
MODEL_MAX_TOKENS = 256
DEFAULT_BATCH_SIZE = 8
EMBEDDING_DIM    = 384
CACHE_DIR        = ".embedding_cache"


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    chunk:     CodeChunk
    embedding: list[float]
    model:     str = EMBEDDING_MODEL


@dataclass
class EmbeddingResult:
    project_id:       str
    total_chunks:     int                  = 0
    embedded_chunks:  int                  = 0
    cached_chunks:    int                  = 0
    failed_chunks:    int                  = 0
    model:            str                  = EMBEDDING_MODEL
    duration_seconds: float                = 0.0
    embedded:         list[EmbeddedChunk]  = field(default_factory=list)


# ── Embedding Cache ────────────────────────────────────────────────────────────

class EmbeddingCache:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str, model: str) -> str:
        raw = f"{model}::{text}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def get(self, text: str, model: str) -> Optional[list[float]]:
        key  = self._key(text, model)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                path.unlink(missing_ok=True)
        return None

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        key  = self._key(text, model)
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, 'w') as f:
                json.dump(embedding, f)
        except IOError as e:
            logger.warning(f"Cache write failed: {e}")

    def clear_project(self, project_id: str) -> None:
        logger.debug(f"Cache clear requested for: {project_id}")

    def size(self) -> int:
        return len(list(self.cache_dir.glob("*.json")))


# ── Embedding Engine ───────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Generates embeddings for code chunks using Sentence Transformers.

    KEY CHANGE FOR RENDER FREE TIER:
    The model is loaded LAZILY — on the first embed_query() or embed_chunks()
    call, NOT at startup. This lets Render detect the open port before
    512MB RAM is consumed by PyTorch + the model weights.

    warm_up() still exists for backward compatibility but is now a no-op
    by default unless called explicitly. The lifespan in main.py calls it
    in a background task AFTER the server is already accepting requests.
    """

    def __init__(
        self,
        model_name: str  = EMBEDDING_MODEL,
        batch_size: int  = DEFAULT_BATCH_SIZE,
        use_cache:  bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_cache  = use_cache
        self._model: Optional[SentenceTransformer] = None
        self._cache = EmbeddingCache() if use_cache else None
        self._is_ready = False

    def _load_model(self) -> None:
        """
        Internal: loads the model if not already loaded.
        Thread-safe enough for single-worker Render deployments.
        """
        if self._is_ready and self._model is not None:
            return

        logger.info(f"🔄 Loading embedding model: {self.model_name}")
        start = time.time()

        self._model = SentenceTransformer(
            self.model_name,
            device="cpu",
        )
        self._is_ready = True
        duration = time.time() - start

        logger.info(
            f"✅ Embedding model loaded in {duration:.2f}s "
            f"({EMBEDDING_DIM} dimensions)"
        )

    def warm_up(self) -> None:
        """
        Loads the model into memory.

        CHANGED: Now simply calls _load_model() which is lazy.
        Call this from main.py lifespan as a background task
        so it runs AFTER the server binds to a port.
        """
        if self._is_ready:
            return
        self._load_model()
        # Warm-up encode to pre-load any lazy model components
        try:
            self._model.encode(
                ["warmup"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            logger.info("✅ Embedding model warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up encode failed (non-fatal): {e}")

    def _ensure_ready(self) -> None:
        """Load the embedding model only when it is actually needed."""
    if not self._is_ready or self._model is None:
        self.warm_up()

    def embed_query(self, query: str) -> list[float]:
        """Embeds a single search query. Auto-loads model if needed."""
        self._ensure_ready()

        if not query.strip():
            raise ValueError("Query cannot be empty")

        vector = self._model.encode(
            [query.strip()],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector[0].tolist()

    def embed_chunks(
        self,
        chunks:     list[CodeChunk],
        project_id: str,
    ) -> EmbeddingResult:
        """Embeds a list of CodeChunks in batches. Auto-loads model if needed."""
        self._ensure_ready()

        start  = time.time()
        result = EmbeddingResult(
            project_id   = project_id,
            total_chunks = len(chunks),
            model        = self.model_name,
        )

        if not chunks:
            logger.warning(f"embed_chunks called with empty list [{project_id}]")
            return result

        logger.info(
            f"🔢 Embedding {len(chunks)} chunks "
            f"for project {project_id} "
            f"(batch_size={self.batch_size})"
        )

        # ── Step 1: Separate cached vs uncached chunks ────────────────────
        cached_results:  dict[str, list[float]] = {}
        uncached_chunks: list[CodeChunk]        = []

        if self.use_cache and self._cache:
            for chunk in chunks:
                cached = self._cache.get(chunk.text, self.model_name)
                if cached is not None:
                    cached_results[chunk.chunk_id] = cached
                    result.cached_chunks += 1
                else:
                    uncached_chunks.append(chunk)
        else:
            uncached_chunks = chunks

        logger.info(
            f"  Cache hits: {result.cached_chunks}, "
            f"to embed: {len(uncached_chunks)}"
        )

        # ── Step 2: Embed uncached chunks in batches ──────────────────────
        newly_embedded: dict[str, list[float]] = {}

        for batch_start in range(0, len(uncached_chunks), self.batch_size):
            batch      = uncached_chunks[batch_start : batch_start + self.batch_size]
            batch_num  = (batch_start // self.batch_size) + 1
            total_batches = (
                len(uncached_chunks) + self.batch_size - 1
            ) // self.batch_size

            logger.debug(
                f"  Batch {batch_num}/{total_batches} "
                f"({len(batch)} chunks)"
            )

            try:
                batch_texts    = [self._prepare_text(c) for c in batch]
                raw_embeddings = self._model.encode(
                    batch_texts,
                    normalize_embeddings = True,
                    show_progress_bar    = False,
                    convert_to_numpy     = True,
                    batch_size           = self.batch_size,
                )

                for chunk, embedding in zip(batch, raw_embeddings):
                    embedding_list = embedding.tolist()

                    if len(embedding_list) != EMBEDDING_DIM:
                        logger.error(
                            f"Unexpected embedding dim "
                            f"{len(embedding_list)} for {chunk.chunk_id}"
                        )
                        result.failed_chunks += 1
                        continue

                    newly_embedded[chunk.chunk_id] = embedding_list

                    if self.use_cache and self._cache:
                        self._cache.set(
                            chunk.text,
                            self.model_name,
                            embedding_list,
                        )

            except Exception as e:
                logger.error(
                    f"Batch {batch_num} embedding failed: {e}",
                    exc_info=True
                )
                result.failed_chunks += len(batch)

        # ── Step 3: Assemble final EmbeddedChunk list ─────────────────────
        all_embeddings = {**cached_results, **newly_embedded}

        for chunk in chunks:
            embedding = all_embeddings.get(chunk.chunk_id)
            if embedding is None:
                continue
            result.embedded.append(EmbeddedChunk(
                chunk     = chunk,
                embedding = embedding,
                model     = self.model_name,
            ))

        result.embedded_chunks  = len(result.embedded)
        result.duration_seconds = time.time() - start

        rate = (
            result.embedded_chunks / result.duration_seconds
            if result.duration_seconds > 0 else 0
        )
        logger.info(
            f"✅ Embedding complete: "
            f"{result.embedded_chunks}/{result.total_chunks} chunks "
            f"in {result.duration_seconds:.2f}s "
            f"({rate:.1f} chunks/sec)"
        )
        if result.cached_chunks:
            logger.info(f"   Cache hits: {result.cached_chunks}")
        if result.failed_chunks:
            logger.warning(f"   Failed: {result.failed_chunks}")

        return result

    def _prepare_text(self, chunk: CodeChunk) -> str:
        """Prepares chunk text for embedding with file/language context prefix."""
        prefix = f"{chunk.language} {chunk.file_path}"
        if chunk.function_name:
            prefix += f" {chunk.function_name}"
        elif chunk.class_name:
            prefix += f" {chunk.class_name}"

        full_text = f"{prefix}\n{chunk.text}"

        if len(full_text) > 600:
            logger.debug(
                f"Long chunk ({len(full_text)} chars) may be truncated: "
                f"{chunk.file_path}:{chunk.chunk_index}"
            )
        return full_text

    def get_stats(self) -> dict:
        return {
            "model":      self.model_name,
            "is_ready":   self._is_ready,
            "batch_size": self.batch_size,
            "cache_size": self._cache.size() if self._cache else 0,
            "dimensions": EMBEDDING_DIM,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
# Model is NOT loaded here — it loads lazily on first use.
# This keeps import-time RAM near zero.

embedding_engine = EmbeddingEngine(
    model_name = settings.embedding_model,
    batch_size = DEFAULT_BATCH_SIZE,
    use_cache  = True,
)
