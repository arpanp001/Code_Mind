# backend/app/core/rag/embedder.py
import json
import hashlib
import time

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from app.models.ingest_models import CodeChunk
from app.config               import settings
from app.utils.logger         import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
MODEL_MAX_TOKENS   = 256
DEFAULT_BATCH_SIZE = 32
EMBEDDING_DIM      = 384
CACHE_DIR          = ".embedding_cache"


@dataclass
class EmbeddedChunk:
    chunk:     CodeChunk
    embedding: list[float]
    model:     str = EMBEDDING_MODEL


@dataclass
class EmbeddingResult:
    project_id:       str
    total_chunks:     int                 = 0
    embedded_chunks:  int                 = 0
    cached_chunks:    int                 = 0
    failed_chunks:    int                 = 0
    model:            str                 = EMBEDDING_MODEL
    duration_seconds: float               = 0.0
    embedded:         list[EmbeddedChunk] = field(default_factory=list)


class EmbeddingCache:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str, model: str) -> str:
        raw = f"{model}::{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str) -> Optional[list[float]]:
        key  = self._key(text, model)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                path.unlink(missing_ok=True)
        return None

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        key  = self._key(text, model)
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w") as f:
                json.dump(embedding, f)
        except IOError as e:
            logger.warning(f"Cache write failed: {e}")

    def clear_project(self, project_id: str) -> None:
        logger.debug(f"Cache clear requested for: {project_id}")

    def size(self) -> int:
        return len(list(self.cache_dir.glob("*.json")))


class EmbeddingEngine:

    def __init__(
        self,
        model_name: str  = EMBEDDING_MODEL,
        batch_size: int  = DEFAULT_BATCH_SIZE,
        use_cache:  bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_cache  = use_cache
        self._model     = None
        self._cache     = EmbeddingCache() if use_cache else None
        self._is_ready  = False

    def _load_model(self) -> None:
        if self._is_ready and self._model is not None:
            return
        logger.info(f"Loading embedding model: {self.model_name}")
        start = time.time()
        from sentence_transformers import SentenceTransformer
        self._model    = SentenceTransformer(self.model_name, device="cpu")
        self._is_ready = True
        duration = time.time() - start
        logger.info(f"Embedding model loaded in {duration:.2f}s")

    def warm_up(self) -> None:
        if self._is_ready:
            return
        self._load_model()
        try:
            self._model.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
            logger.info("Embedding model warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up encode failed (non-fatal): {e}")

    def _ensure_ready(self) -> None:
        if not self._is_ready or self._model is None:
            self._load_model()

    def embed_query(self, query: str) -> list[float]:
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

    def embed_chunks(self, chunks: list[CodeChunk], project_id: str) -> EmbeddingResult:
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

        logger.info(f"Embedding {len(chunks)} chunks for {project_id}")

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

        logger.info(f"Cache hits: {result.cached_chunks}, to embed: {len(uncached_chunks)}")

        newly_embedded: dict[str, list[float]] = {}

        for batch_start in range(0, len(uncached_chunks), self.batch_size):
            batch         = uncached_chunks[batch_start : batch_start + self.batch_size]
            batch_num     = (batch_start // self.batch_size) + 1
            total_batches = (len(uncached_chunks) + self.batch_size - 1) // self.batch_size
            logger.debug(f"Batch {batch_num}/{total_batches} ({len(batch)} chunks)")

            try:
                batch_texts    = [self._prepare_text(c) for c in batch]
                raw_embeddings = self._model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=self.batch_size,
                )
                for chunk, embedding in zip(batch, raw_embeddings):
                    embedding_list = embedding.tolist()
                    if len(embedding_list) != EMBEDDING_DIM:
                        logger.error(f"Wrong dim {len(embedding_list)} for {chunk.chunk_id}")
                        result.failed_chunks += 1
                        continue
                    newly_embedded[chunk.chunk_id] = embedding_list
                    if self.use_cache and self._cache:
                        self._cache.set(chunk.text, self.model_name, embedding_list)
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}", exc_info=True)
                result.failed_chunks += len(batch)

        all_embeddings = {**cached_results, **newly_embedded}

        for chunk in chunks:
            embedding = all_embeddings.get(chunk.chunk_id)
            if embedding is None:
                continue
            result.embedded.append(EmbeddedChunk(
                chunk=chunk, embedding=embedding, model=self.model_name
            ))

        result.embedded_chunks  = len(result.embedded)
        result.duration_seconds = time.time() - start
        rate = result.embedded_chunks / result.duration_seconds if result.duration_seconds > 0 else 0
        logger.info(
            f"Embedding complete: {result.embedded_chunks}/{result.total_chunks} "
            f"chunks in {result.duration_seconds:.2f}s ({rate:.1f} chunks/sec)"
        )
        return result

    def _prepare_text(self, chunk: CodeChunk) -> str:
        prefix = f"{chunk.language} {chunk.file_path}"
        if chunk.function_name:
            prefix += f" {chunk.function_name}"
        elif chunk.class_name:
            prefix += f" {chunk.class_name}"
        return f"{prefix}\n{chunk.text}"

    def get_stats(self) -> dict:
        return {
            "model":      self.model_name,
            "is_ready":   self._is_ready,
            "batch_size": self.batch_size,
            "cache_size": self._cache.size() if self._cache else 0,
            "dimensions": EMBEDDING_DIM,
        }


embedding_engine = EmbeddingEngine(
    model_name=settings.embedding_model,
    batch_size=DEFAULT_BATCH_SIZE,
    use_cache=True,
)