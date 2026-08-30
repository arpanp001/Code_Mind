# backend/app/config.py

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # Gemini API
    gemini_api_key: str
    gemini_model:   str = "models/gemini-2.5-flash"

    # App settings
    app_env:  str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ChromaDB
    chroma_persist_path:      str = "./chroma_db"
    chroma_collection_code:   str = "code_chunks"
    chroma_collection_memory: str = "project_memory"

    # File upload
    upload_dir:         str = "./uploads"
    max_upload_size_mb: int = 100

    # RAG settings
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_results:   int = 5
    chunk_size:      int = 500
    chunk_overlap:   int = 50

    # CORS
    frontend_url: str = "http://localhost:5173"

    class Config:
        env_file       = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()