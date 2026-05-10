"""
configs/settings.py
Single source of truth for all configuration.
Every module imports from here — never read os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str   = Field(default="gpt-4o", env="OPENAI_MODEL")

    # Qdrant
    qdrant_host:       str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port:       int = Field(default=6333,        env="QDRANT_PORT")
    qdrant_collection: str = Field(default="enterprise_policies", env="QDRANT_COLLECTION")

    # Embeddings
    embedding_model:      str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")

    # Reranker
    reranker_model: str = Field(default="BAAI/bge-reranker-large", env="RERANKER_MODEL")
    reranker_top_k: int = Field(default=5,  env="RERANKER_TOP_K")
    retrieval_top_k: int = Field(default=25, env="RETRIEVAL_TOP_K")

    # Chunking
    max_chunk_tokens:  int   = Field(default=512,  env="MAX_CHUNK_TOKENS")
    chunk_overlap_pct: float = Field(default=0.12, env="CHUNK_OVERLAP_PCT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
