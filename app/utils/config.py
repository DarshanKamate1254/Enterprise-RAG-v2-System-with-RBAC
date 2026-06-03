"""Application settings loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Vector store
    vectorstore_path: str = "db/chroma"

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retriever — RAG v2 settings
    retriever_k: int = 6           # candidates per retriever arm
    reranker_top_n: int = 4        # final docs after reranking
    hyde_enabled: bool = True      # HyDE: generate hypothetical answer for query
    hybrid_alpha: float = 0.5      # 0 = pure BM25, 1 = pure vector
    query_expansion_enabled: bool = True  # multi-query retrieval

    # Chunking — parent/child
    parent_chunk_size: int = 1024   # parent window stored for context
    child_chunk_size: int = 256     # child chunks indexed for retrieval
    chunk_overlap: int = 50

    # LLM provider — openai | huggingface
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Credentials
    openai_api_key: str | None = None
    hf_token: str | None = None

    # Flask secret key
    flask_secret_key: str = "change-me-in-production"

    model_config = {"env_file": ".env", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
