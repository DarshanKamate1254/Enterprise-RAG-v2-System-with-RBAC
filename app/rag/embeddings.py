"""Embedding model factory for LlamaIndex."""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.utils.config import get_settings

_embed_model = None


def get_embed_model() -> HuggingFaceEmbedding:
    global _embed_model
    if _embed_model is None:
        settings = get_settings()
        _embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            embed_batch_size=64,
        )
    return _embed_model
