"""
RAG v2: Cross-encoder re-ranking.

After retrieval, a cross-encoder model scores every (query, chunk) pair
jointly, which is far more accurate than cosine similarity alone.
The top-N chunks after reranking are passed to the LLM.

We use LlamaIndex's SentenceTransformerRerank (local, no API key needed)
as the default. OpenAI reranker is supported as an alternative.
"""

from llama_index.core.postprocessor import SentenceTransformerRerank
from app.utils.config import get_settings

_reranker = None


def get_reranker():
    """Return a cached cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=settings.reranker_top_n,
        )
        print("  [Reranker] cross-encoder/ms-marco-MiniLM-L-6-v2 loaded")
    return _reranker
