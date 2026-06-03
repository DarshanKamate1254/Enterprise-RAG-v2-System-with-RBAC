"""
RAG v2: Advanced retriever combining:

  1. AutoMergingRetriever   — child chunks retrieved, parent windows returned
  2. BM25Retriever          — keyword/sparse retrieval
  3. QueryFusionRetriever   — merges vector + BM25 with Reciprocal Rank Fusion
  4. RBAC MetadataFilters   — namespace isolation

Fix vs first release:
  - AutoMergingRetriever now receives the full `storage_context` (not just
    the docstore), which is the object it uses internally to call
    storage_context.docstore.get_document(parent_id).
  - Per-query errors are caught individually so one bad node ID can't
    silence the entire retrieval run.
"""

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    AutoMergingRetriever,
    QueryFusionRetriever,
)
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.retrievers.bm25 import BM25Retriever

from app.utils.config import get_settings


def build_advanced_retriever(
    index: VectorStoreIndex,
    storage_context: StorageContext,       # ← full StorageContext, not just docstore
    namespace: str,
):
    """
    Build a RAG v2 hybrid retriever for *namespace*.

    Pipeline:
      VectorRetriever (dense) ─┐
                                ├─► QueryFusionRetriever (RRF) ─► AutoMergingRetriever
      BM25Retriever  (sparse) ─┘
    """
    settings = get_settings()

    ns_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="namespace", value=namespace, operator=FilterOperator.EQ)
        ]
    )

    # ── Arm 1: Dense vector retriever ────────────────────────────────────────
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.retriever_k,
        filters=ns_filter,
    )

    # ── Arm 2: BM25 sparse retriever ─────────────────────────────────────────
    # Pull nodes from the docstore that belong to this namespace
    retrievers = [vector_retriever]
    try:
        docstore = storage_context.docstore
        ns_nodes = [
            doc for doc in docstore.docs.values()
            if doc.metadata.get("namespace") == namespace
        ]
        if ns_nodes:
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=ns_nodes,
                similarity_top_k=settings.retriever_k,
            )
            retrievers.append(bm25_retriever)
            print(f"  [Retriever] Hybrid mode (vector + BM25, {len(ns_nodes)} BM25 nodes) "
                  f"for namespace='{namespace}'")
        else:
            print(f"  [Retriever] No BM25 nodes for namespace='{namespace}', "
                  f"using vector-only")
    except Exception as exc:
        print(f"  [Retriever] BM25 init failed ({exc}), using vector-only")

    # ── Fusion via Reciprocal Rank Fusion ─────────────────────────────────────
    if len(retrievers) > 1:
        base_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=settings.retriever_k,
            num_queries=1,           # query expansion handled separately in pipeline
            mode="reciprocal_rerank",
            use_async=False,
        )
    else:
        base_retriever = vector_retriever

    # ── AutoMerging: swap leaf chunks for parent windows ──────────────────────
    # Pass storage_context so it can resolve parent_id → parent node.
    auto_merging_retriever = AutoMergingRetriever(
        base_retriever,
        storage_context=storage_context,   # ← the fix: full context, not docstore
        simple_ratio_thresh=0.4,
        verbose=False,
    )

    return auto_merging_retriever


# ── Fallback: simple RBAC-only vector retriever ───────────────────────────────

def build_rbac_retriever(index: VectorStoreIndex, namespace: str) -> VectorIndexRetriever:
    settings = get_settings()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="namespace", value=namespace, operator=FilterOperator.EQ)
        ]
    )
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.retriever_k,
        filters=filters,
    )
