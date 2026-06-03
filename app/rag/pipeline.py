"""
RAG v2 pipeline using LlamaIndex.

Pipeline steps:
  1. HyDE            — embed a hypothetical answer rather than the raw question
  2. Query Expansion — generate 2 paraphrases for broader recall
  3. Hybrid Retrieval — VectorRetriever + BM25 fused with RRF
  4. AutoMerging     — swap matching leaf clusters for their parent windows
  5. Deduplication   — deduplicate across multi-query results
  6. Cross-Encoder   — rerank with ms-marco-MiniLM-L-6-v2
  7. LLM Synthesis   — generate answer from top-N chunks
  8. Redaction       — mask sensitive terms before returning

Bug fixed vs first release:
  - `load_index` now returns (index, storage_context) instead of
    (index, docstore).  AutoMergingRetriever receives the full
    storage_context so it can resolve parent IDs without crashing.
  - Each query in the expansion loop is individually try/caught so
    one bad node lookup never silences all queries.
"""

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.auth.rbac import redact_sensitive_text
from app.db.vectorstore import load_index
from app.rag.retriever import build_advanced_retriever, build_rbac_retriever
from app.rag.reranker import get_reranker
from app.rag.hyde import generate_hypothetical_document, expand_queries
from app.utils.config import get_settings

SYSTEM_PROMPT = """\
You are a secure internal company assistant.

Rules:
- Answer ONLY using the information in the provided context.
- If the context does not contain the answer, say exactly:
  "I don't have access to that information."
- Never reveal data outside the user's permitted namespace.
- Be concise and professional.
- When relevant, cite which document section the answer comes from.
"""

_index_cache: VectorStoreIndex | None = None
_storage_context_cache = None          # ← was _docstore_cache


def _get_index_and_storage_context():
    global _index_cache, _storage_context_cache
    if _index_cache is None:
        settings = get_settings()
        # load_index now returns (index, storage_context)
        _index_cache, _storage_context_cache = load_index(settings.vectorstore_path)
    return _index_cache, _storage_context_cache


def _get_llm():
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=settings.llm_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
            system_prompt=SYSTEM_PROMPT,
        )

    if provider == "huggingface":
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
        return HuggingFaceInferenceAPI(
            model_name=settings.llm_model,
            token=settings.hf_token,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. "
        "Set LLM_PROVIDER to one of: openai, huggingface."
    )


def _deduplicate_nodes(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Remove duplicates by node_id, keeping the highest score."""
    seen: dict[str, NodeWithScore] = {}
    for node in nodes:
        nid = node.node_id
        if nid not in seen or (node.score or 0) > (seen[nid].score or 0):
            seen[nid] = node
    return list(seen.values())


def run_chat(question: str, namespace: str, user: dict) -> dict:
    """
    Run the full RAG v2 pipeline for *question* restricted to *namespace*.

    Returns:
        answer           — LLM response (sensitive terms redacted)
        source_documents — list of source metadata dicts
        redacted         — True if any sensitive terms were found
        contexts         — raw retrieved text chunks (for RAGAS evaluation)
        retrieval_info   — diagnostic info about each pipeline step
    """
    settings = get_settings()
    index, storage_context = _get_index_and_storage_context()
    llm = _get_llm()

    retrieval_info: dict = {
        "hyde_used": False,
        "query_expansion_used": False,
        "reranker_used": False,
        "hybrid_used": True,
        "auto_merging_used": True,
        "queries_attempted": 1,
        "queries_succeeded": 0,
    }

    # ── Step 1: HyDE — generate hypothetical document ─────────────────────────
    retrieval_query = question
    if settings.hyde_enabled:
        enriched = generate_hypothetical_document(question, llm)
        if enriched != question:
            retrieval_query = enriched
            retrieval_info["hyde_used"] = True
            print(f"  [HyDE] Query enriched to {len(retrieval_query)} chars")

    # ── Step 2: Query Expansion ───────────────────────────────────────────────
    extra_queries: list[str] = []
    if settings.query_expansion_enabled:
        all_queries = expand_queries(question, llm, n=2)
        extra_queries = all_queries[1:]   # skip index-0 which is the original
        if extra_queries:
            retrieval_info["query_expansion_used"] = True
            retrieval_info["queries_attempted"] = 1 + len(extra_queries)
            print(f"  [QueryExpansion] {len(extra_queries)} extra queries generated")

    # ── Step 3: Build hybrid retriever ───────────────────────────────────────
    try:
        retriever = build_advanced_retriever(index, storage_context, namespace)
    except Exception as exc:
        print(f"  [Retriever] Advanced init failed ({exc}), using fallback")
        retriever = build_rbac_retriever(index, namespace)
        retrieval_info["hybrid_used"] = False
        retrieval_info["auto_merging_used"] = False

    # ── Step 4: Retrieve — primary query (HyDE-enriched) ─────────────────────
    all_nodes: list[NodeWithScore] = []

    def _safe_retrieve(query_str: str, label: str) -> list[NodeWithScore]:
        """Retrieve nodes for one query, returning [] on any error."""
        try:
            nodes = retriever.retrieve(QueryBundle(query_str=query_str))
            print(f"  [Retriever] {label}: {len(nodes)} nodes")
            retrieval_info["queries_succeeded"] = (
                retrieval_info["queries_succeeded"] + 1
            )
            return nodes
        except Exception as exc:
            # Common cause: a parent node ID referenced by a leaf is not in
            # the docstore (e.g. stale index vs docstore mismatch).
            # Log it and carry on — other queries may still succeed.
            print(f"  [Retriever] {label} failed: {exc}")
            return []

    all_nodes.extend(_safe_retrieve(retrieval_query, "primary (HyDE)"))

    # ── Step 5: Retrieve — expansion queries ─────────────────────────────────
    for i, eq in enumerate(extra_queries, start=1):
        all_nodes.extend(_safe_retrieve(eq, f"expansion-{i}"))

    all_nodes = _deduplicate_nodes(all_nodes)
    print(f"  [Retrieval] {len(all_nodes)} unique nodes after dedup")

    # ── Step 6: Cross-encoder re-ranking ─────────────────────────────────────
    ranked_nodes: list[NodeWithScore] = []
    if all_nodes:
        try:
            reranker = get_reranker()
            ranked_nodes = reranker.postprocess_nodes(
                all_nodes,
                query_bundle=QueryBundle(query_str=question),  # rank vs ORIGINAL question
            )
            retrieval_info["reranker_used"] = True
            print(f"  [Reranker] Kept {len(ranked_nodes)} nodes after reranking")
        except Exception as exc:
            print(f"  [Reranker] Failed ({exc}), using unranked nodes")
            ranked_nodes = all_nodes[:settings.reranker_top_n]
    else:
        print("  [Retrieval] No nodes retrieved — LLM will report no context")
        ranked_nodes = []

    # ── Step 7: Synthesize answer ─────────────────────────────────────────────
    response_synthesizer = get_response_synthesizer(llm=llm, verbose=False)

    if ranked_nodes:
        response = response_synthesizer.synthesize(
            query=QueryBundle(query_str=question),
            nodes=ranked_nodes,
        )
    else:
        # Nothing retrieved — synthesize with empty context so the LLM
        # returns its "I don't have access" message cleanly.
        from llama_index.core.query_engine import RetrieverQueryEngine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        response = query_engine.query(question)

    raw_answer: str = str(response)
    redacted_answer = redact_sensitive_text(raw_answer)

    source_nodes = ranked_nodes or (getattr(response, "source_nodes", None) or [])
    sources = [
        {
            "source": node.metadata.get("source"),
            "filename": node.metadata.get("filename"),
            "namespace": node.metadata.get("namespace"),
            "snippet": node.get_content()[:400],
            "score": round(float(node.score), 4) if node.score is not None else None,
        }
        for node in source_nodes
    ]

    contexts = [node.get_content() for node in source_nodes]

    return {
        "answer": redacted_answer,
        "source_documents": sources,
        "redacted": redacted_answer != raw_answer,
        "contexts": contexts,
        "retrieval_info": retrieval_info,
    }
