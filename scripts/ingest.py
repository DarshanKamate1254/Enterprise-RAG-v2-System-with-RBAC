"""
RAG v2 Ingestion script — run once to build the ChromaDB vector store.

Usage (from the project root):
    python scripts/ingest.py

RAG v2 changes:
  - Uses HierarchicalNodeParser (parent-child chunking)
  - Stores all nodes (parents + leaves) in a SimpleDocumentStore
  - Only leaf (child) nodes are indexed in ChromaDB
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loader import load_documents, chunk_documents_hierarchical
from app.db.vectorstore import create_index
from app.utils.config import get_settings


def main() -> None:
    settings = get_settings()

    print("=" * 60)
    print("  Company RAG v2 — LlamaIndex Ingestion Pipeline")
    print("=" * 60)
    print(f"\n  Config:")
    print(f"    embedding_model   : {settings.embedding_model}")
    print(f"    parent_chunk_size : {settings.parent_chunk_size}")
    print(f"    child_chunk_size  : {settings.child_chunk_size}")
    print(f"    vectorstore_path  : {settings.vectorstore_path}")

    print("\n[1/3] Loading documents from 'data/' ...")
    documents = load_documents("data")

    if not documents:
        print("  No documents loaded. Add files to data/ and retry.")
        sys.exit(1)

    print(f"\n  Total documents loaded: {len(documents)}")

    # Namespace distribution
    ns_counts: dict[str, int] = {}
    for doc in documents:
        ns = doc.metadata.get("namespace", "unknown")
        ns_counts[ns] = ns_counts.get(ns, 0) + 1
    print("\n  Namespace distribution (raw documents):")
    for ns, count in sorted(ns_counts.items()):
        print(f"    {ns:<20} {count} documents")

    print(f"\n[2/3] Hierarchical chunking (parent-child) ...")
    all_nodes, leaf_nodes = chunk_documents_hierarchical(documents)

    # Leaf namespace distribution
    leaf_ns: dict[str, int] = {}
    for n in leaf_nodes:
        ns = n.metadata.get("namespace", "unknown")
        leaf_ns[ns] = leaf_ns.get(ns, 0) + 1
    print("\n  Leaf node namespace distribution:")
    for ns, count in sorted(leaf_ns.items()):
        print(f"    {ns:<20} {count} leaf nodes")

    print(f"\n[3/3] Building vector store at '{settings.vectorstore_path}' ...")
    create_index(all_nodes, leaf_nodes, persist_directory=settings.vectorstore_path)

    print(
        f"\n✓ Done — {len(leaf_nodes)} leaf nodes indexed, "
        f"{len(all_nodes)} total nodes in docstore.\n"
        f"  Run 'python main.py' to start the Flask server.\n"
    )


if __name__ == "__main__":
    main()