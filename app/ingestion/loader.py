"""
RAG v2 Data ingestion pipeline using LlamaIndex.

Supports:
  .md / .txt   → plain text / markdown
  .csv         → row-per-document
  .xlsx / .xls → sheet rows

RAG v2 upgrade: Parent-Child chunking strategy.
  - Large "parent" chunks (1024 tokens) are stored in a docstore for context.
  - Small "child" chunks (256 tokens) are indexed in the vector store.
  - At query time, child chunks are retrieved, then their parent windows
    are returned to the LLM — giving focused retrieval + rich context.
"""

from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser, get_leaf_nodes, get_root_nodes

from app.ingestion.metadata import tag_document
from app.utils.config import get_settings


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_markdown(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    meta = tag_document(str(path))
    return [Document(text=text, metadata=meta)]


def _load_csv(path: Path) -> list[Document]:
    import csv
    documents: list[Document] = []
    meta_base = tag_document(str(path))
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            content = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            if content.strip():
                documents.append(Document(text=content, metadata=dict(meta_base)))
    return documents


def _load_excel(path: Path) -> list[Document]:
    import pandas as pd
    documents: list[Document] = []
    meta_base = tag_document(str(path))
    try:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name).fillna("")
            for _, row in df.iterrows():
                content = "\n".join(
                    f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()
                )
                if content.strip():
                    meta = {**meta_base, "sheet": sheet_name}
                    documents.append(Document(text=content, metadata=meta))
    except Exception as exc:
        print(f"  [WARN] Excel parse error for {path.name}: {exc}")
    return documents


# ── Public API ────────────────────────────────────────────────────────────────

def load_documents(data_path: str = "data") -> list[Document]:
    """
    Recursively load .md, .txt, .csv, .xlsx, .xls files under *data_path*.
    Returns a flat list of LlamaIndex Document objects with namespace metadata.
    """
    path = Path(data_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in: {path}")

    documents: list[Document] = []
    for file in sorted(files):
        suffix = file.suffix.lower()
        try:
            if suffix in {".md", ".txt"}:
                docs = _load_markdown(file)
            elif suffix == ".csv":
                docs = _load_csv(file)
            elif suffix in {".xlsx", ".xls"}:
                docs = _load_excel(file)
            else:
                print(f"  [SKIP] Unsupported format: {file.name}")
                continue
            documents.extend(docs)
            print(f"  Loaded {len(docs):>4} doc(s) from {file.name}")
        except Exception as exc:
            print(f"  [WARN] Skipping {file.name}: {exc}")

    return documents


def chunk_documents_hierarchical(documents: list[Document]) -> tuple[list, list]:
    """
    RAG v2: Hierarchical (Parent-Child) chunking.

    Returns:
        all_nodes  — full node list for docstore (includes parents)
        leaf_nodes — child nodes to index in vector store

    Parent nodes supply wide context; leaf nodes enable precise retrieval.
    """
    settings = get_settings()

    # HierarchicalNodeParser creates nodes at multiple granularities
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[settings.parent_chunk_size, settings.child_chunk_size]
    )

    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)          # small, precise chunks for indexing
    root_nodes = get_root_nodes(all_nodes)          # large parent windows

    print(f"  Hierarchical chunking: {len(all_nodes)} total nodes "
          f"({len(leaf_nodes)} leaf / {len(root_nodes)} root)")
    return all_nodes, leaf_nodes


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Flat chunking fallback (used by simple ingest path).
    """
    settings = get_settings()
    splitter = SentenceSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    chunks: list[Document] = []
    for node in nodes:
        doc = Document(text=node.get_content(), metadata=dict(node.metadata))
        chunks.append(doc)
    return chunks
