"""Namespace tagging based on file path."""

from pathlib import Path

_NAMESPACE_RULES: list[tuple[str, str]] = [
    ("finance", "finance"),
    ("marketing", "marketing"),
    ("hr", "hr"),
    ("human_resources", "hr"),
    ("engineering", "engineering"),
    ("general", "general"),
]


def tag_document(file_path: str) -> dict:
    """
    Derive metadata for a document from its file path.

    Returns:
        source    — original file path string
        namespace — RBAC namespace
        filename  — bare file name
        file_type — 'markdown', 'excel', 'csv', or 'text'
    """
    path = Path(file_path)
    lower = str(path).lower().replace("\\", "/")

    namespace = "general"
    for fragment, ns in _NAMESPACE_RULES:
        if fragment in lower:
            namespace = ns
            break

    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        file_type = "markdown"
    elif suffix in {".xlsx", ".xls"}:
        file_type = "excel"
    elif suffix == ".csv":
        file_type = "csv"
    else:
        file_type = "text"

    return {
        "source": str(path),
        "filename": path.name,
        "namespace": namespace,
        "file_type": file_type,
    }
