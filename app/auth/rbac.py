"""Role-based access control."""

ROLE_PERMISSIONS: dict[str, list[str]] = {
    "finance": ["finance"],
    "marketing": ["marketing"],
    "hr": ["hr"],
    "engineering": ["engineering"],
    "employee": ["general"],
    "executive": ["finance", "marketing", "hr", "engineering", "general"],
}

SENSITIVE_TERMS: list[str] = [
    "ssn",
    "salary",
    "social security",
    "payroll",
    "account number",
    "routing number",
    "password",
    "secret",
]


def get_allowed_namespaces(role: str) -> list[str]:
    return ROLE_PERMISSIONS.get(role, [])


def authorize_query(role: str, namespace: str) -> bool:
    return namespace in get_allowed_namespaces(role)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    lower = text.lower()
    for term in SENSITIVE_TERMS:
        idx = lower.find(term)
        while idx != -1:
            redacted = redacted[:idx] + "[REDACTED]" + redacted[idx + len(term):]
            lower = redacted.lower()
            idx = lower.find(term, idx + len("[REDACTED]"))
    return redacted
