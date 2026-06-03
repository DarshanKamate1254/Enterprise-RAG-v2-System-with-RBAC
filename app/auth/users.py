"""User store with password-based authentication.

Passwords are stored as bcrypt hashes. In production, replace this
in-memory store with a real database.

Default credentials for each user:
  hr_user   / hr_pass123
  alice     / finance_pass
  bob       / marketing_pass
  dave      / engineering_pass
  eve       / employee_pass
  ceo       / ceo_pass
"""

import hashlib
import hmac

# Simple SHA-256 based hashing for demo (replace with bcrypt in production)
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


USERS: dict[str, dict] = {
    "hr_user": {
        "name": "HR Manager",
        "role": "hr",
        "default_namespace": "hr",
        "password_hash": _hash_password("hr_pass123"),
    },
    "alice": {
        "name": "Alice Finance",
        "role": "finance",
        "default_namespace": "finance",
        "password_hash": _hash_password("finance_pass"),
    },
    "bob": {
        "name": "Bob Marketing",
        "role": "marketing",
        "default_namespace": "marketing",
        "password_hash": _hash_password("marketing_pass"),
    },
    "dave": {
        "name": "Dave Engineer",
        "role": "engineering",
        "default_namespace": "engineering",
        "password_hash": _hash_password("engineering_pass"),
    },
    "eve": {
        "name": "Eve Employee",
        "role": "employee",
        "default_namespace": "general",
        "password_hash": _hash_password("employee_pass"),
    },
    "ceo": {
        "name": "Chief Executive",
        "role": "executive",
        "default_namespace": "general",
        "password_hash": _hash_password("ceo_pass"),
    },
}


def get_user(user_id: str) -> dict | None:
    """Return user dict for *user_id* (without password hash), or None."""
    user = USERS.get(user_id)
    if user is None:
        return None
    # Return a copy without the password hash
    return {k: v for k, v in user.items() if k != "password_hash"}


def authenticate(username: str, password: str) -> dict | None:
    """
    Verify *username* / *password*.
    Returns the user dict (without password_hash) on success, None on failure.
    """
    user = USERS.get(username)
    if user is None:
        return None
    expected = user["password_hash"]
    provided = _hash_password(password)
    if hmac.compare_digest(expected, provided):
        return {k: v for k, v in user.items() if k != "password_hash"}
    return None
