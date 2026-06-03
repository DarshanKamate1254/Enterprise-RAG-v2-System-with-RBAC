"""
Flask application with session-based login and RBAC chat.

Routes:
  GET  /           → redirect to /login or /chat
  GET  /login      → login form
  POST /login      → authenticate and set session
  GET  /logout     → clear session
  GET  /chat       → chat UI (requires login)
  POST /api/chat   → JSON chat endpoint (requires login)
"""

from functools import wraps
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from app.auth.rbac import authorize_query, get_allowed_namespaces
from app.auth.users import authenticate, get_user
from app.rag.pipeline import run_chat
from app.utils.config import get_settings


def create_app() -> Flask:
    settings = get_settings()
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.secret_key = settings.flask_secret_key

    # ── Auth helpers ──────────────────────────────────────────────────────────

    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        if "user_id" in session:
            return redirect(url_for("chat_ui"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = authenticate(username, password)
            if user:
                session["user_id"] = username
                session["user_name"] = user["name"]
                session["user_role"] = user["role"]
                return redirect(url_for("chat_ui"))
            error = "Invalid username or password."
        return render_template("login.html", error=error)

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/chat")
    @login_required
    def chat_ui():
        user_id = session["user_id"]
        user = get_user(user_id)
        allowed_ns = get_allowed_namespaces(user["role"])
        return render_template(
            "chat.html",
            user_name=session["user_name"],
            user_role=session["user_role"],
            allowed_namespaces=allowed_ns,
            default_namespace=user["default_namespace"],
        )

    @app.post("/api/chat")
    @login_required
    def api_chat():
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        namespace = data.get("namespace") or None

        if not question:
            return jsonify({"error": "question is required"}), 400

        user_id = session["user_id"]
        user = get_user(user_id)

        if namespace is None:
            namespace = user["default_namespace"]

        if not authorize_query(user["role"], namespace):
            allowed = get_allowed_namespaces(user["role"])
            return jsonify({
                "error": (
                    f"Role '{user['role']}' cannot access namespace '{namespace}'. "
                    f"Allowed: {allowed}"
                )
            }), 403

        try:
            result = run_chat(question, namespace, user)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({
            "answer": result["answer"],
            "source_documents": result["source_documents"],
            "redacted": result["redacted"],
            "namespace_used": namespace,
            "retrieval_info": result.get("retrieval_info", {}),
        })

    return app
