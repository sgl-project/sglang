#!/usr/bin/env python3
"""
fm_chatbot_engine.py — FractalMesh OMEGA Titan AI Chatbot & Conversational Engine
Port: 7892

AI chatbot engine with persistent conversation history, persona configuration,
knowledge base integration, and widget embed support.  Uses the Anthropic
Messages API (claude-haiku-4-5) with system-prompt injection, knowledge
context prepending, and full conversation threading.

Samuel James Hiotis | ABN 56 628 117 363
"""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading  — MUST come before any os.getenv calls
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT             = int(os.environ.get("CHATBOT_ENGINE_PORT", "7892"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_SECRET     = os.environ.get("ADMIN_SECRET", "")

ROOT     = Path.home() / "fmsaas"
DB_PATH  = ROOT / "database" / "sovereign.db"
LOG_PATH = ROOT / "logs" / "chatbot_engine.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# Default persona seed data
_SEED_PERSONAS = [
    {
        "name": "FractalMesh Assistant",
        "system_prompt": (
            "You are the FractalMesh Assistant, a helpful and knowledgeable AI for the "
            "FractalMesh OMEGA Titan platform. You assist users with platform features, "
            "answer questions about services, help troubleshoot issues, and guide users "
            "through workflows. Be concise, professional, and always steer users toward "
            "the best solutions available on the platform."
        ),
        "model": "claude-haiku-4-5",
        "temperature": 0.7,
        "max_tokens": 1024,
        "welcome_message": "Welcome to FractalMesh! I'm your AI assistant. How can I help you today?",
        "color": "#6366f1",
    },
    {
        "name": "Sales Bot",
        "system_prompt": (
            "You are the FractalMesh Sales Bot, an energetic and persuasive AI designed "
            "to convert prospects into paying customers. Highlight the value proposition "
            "of FractalMesh OMEGA Titan, address objections confidently, present pricing "
            "tiers clearly, and guide prospects toward signing up. Always be enthusiastic, "
            "solution-focused, and close every conversation with a clear call to action."
        ),
        "model": "claude-haiku-4-5",
        "temperature": 0.75,
        "max_tokens": 1024,
        "welcome_message": "Hi there! Ready to supercharge your business with FractalMesh? Let's talk!",
        "color": "#10b981",
    },
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()


def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] [{level}] {msg}\n"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as fh:
                fh.write(line)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db() -> None:
    with _db_lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS personas (
                    id              INTEGER PRIMARY KEY,
                    persona_id      TEXT    UNIQUE NOT NULL,
                    name            TEXT    NOT NULL,
                    system_prompt   TEXT    NOT NULL,
                    model           TEXT    DEFAULT 'claude-haiku-4-5',
                    temperature     REAL    DEFAULT 0.7,
                    max_tokens      INTEGER DEFAULT 1024,
                    welcome_message TEXT,
                    avatar_url      TEXT,
                    color           TEXT    DEFAULT '#6366f1',
                    active          INTEGER DEFAULT 1,
                    created_at      REAL,
                    updated_at      REAL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id            INTEGER PRIMARY KEY,
                    conv_id       TEXT    UNIQUE NOT NULL,
                    persona_id    TEXT,
                    user_id       TEXT,
                    session_id    TEXT,
                    title         TEXT,
                    message_count INTEGER DEFAULT 0,
                    last_active   REAL,
                    metadata      TEXT    DEFAULT '{}',
                    created_at    REAL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY,
                    msg_id      TEXT    UNIQUE NOT NULL,
                    conv_id     TEXT,
                    role        TEXT,
                    content     TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    latency_ms  INTEGER DEFAULT 0,
                    created_at  REAL
                );

                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id         INTEGER PRIMARY KEY,
                    item_id    TEXT    UNIQUE NOT NULL,
                    persona_id TEXT,
                    title      TEXT    NOT NULL,
                    content    TEXT    NOT NULL,
                    tags       TEXT    DEFAULT '[]',
                    active     INTEGER DEFAULT 1,
                    created_at REAL,
                    updated_at REAL
                );

                CREATE TABLE IF NOT EXISTS widget_configs (
                    id              INTEGER PRIMARY KEY,
                    widget_id       TEXT    UNIQUE NOT NULL,
                    persona_id      TEXT,
                    domain          TEXT,
                    allowed_origins TEXT    DEFAULT '[]',
                    theme           TEXT    DEFAULT '{}',
                    created_at      REAL
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                    ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_persona_id
                    ON conversations(persona_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_last_active
                    ON conversations(last_active);
                CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                    ON messages(conv_id);
                CREATE INDEX IF NOT EXISTS idx_messages_created_at
                    ON messages(created_at);
                CREATE INDEX IF NOT EXISTS idx_knowledge_items_persona_id
                    ON knowledge_items(persona_id);
                CREATE INDEX IF NOT EXISTS idx_widget_configs_persona_id
                    ON widget_configs(persona_id);
            """)
            conn.commit()
            _log("INFO", "Database initialised")
        finally:
            conn.close()


def _seed_personas() -> None:
    """Insert default personas if the personas table is empty."""
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM personas").fetchone()
            if row["cnt"] > 0:
                return
            now = time.time()
            for p in _SEED_PERSONAS:
                pid = _new_id("persona")
                conn.execute(
                    """
                    INSERT INTO personas
                        (persona_id, name, system_prompt, model, temperature,
                         max_tokens, welcome_message, color, active,
                         created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,1,?,?)
                    """,
                    (
                        pid, p["name"], p["system_prompt"], p["model"],
                        p["temperature"], p["max_tokens"], p["welcome_message"],
                        p["color"], now, now,
                    ),
                )
            conn.commit()
            _log("INFO", f"Seeded {len(_SEED_PERSONAS)} default personas")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id(prefix: str = "") -> str:
    """Generate a short unique ID."""
    raw = secrets.token_hex(8)
    if prefix:
        return f"{prefix}_{raw}"
    return raw


def _ok(handler: "ChatbotHandler", data: object, status: int = 200) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _err(handler: "ChatbotHandler", msg: str, status: int = 400) -> None:
    _ok(handler, {"error": msg}, status)


def _parse_body(handler: "ChatbotHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    try:
        return json.loads(handler.rfile.read(length))
    except (json.JSONDecodeError, ValueError):
        return {}


def _check_admin(handler: "ChatbotHandler") -> bool:
    """Return True if X-Admin-Secret header matches ADMIN_SECRET."""
    if not ADMIN_SECRET:
        return True  # no secret configured — open (dev mode)
    provided = handler.headers.get("X-Admin-Secret", "")
    return hmac.compare_digest(
        provided.encode("utf-8"),
        ADMIN_SECRET.encode("utf-8"),
    )


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


# ---------------------------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------------------------

def _get_knowledge(persona_id: str) -> list:
    """Return all active knowledge items for a persona."""
    with _db_lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT title, content FROM knowledge_items "
                "WHERE persona_id=? AND active=1 ORDER BY created_at",
                (persona_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def _call_claude(persona: dict, history: list, user_message: str):
    """
    Call the Anthropic Messages API.

    Returns (assistant_text, output_tokens, latency_ms).
    Raises urllib.error.URLError or ValueError on failure.
    """
    # Build knowledge context
    knowledge = _get_knowledge(persona["persona_id"])
    system = persona["system_prompt"]
    if knowledge:
        context = "\n\n".join(
            f"## {k['title']}\n{k['content']}" for k in knowledge
        )
        system = f"{system}\n\n# Knowledge Base\n{context}"

    messages = history[-20:] + [{"role": "user", "content": user_message}]

    payload = json.dumps({
        "model": persona.get("model", "claude-haiku-4-5"),
        "max_tokens": persona.get("max_tokens", 1024),
        "temperature": persona.get("temperature", 0.7),
        "system": system,
        "messages": messages,
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
    )
    req.add_header("x-api-key", os.getenv("ANTHROPIC_API_KEY", ""))
    req.add_header("anthropic-version", "2023-06-01")
    req.add_header("content-type", "application/json")

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    latency_ms = int((time.time() - t0) * 1000)

    return data["content"][0]["text"], data["usage"]["output_tokens"], latency_ms


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------

def _get_persona(persona_id: str) -> dict:
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM personas WHERE persona_id=? AND active=1",
                (persona_id,),
            ).fetchone()
            return _row_to_dict(row)
        finally:
            conn.close()


def _get_conversation(conv_id: str) -> dict:
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conv_id=?",
                (conv_id,),
            ).fetchone()
            return _row_to_dict(row)
        finally:
            conn.close()


def _get_messages(conv_id: str, limit: int = 200) -> list:
    with _db_lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT role, content, tokens_used, latency_ms, created_at "
                "FROM messages WHERE conv_id=? ORDER BY created_at LIMIT ?",
                (conv_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def _create_conversation(persona_id: str, user_id: str,
                         session_id: str, title: str) -> str:
    conv_id = _new_id("conv")
    now = time.time()
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute(
                """
                INSERT INTO conversations
                    (conv_id, persona_id, user_id, session_id, title,
                     message_count, last_active, metadata, created_at)
                VALUES (?,?,?,?,?,0,?,?,?)
                """,
                (conv_id, persona_id, user_id, session_id,
                 title or "New Conversation", now, "{}", now),
            )
            conn.commit()
        finally:
            conn.close()
    return conv_id


def _store_message(conv_id: str, role: str, content: str,
                   tokens_used: int = 0, latency_ms: int = 0) -> str:
    msg_id = _new_id("msg")
    now = time.time()
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute(
                """
                INSERT INTO messages
                    (msg_id, conv_id, role, content,
                     tokens_used, latency_ms, created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (msg_id, conv_id, role, content, tokens_used, latency_ms, now),
            )
            conn.execute(
                """
                UPDATE conversations
                SET message_count = message_count + 1,
                    last_active = ?
                WHERE conv_id = ?
                """,
                (now, conv_id),
            )
            conn.commit()
        finally:
            conn.close()
    return msg_id


# ---------------------------------------------------------------------------
# Background daemon — cleanup stale conversations
# ---------------------------------------------------------------------------

def _cleanup_daemon() -> None:
    """
    Daemon thread: runs every 3600 s.
    Deletes empty conversations older than 30 days.
    Archives (summarises) very old message threads by removing messages that
    are older than 90 days while preserving metadata counts.
    """
    while True:
        time.sleep(3600)
        try:
            cutoff_30d = time.time() - (30 * 86400)
            cutoff_90d = time.time() - (90 * 86400)
            with _db_lock:
                conn = _get_conn()
                try:
                    # Delete conversations with zero messages and older than 30d
                    result = conn.execute(
                        """
                        DELETE FROM conversations
                        WHERE message_count = 0
                          AND last_active < ?
                        """,
                        (cutoff_30d,),
                    )
                    deleted_convs = result.rowcount

                    # Archive: remove messages older than 90 days
                    result = conn.execute(
                        "DELETE FROM messages WHERE created_at < ?",
                        (cutoff_90d,),
                    )
                    archived_msgs = result.rowcount

                    conn.commit()
                    if deleted_convs or archived_msgs:
                        _log(
                            "INFO",
                            f"Cleanup: removed {deleted_convs} stale conversations, "
                            f"archived {archived_msgs} old messages",
                        )
                finally:
                    conn.close()
        except Exception as exc:  # pylint: disable=broad-except
            _log("ERROR", f"Cleanup daemon error: {exc}")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ChatbotHandler(BaseHTTPRequestHandler):
    """
    Routes:
        GET  /health
        POST /personas
        GET  /personas
        GET  /personas/{persona_id}
        PUT  /personas/{persona_id}
        POST /chat
        GET  /conversations/{conv_id}
        GET  /conversations
        DELETE /conversations/{conv_id}
        POST /knowledge
        GET  /knowledge/{persona_id}
        DELETE /knowledge/{item_id}
        POST /widget
        GET  /widget/{widget_id}
        GET  /stats
    """

    # Suppress default request logging to avoid noise
    def log_message(self, fmt, *args):  # noqa: D102
        _log("ACCESS", fmt % args)

    # ------------------------------------------------------------------
    # CORS pre-flight
    # ------------------------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, X-Admin-Secret, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        try:
            if path == "/health":
                self._handle_health()
            elif path == "/personas":
                self._handle_list_personas()
            elif len(parts) == 2 and parts[0] == "personas":
                self._handle_get_persona(parts[1])
            elif path == "/conversations":
                self._handle_list_conversations()
            elif len(parts) == 2 and parts[0] == "conversations":
                self._handle_get_conversation(parts[1])
            elif len(parts) == 2 and parts[0] == "knowledge":
                self._handle_list_knowledge(parts[1])
            elif len(parts) == 2 and parts[0] == "widget":
                self._handle_get_widget(parts[1])
            elif path == "/stats":
                self._handle_stats()
            else:
                _err(self, "Not found", 404)
        except Exception as exc:  # pylint: disable=broad-except
            _log("ERROR", f"GET {path}: {exc}")
            _err(self, "Internal server error", 500)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")

        try:
            if path == "/personas":
                self._handle_create_persona()
            elif path == "/chat":
                self._handle_chat()
            elif path == "/knowledge":
                self._handle_add_knowledge()
            elif path == "/widget":
                self._handle_create_widget()
            else:
                _err(self, "Not found", 404)
        except Exception as exc:  # pylint: disable=broad-except
            _log("ERROR", f"POST {path}: {exc}")
            _err(self, "Internal server error", 500)

    def do_PUT(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        try:
            if len(parts) == 2 and parts[0] == "personas":
                self._handle_update_persona(parts[1])
            else:
                _err(self, "Not found", 404)
        except Exception as exc:  # pylint: disable=broad-except
            _log("ERROR", f"PUT {path}: {exc}")
            _err(self, "Internal server error", 500)

    def do_DELETE(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        try:
            if len(parts) == 2 and parts[0] == "conversations":
                self._handle_delete_conversation(parts[1])
            elif len(parts) == 2 and parts[0] == "knowledge":
                self._handle_delete_knowledge(parts[1])
            else:
                _err(self, "Not found", 404)
        except Exception as exc:  # pylint: disable=broad-except
            _log("ERROR", f"DELETE {path}: {exc}")
            _err(self, "Internal server error", 500)

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------
    def _handle_health(self):
        _ok(self, {
            "status": "ok",
            "service": "fm_chatbot_engine",
            "port": PORT,
            "uptime_s": round(time.time() - START_TIME, 1),
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # /personas  (POST — admin)
    # ------------------------------------------------------------------
    def _handle_create_persona(self):
        if not _check_admin(self):
            _err(self, "Forbidden", 403)
            return
        body = _parse_body(self)
        name = body.get("name", "").strip()
        system_prompt = body.get("system_prompt", "").strip()
        if not name or not system_prompt:
            _err(self, "name and system_prompt are required")
            return

        pid = _new_id("persona")
        now = time.time()
        model = body.get("model", "claude-haiku-4-5")
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens", 1024))
        welcome_message = body.get("welcome_message", "")
        avatar_url = body.get("avatar_url", "")
        color = body.get("color", "#6366f1")

        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO personas
                        (persona_id, name, system_prompt, model, temperature,
                         max_tokens, welcome_message, avatar_url, color,
                         active, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,1,?,?)
                    """,
                    (pid, name, system_prompt, model, temperature,
                     max_tokens, welcome_message, avatar_url, color, now, now),
                )
                conn.commit()
            finally:
                conn.close()

        _log("INFO", f"Created persona {pid} name={name!r}")
        _ok(self, {"persona_id": pid, "name": name, "created_at": now}, 201)

    # ------------------------------------------------------------------
    # /personas  (GET — public)
    # ------------------------------------------------------------------
    def _handle_list_personas(self):
        with _db_lock:
            conn = _get_conn()
            try:
                rows = conn.execute(
                    """
                    SELECT persona_id, name, model, temperature, max_tokens,
                           welcome_message, avatar_url, color, created_at
                    FROM personas
                    WHERE active=1
                    ORDER BY created_at
                    """
                ).fetchall()
                personas = [dict(r) for r in rows]
            finally:
                conn.close()
        _ok(self, {"personas": personas, "total": len(personas)})

    # ------------------------------------------------------------------
    # /personas/{persona_id}  (GET — public, no system_prompt)
    # ------------------------------------------------------------------
    def _handle_get_persona(self, persona_id: str):
        with _db_lock:
            conn = _get_conn()
            try:
                row = conn.execute(
                    """
                    SELECT persona_id, name, model, temperature, max_tokens,
                           welcome_message, avatar_url, color, active, created_at
                    FROM personas
                    WHERE persona_id=? AND active=1
                    """,
                    (persona_id,),
                ).fetchone()
            finally:
                conn.close()
        if not row:
            _err(self, "Persona not found", 404)
            return
        _ok(self, dict(row))

    # ------------------------------------------------------------------
    # /personas/{persona_id}  (PUT — admin)
    # ------------------------------------------------------------------
    def _handle_update_persona(self, persona_id: str):
        if not _check_admin(self):
            _err(self, "Forbidden", 403)
            return
        existing = _get_persona(persona_id)
        if not existing:
            _err(self, "Persona not found", 404)
            return

        body = _parse_body(self)
        now = time.time()

        # Build SET clause dynamically from provided fields
        allowed_fields = {
            "name", "system_prompt", "model", "temperature",
            "max_tokens", "welcome_message", "avatar_url", "color", "active",
        }
        updates = {}
        for field in allowed_fields:
            if field in body:
                updates[field] = body[field]

        if not updates:
            _err(self, "No updatable fields provided")
            return

        updates["updated_at"] = now
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [persona_id]

        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute(
                    f"UPDATE personas SET {set_clause} WHERE persona_id=?",
                    values,
                )
                conn.commit()
            finally:
                conn.close()

        _log("INFO", f"Updated persona {persona_id} fields={list(updates.keys())}")
        _ok(self, {"persona_id": persona_id, "updated": True, "updated_at": now})

    # ------------------------------------------------------------------
    # /chat  (POST)
    # ------------------------------------------------------------------
    def _handle_chat(self):
        body = _parse_body(self)
        persona_id = body.get("persona_id", "").strip()
        user_message = body.get("message", "").strip()
        if not persona_id or not user_message:
            _err(self, "persona_id and message are required")
            return

        persona = _get_persona(persona_id)
        if not persona:
            _err(self, "Persona not found", 404)
            return

        user_id = body.get("user_id", "anonymous")
        session_id = body.get("session_id", "")
        conv_id = body.get("conv_id", "").strip()

        # Resolve or create conversation
        if conv_id:
            conv = _get_conversation(conv_id)
            if not conv:
                _err(self, "Conversation not found", 404)
                return
        else:
            # Auto-create a new conversation; title from first ~50 chars of message
            title = user_message[:50] + ("…" if len(user_message) > 50 else "")
            conv_id = _create_conversation(persona_id, user_id, session_id, title)

        # Fetch last 20 messages as history (role/content pairs)
        raw_messages = _get_messages(conv_id, limit=20)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in raw_messages
        ]

        # Store user message first
        _store_message(conv_id, "user", user_message)

        # Call Claude
        try:
            assistant_text, tokens_used, latency_ms = _call_claude(
                persona, history, user_message
            )
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            _log("ERROR", f"Anthropic HTTP error {exc.code}: {err_body}")
            _err(self, f"AI service error: {exc.code}", 502)
            return
        except urllib.error.URLError as exc:
            _log("ERROR", f"Anthropic connection error: {exc.reason}")
            _err(self, "AI service unavailable", 503)
            return
        except (KeyError, IndexError, ValueError) as exc:
            _log("ERROR", f"Anthropic response parse error: {exc}")
            _err(self, "AI response parsing failed", 502)
            return

        # Store assistant message
        _store_message(conv_id, "assistant", assistant_text,
                       tokens_used=tokens_used, latency_ms=latency_ms)

        _log("INFO",
             f"Chat conv={conv_id} persona={persona_id} "
             f"tokens={tokens_used} latency={latency_ms}ms")

        _ok(self, {
            "conv_id": conv_id,
            "message": assistant_text,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
        })

    # ------------------------------------------------------------------
    # /conversations/{conv_id}  (GET)
    # ------------------------------------------------------------------
    def _handle_get_conversation(self, conv_id: str):
        conv = _get_conversation(conv_id)
        if not conv:
            _err(self, "Conversation not found", 404)
            return
        messages = _get_messages(conv_id)
        conv["messages"] = messages
        _ok(self, conv)

    # ------------------------------------------------------------------
    # /conversations  (GET)  — query param: user_id
    # ------------------------------------------------------------------
    def _handle_list_conversations(self):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        user_id = params.get("user_id", [None])[0]

        with _db_lock:
            conn = _get_conn()
            try:
                if user_id:
                    rows = conn.execute(
                        """
                        SELECT conv_id, persona_id, user_id, session_id,
                               title, message_count, last_active, created_at
                        FROM conversations
                        WHERE user_id=?
                        ORDER BY last_active DESC
                        LIMIT 100
                        """,
                        (user_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT conv_id, persona_id, user_id, session_id,
                               title, message_count, last_active, created_at
                        FROM conversations
                        ORDER BY last_active DESC
                        LIMIT 100
                        """
                    ).fetchall()
                convs = [dict(r) for r in rows]
            finally:
                conn.close()

        _ok(self, {"conversations": convs, "total": len(convs)})

    # ------------------------------------------------------------------
    # /conversations/{conv_id}  (DELETE)
    # ------------------------------------------------------------------
    def _handle_delete_conversation(self, conv_id: str):
        conv = _get_conversation(conv_id)
        if not conv:
            _err(self, "Conversation not found", 404)
            return

        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute(
                    "DELETE FROM messages WHERE conv_id=?", (conv_id,)
                )
                conn.execute(
                    "UPDATE conversations SET message_count=0 WHERE conv_id=?",
                    (conv_id,),
                )
                conn.commit()
            finally:
                conn.close()

        _log("INFO", f"Cleared conversation {conv_id}")
        _ok(self, {"conv_id": conv_id, "cleared": True})

    # ------------------------------------------------------------------
    # /knowledge  (POST — admin)
    # ------------------------------------------------------------------
    def _handle_add_knowledge(self):
        if not _check_admin(self):
            _err(self, "Forbidden", 403)
            return
        body = _parse_body(self)
        persona_id = body.get("persona_id", "").strip()
        title = body.get("title", "").strip()
        content = body.get("content", "").strip()
        if not persona_id or not title or not content:
            _err(self, "persona_id, title, and content are required")
            return

        persona = _get_persona(persona_id)
        if not persona:
            _err(self, "Persona not found", 404)
            return

        tags = body.get("tags", [])
        if isinstance(tags, list):
            tags_json = json.dumps(tags)
        else:
            tags_json = "[]"

        item_id = _new_id("ki")
        now = time.time()

        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO knowledge_items
                        (item_id, persona_id, title, content, tags,
                         active, created_at, updated_at)
                    VALUES (?,?,?,?,?,1,?,?)
                    """,
                    (item_id, persona_id, title, content, tags_json, now, now),
                )
                conn.commit()
            finally:
                conn.close()

        _log("INFO", f"Added knowledge item {item_id} to persona {persona_id}")
        _ok(self, {"item_id": item_id, "persona_id": persona_id, "title": title}, 201)

    # ------------------------------------------------------------------
    # /knowledge/{persona_id}  (GET)
    # ------------------------------------------------------------------
    def _handle_list_knowledge(self, persona_id: str):
        with _db_lock:
            conn = _get_conn()
            try:
                rows = conn.execute(
                    """
                    SELECT item_id, persona_id, title, content, tags,
                           active, created_at, updated_at
                    FROM knowledge_items
                    WHERE persona_id=? AND active=1
                    ORDER BY created_at
                    """,
                    (persona_id,),
                ).fetchall()
                items = [dict(r) for r in rows]
            finally:
                conn.close()

        for item in items:
            try:
                item["tags"] = json.loads(item.get("tags", "[]"))
            except (ValueError, TypeError):
                item["tags"] = []

        _ok(self, {"items": items, "total": len(items)})

    # ------------------------------------------------------------------
    # /knowledge/{item_id}  (DELETE — admin)
    # ------------------------------------------------------------------
    def _handle_delete_knowledge(self, item_id: str):
        if not _check_admin(self):
            _err(self, "Forbidden", 403)
            return

        with _db_lock:
            conn = _get_conn()
            try:
                result = conn.execute(
                    "UPDATE knowledge_items SET active=0 WHERE item_id=?",
                    (item_id,),
                )
                conn.commit()
                affected = result.rowcount
            finally:
                conn.close()

        if affected == 0:
            _err(self, "Knowledge item not found", 404)
            return

        _log("INFO", f"Deactivated knowledge item {item_id}")
        _ok(self, {"item_id": item_id, "deactivated": True})

    # ------------------------------------------------------------------
    # /widget  (POST — admin)
    # ------------------------------------------------------------------
    def _handle_create_widget(self):
        if not _check_admin(self):
            _err(self, "Forbidden", 403)
            return
        body = _parse_body(self)
        persona_id = body.get("persona_id", "").strip()
        domain = body.get("domain", "").strip()
        if not persona_id or not domain:
            _err(self, "persona_id and domain are required")
            return

        persona = _get_persona(persona_id)
        if not persona:
            _err(self, "Persona not found", 404)
            return

        allowed_origins = body.get("allowed_origins", [])
        if isinstance(allowed_origins, list):
            origins_json = json.dumps(allowed_origins)
        else:
            origins_json = "[]"

        theme = body.get("theme", {})
        if isinstance(theme, dict):
            theme_json = json.dumps(theme)
        else:
            theme_json = "{}"

        widget_id = _new_id("widget")
        now = time.time()

        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO widget_configs
                        (widget_id, persona_id, domain, allowed_origins,
                         theme, created_at)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (widget_id, persona_id, domain,
                     origins_json, theme_json, now),
                )
                conn.commit()
            finally:
                conn.close()

        # Generate embed code snippet
        embed_code = (
            f'<!-- FractalMesh Chatbot Widget -->\n'
            f'<script\n'
            f'  src="https://{domain}/fm-widget.js"\n'
            f'  data-widget-id="{widget_id}"\n'
            f'  data-api="http://localhost:{PORT}"\n'
            f'  async\n'
            f'></script>'
        )

        _log("INFO", f"Created widget {widget_id} for persona {persona_id} domain={domain}")
        _ok(self, {
            "widget_id": widget_id,
            "persona_id": persona_id,
            "domain": domain,
            "embed_code": embed_code,
            "created_at": now,
        }, 201)

    # ------------------------------------------------------------------
    # /widget/{widget_id}  (GET)
    # ------------------------------------------------------------------
    def _handle_get_widget(self, widget_id: str):
        with _db_lock:
            conn = _get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM widget_configs WHERE widget_id=?",
                    (widget_id,),
                ).fetchone()
            finally:
                conn.close()

        if not row:
            _err(self, "Widget not found", 404)
            return

        data = dict(row)
        try:
            data["allowed_origins"] = json.loads(data.get("allowed_origins", "[]"))
        except (ValueError, TypeError):
            data["allowed_origins"] = []
        try:
            data["theme"] = json.loads(data.get("theme", "{}"))
        except (ValueError, TypeError):
            data["theme"] = {}

        _ok(self, data)

    # ------------------------------------------------------------------
    # /stats  (GET — admin)
    # ------------------------------------------------------------------
    def _handle_stats(self):
        if not _check_admin(self):
            _err(self, "Forbidden", 403)
            return

        today_start = time.time() - (time.time() % 86400)

        with _db_lock:
            conn = _get_conn()
            try:
                total_convs = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM conversations"
                ).fetchone()["cnt"]

                total_msgs = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM messages"
                ).fetchone()["cnt"]

                avg_latency_row = conn.execute(
                    "SELECT AVG(latency_ms) AS avg_lat FROM messages "
                    "WHERE role='assistant' AND latency_ms > 0"
                ).fetchone()
                avg_latency = round(avg_latency_row["avg_lat"] or 0, 1)

                active_personas = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM personas WHERE active=1"
                ).fetchone()["cnt"]

                tokens_today_row = conn.execute(
                    "SELECT SUM(tokens_used) AS total "
                    "FROM messages WHERE created_at >= ?",
                    (today_start,),
                ).fetchone()
                tokens_today = tokens_today_row["total"] or 0

                total_tokens = conn.execute(
                    "SELECT SUM(tokens_used) AS total FROM messages"
                ).fetchone()["total"] or 0

                active_convs = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM conversations "
                    "WHERE last_active >= ?",
                    (time.time() - 86400,),
                ).fetchone()["cnt"]
            finally:
                conn.close()

        _ok(self, {
            "total_conversations": total_convs,
            "total_messages": total_msgs,
            "avg_response_time_ms": avg_latency,
            "active_personas": active_personas,
            "tokens_used_today": tokens_today,
            "total_tokens_used": total_tokens,
            "active_conversations_24h": active_convs,
            "uptime_s": round(time.time() - START_TIME, 1),
            "timestamp": time.time(),
        })


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def _start_cleanup_daemon() -> None:
    t = threading.Thread(target=_cleanup_daemon, name="chatbot-cleanup", daemon=True)
    t.start()
    _log("INFO", "Cleanup daemon started")


def run_server() -> None:
    _init_db()
    _seed_personas()
    _start_cleanup_daemon()

    server = HTTPServer(("0.0.0.0", PORT), ChatbotHandler)
    server.socket.setsockopt(
        __import__("socket").SOL_SOCKET,
        __import__("socket").SO_REUSEADDR,
        1,
    )
    _log("INFO", f"fm_chatbot_engine listening on port {PORT}")
    print(f"[fm_chatbot_engine] Listening on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("INFO", "Server shutdown via KeyboardInterrupt")
        print("\n[fm_chatbot_engine] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
