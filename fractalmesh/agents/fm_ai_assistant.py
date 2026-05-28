#!/usr/bin/env python3
"""
fm_ai_assistant.py — AI Assistant / LLM Gateway (Port 7865)
FractalMesh OMEGA Titan | Unified multi-provider LLM gateway with
conversation threading, token tracking, cost estimation, and prompt templates.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import gzip
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ──────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT               = int(os.getenv("AI_ASSISTANT_PORT", "7865"))
ROOT               = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB_PATH            = ROOT / "database" / "sovereign.db"
LOG_DIR            = ROOT / "logs"
ADMIN_SECRET       = os.getenv("ADMIN_SECRET", "")

ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ── provider cost tables (USD per 1M tokens) ───────────────────────────────────
_COST_TABLE = {
    # anthropic
    "claude-sonnet-4-6":             (3.00,  15.00),
    "claude-opus-4-7":               (15.00, 75.00),
    "claude-haiku-4-5-20251001":     (0.80,   4.00),
    # openai
    "gpt-4o":                        (5.00,  15.00),
    "gpt-4o-mini":                   (0.15,   0.60),
    "gpt-3.5-turbo":                 (0.50,   1.50),
}

def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _COST_TABLE.get(model, (1.00, 3.00))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000

# ── seeded prompt templates ────────────────────────────────────────────────────
_SEED_TEMPLATES = [
    (
        "sales_email", "sales",
        "Write a compelling B2B cold outreach email for {product_name} targeting {company_name}. "
        "Address their pain point: {pain_point}. End with a clear call-to-action: {cta}. "
        "Keep it under 150 words. Be professional and value-focused.",
        '["product_name","company_name","pain_point","cta"]',
        "B2B cold outreach email template for sales prospecting",
    ),
    (
        "content_brief", "content",
        "Create a detailed SEO content brief for the topic: {topic}. "
        "Target audience: {target_audience}. Primary keywords: {keywords}. "
        "Include: suggested title, meta description, H2/H3 structure, key points to cover, "
        "internal linking opportunities, and estimated word count.",
        '["topic","target_audience","keywords"]',
        "SEO content brief with structure and keyword guidance",
    ),
    (
        "code_review", "dev",
        "Perform a systematic code review of the following {language} code. "
        "Evaluate: correctness, edge cases, performance, security vulnerabilities, "
        "readability, and adherence to best practices. Provide specific line-by-line feedback.\n\n"
        "```{language}\n{code_snippet}\n```",
        '["language","code_snippet"]',
        "Systematic code review checklist across multiple quality dimensions",
    ),
    (
        "support_response", "support",
        "Write an empathetic and helpful customer support response for the following issue: {issue}. "
        "Tone: {tone}. Include: acknowledgment of the problem, clear resolution steps, "
        "and a follow-up offer. Be concise and avoid jargon.",
        '["issue","tone"]',
        "Empathetic customer support reply template",
    ),
    (
        "deal_summary", "crm",
        "Generate a concise CRM deal summary from the following notes: {deal_notes}. "
        "Current stage: {stage}. Include: deal overview, key stakeholders mentioned, "
        "next steps, risks or blockers, and recommended actions for the sales team.",
        '["deal_notes","stage"]',
        "CRM deal summary generator from raw meeting or call notes",
    ),
]

# ── database ───────────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id              INTEGER PRIMARY KEY,
            thread_id       TEXT UNIQUE,
            title           TEXT,
            model           TEXT,
            provider        TEXT,
            system_prompt   TEXT,
            message_count   INTEGER DEFAULT 0,
            total_tokens    INTEGER DEFAULT 0,
            estimated_cost  REAL DEFAULT 0,
            created_at      REAL,
            updated_at      REAL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY,
            thread_id       TEXT,
            role            TEXT,
            content         TEXT,
            model           TEXT,
            provider        TEXT,
            input_tokens    INTEGER DEFAULT 0,
            output_tokens   INTEGER DEFAULT 0,
            cost            REAL DEFAULT 0,
            latency_ms      REAL,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE,
            category    TEXT,
            template    TEXT,
            variables   TEXT,
            description TEXT,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS usage_stats (
            id              INTEGER PRIMARY KEY,
            provider        TEXT,
            model           TEXT,
            date            TEXT,
            input_tokens    INTEGER DEFAULT 0,
            output_tokens   INTEGER DEFAULT 0,
            cost            REAL DEFAULT 0,
            request_count   INTEGER DEFAULT 0
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_stats_provider_model_date
            ON usage_stats(provider, model, date);
        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
    """)
    conn.commit()

    # Seed prompt templates if table is empty
    row = conn.execute("SELECT COUNT(*) FROM prompt_templates").fetchone()
    if row[0] == 0:
        now = time.time()
        conn.executemany(
            "INSERT OR IGNORE INTO prompt_templates "
            "(name, category, template, variables, description, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [(n, c, t, v, d, now) for n, c, t, v, d in _SEED_TEMPLATES],
        )
        conn.commit()

    conn.close()


# ── provider routing ───────────────────────────────────────────────────────────

def _select_provider(model_hint: str, provider_hint: str) -> tuple[str, str]:
    """Returns (provider, model) after routing logic."""
    anthropic_key  = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key     = os.getenv("OPENAI_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    if provider_hint:
        provider = provider_hint
        if not model_hint:
            defaults = {
                "anthropic":  "claude-sonnet-4-6",
                "openai":     "gpt-4o",
                "openrouter": "openai/gpt-4o",
                "together":   "meta-llama/Llama-3-8b-chat-hf",
            }
            model_hint = defaults.get(provider, "claude-sonnet-4-6")
        return provider, model_hint

    if model_hint.startswith("claude"):
        return "anthropic", model_hint
    if model_hint.startswith("gpt"):
        return "openai", model_hint

    # Default fallback order based on available keys
    if anthropic_key:
        return "anthropic", model_hint or "claude-sonnet-4-6"
    if openai_key:
        return "openai", model_hint or "gpt-4o"
    if openrouter_key:
        return "openrouter", model_hint or "openai/gpt-4o"
    return "together", model_hint or "meta-llama/Llama-3-8b-chat-hf"


# ── provider API calls ─────────────────────────────────────────────────────────

def _http_post(url: str, headers: dict, body: dict, timeout: int = 120) -> dict:
    """POST JSON, return parsed JSON response. Raises on HTTP error."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        # Handle gzip
        if resp.info().get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw)


def _call_anthropic(model: str, messages: list, system_prompt: str,
                    max_tokens: int) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    body: dict = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   messages,
    }
    if system_prompt:
        body["system"] = system_prompt

    result = _http_post("https://api.anthropic.com/v1/messages", headers, body)
    content      = result["content"][0]["text"]
    in_tokens    = result["usage"]["input_tokens"]
    out_tokens   = result["usage"]["output_tokens"]
    return {"content": content, "input_tokens": in_tokens, "output_tokens": out_tokens}


def _call_openai_compat(url: str, api_key: str, model: str,
                        messages: list, max_tokens: int) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":      model,
        "messages":   messages,
        "max_tokens": max_tokens,
    }
    result    = _http_post(url, headers, body)
    content   = result["choices"][0]["message"]["content"]
    in_tokens = result["usage"]["prompt_tokens"]
    out_tokens = result["usage"]["completion_tokens"]
    return {"content": content, "input_tokens": in_tokens, "output_tokens": out_tokens}


def _call_openai(model: str, messages: list, max_tokens: int) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return _call_openai_compat(
        "https://api.openai.com/v1/chat/completions", api_key, model, messages, max_tokens
    )


def _call_openrouter(model: str, messages: list, max_tokens: int) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return _call_openai_compat(
        "https://openrouter.ai/api/v1/chat/completions", api_key, model, messages, max_tokens
    )


def _call_together(model: str, messages: list, max_tokens: int) -> dict:
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set")
    return _call_openai_compat(
        "https://api.together.xyz/v1/chat/completions", api_key, model, messages, max_tokens
    )


def _dispatch(provider: str, model: str, messages: list, system_prompt: str,
              max_tokens: int) -> dict:
    """Route to correct provider and return normalized result dict."""
    if provider == "anthropic":
        return _call_anthropic(model, messages, system_prompt, max_tokens)
    # For OpenAI-compatible providers inject system prompt as first message
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    if provider == "openai":
        return _call_openai(model, messages, max_tokens)
    if provider == "openrouter":
        return _call_openrouter(model, messages, max_tokens)
    if provider == "together":
        return _call_together(model, messages, max_tokens)
    raise ValueError(f"Unknown provider: {provider}")


# ── conversation helpers ───────────────────────────────────────────────────────

def _new_thread_id() -> str:
    import secrets as _s
    return "thr_" + _s.token_hex(12)


def _get_or_create_thread(conn: sqlite3.Connection, thread_id: str | None,
                           model: str, provider: str, system_prompt: str,
                           first_message: str) -> str:
    if thread_id:
        row = conn.execute(
            "SELECT thread_id FROM conversations WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if row:
            return thread_id

    tid = thread_id or _new_thread_id()
    # Auto-title from first 60 chars of message
    title = (first_message[:57] + "...") if len(first_message) > 60 else first_message
    now = time.time()
    conn.execute(
        "INSERT INTO conversations "
        "(thread_id, title, model, provider, system_prompt, message_count, "
        " total_tokens, estimated_cost, created_at, updated_at) "
        "VALUES (?,?,?,?,?,0,0,0.0,?,?)",
        (tid, title, model, provider, system_prompt, now, now),
    )
    conn.commit()
    return tid


def _load_thread_messages(conn: sqlite3.Connection, thread_id: str) -> list:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE thread_id=? ORDER BY created_at ASC",
        (thread_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _save_message(conn: sqlite3.Connection, thread_id: str, role: str,
                  content: str, model: str, provider: str,
                  in_tok: int, out_tok: int, cost: float, latency_ms: float):
    now = time.time()
    conn.execute(
        "INSERT INTO messages "
        "(thread_id, role, content, model, provider, input_tokens, output_tokens, "
        " cost, latency_ms, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (thread_id, role, content, model, provider, in_tok, out_tok, cost, latency_ms, now),
    )


def _update_thread_stats(conn: sqlite3.Connection, thread_id: str,
                         tokens: int, cost: float):
    now = time.time()
    conn.execute(
        "UPDATE conversations SET "
        "message_count = message_count + 2, "
        "total_tokens  = total_tokens + ?, "
        "estimated_cost = estimated_cost + ?, "
        "updated_at = ? "
        "WHERE thread_id=?",
        (tokens, cost, now, thread_id),
    )


def _update_usage_stats(conn: sqlite3.Connection, provider: str, model: str,
                        in_tok: int, out_tok: int, cost: float):
    date_str = time.strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO usage_stats (provider, model, date, input_tokens, output_tokens, cost, request_count) "
        "VALUES (?,?,?,?,?,?,1) "
        "ON CONFLICT(provider, model, date) DO UPDATE SET "
        "input_tokens  = input_tokens  + excluded.input_tokens, "
        "output_tokens = output_tokens + excluded.output_tokens, "
        "cost          = cost          + excluded.cost, "
        "request_count = request_count + 1",
        (provider, model, date_str, in_tok, out_tok, cost),
    )


# ── template rendering ─────────────────────────────────────────────────────────

def _render_template(template: str, variables: dict) -> str:
    result = template
    for k, v in variables.items():
        result = result.replace("{" + k + "}", str(v))
    return result


# ── auth helper ────────────────────────────────────────────────────────────────

def _is_admin(headers) -> bool:
    secret = ADMIN_SECRET
    if not secret:
        return True  # No secret set — open admin
    token = headers.get("X-Admin-Secret") or headers.get("x-admin-secret", "")
    return hmac.compare_digest(token, secret)


# ── request handler ────────────────────────────────────────────────────────────

class AIAssistantHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-AIAssistant/1.0"

    # ── low-level helpers ──────────────────────────────────────────────────────

    def log_message(self, fmt, *args):  # suppress default access log noise
        pass

    def _send_json(self, status: int, payload: dict | list):
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str):
        self._send_json(status, {"error": message})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _path_parts(self) -> list[str]:
        return [p for p in self.path.split("?")[0].split("/") if p]

    def _query_params(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        return dict(urllib.parse.parse_qsl(parsed.query))

    # ── routing ────────────────────────────────────────────────────────────────

    def do_GET(self):
        parts = self._path_parts()
        params = self._query_params()

        if not parts:
            self._send_json(200, {"service": "AI Assistant", "status": "ok"})
            return

        route = parts[0]

        if route == "health":
            self._handle_health()
        elif route == "conversations" and len(parts) == 1:
            self._handle_list_conversations(params)
        elif route == "conversations" and len(parts) == 2:
            self._handle_get_conversation(parts[1])
        elif route == "templates" and len(parts) == 1:
            self._handle_list_templates(params)
        elif route == "templates" and len(parts) == 2:
            self._handle_get_template(parts[1])
        elif route == "usage":
            self._handle_usage()
        elif route == "models":
            self._handle_models()
        else:
            self._send_error(404, "Endpoint not found")

    def do_POST(self):
        parts = self._path_parts()
        if not parts:
            self._send_error(404, "Endpoint not found")
            return

        route = parts[0]

        if route == "chat" and len(parts) == 1:
            self._handle_chat()
        elif route == "chat" and len(parts) == 2 and parts[1] == "batch":
            self._handle_batch()
        elif route == "templates":
            self._handle_create_template()
        else:
            self._send_error(404, "Endpoint not found")

    def do_DELETE(self):
        parts = self._path_parts()
        if len(parts) == 2 and parts[0] == "conversations":
            self._handle_delete_conversation(parts[1])
        else:
            self._send_error(404, "Endpoint not found")

    # ── handlers ───────────────────────────────────────────────────────────────

    def _handle_health(self):
        conn = _db()
        row = conn.execute(
            "SELECT COUNT(*) as cnt, "
            "COALESCE(SUM(total_tokens),0) as ttok, "
            "COALESCE(SUM(estimated_cost),0) as tcost "
            "FROM conversations"
        ).fetchone()
        conn.close()
        self._send_json(200, {
            "status":              "ok",
            "uptime_seconds":      round(time.time() - START_TIME, 1),
            "conversation_count":  row["cnt"],
            "total_tokens":        row["ttok"],
            "estimated_total_cost": round(row["tcost"], 6),
            "port":                PORT,
        })

    def _handle_list_conversations(self, params: dict):
        model_filter = params.get("model", "")
        limit        = min(int(params.get("limit", "50")), 200)
        conn = _db()
        if model_filter:
            rows = conn.execute(
                "SELECT thread_id, title, model, provider, message_count, "
                "total_tokens, estimated_cost, created_at, updated_at "
                "FROM conversations WHERE model=? ORDER BY updated_at DESC LIMIT ?",
                (model_filter, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT thread_id, title, model, provider, message_count, "
                "total_tokens, estimated_cost, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        self._send_json(200, [dict(r) for r in rows])

    def _handle_get_conversation(self, thread_id: str):
        conn = _db()
        conv = conn.execute(
            "SELECT * FROM conversations WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if not conv:
            conn.close()
            self._send_error(404, "Thread not found")
            return
        msgs = conn.execute(
            "SELECT id, role, content, model, provider, input_tokens, output_tokens, "
            "cost, latency_ms, created_at "
            "FROM messages WHERE thread_id=? ORDER BY created_at ASC",
            (thread_id,),
        ).fetchall()
        conn.close()
        data = dict(conv)
        data["messages"] = [dict(m) for m in msgs]
        self._send_json(200, data)

    def _handle_list_templates(self, params: dict):
        category = params.get("category", "")
        conn = _db()
        if category:
            rows = conn.execute(
                "SELECT id, name, category, variables, description, created_at "
                "FROM prompt_templates WHERE category=? ORDER BY name",
                (category,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, category, variables, description, created_at "
                "FROM prompt_templates ORDER BY category, name"
            ).fetchall()
        conn.close()
        self._send_json(200, [dict(r) for r in rows])

    def _handle_get_template(self, name: str):
        conn = _db()
        row = conn.execute(
            "SELECT * FROM prompt_templates WHERE name=?", (name,)
        ).fetchone()
        conn.close()
        if not row:
            self._send_error(404, "Template not found")
            return
        self._send_json(200, dict(row))

    def _handle_usage(self):
        month = time.strftime("%Y-%m")
        conn = _db()
        rows = conn.execute(
            "SELECT provider, model, "
            "SUM(input_tokens) as input_tokens, "
            "SUM(output_tokens) as output_tokens, "
            "SUM(cost) as cost, "
            "SUM(request_count) as request_count "
            "FROM usage_stats WHERE date LIKE ? "
            "GROUP BY provider, model ORDER BY cost DESC",
            (month + "%",),
        ).fetchall()
        total_cost = sum(r["cost"] for r in rows)
        total_requests = sum(r["request_count"] for r in rows)
        conn.close()
        self._send_json(200, {
            "month":          month,
            "total_cost_usd": round(total_cost, 6),
            "total_requests": total_requests,
            "breakdown":      [dict(r) for r in rows],
        })

    def _handle_models(self):
        anthropic_key  = os.getenv("ANTHROPIC_API_KEY", "")
        openai_key     = os.getenv("OPENAI_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        together_key   = os.getenv("TOGETHER_API_KEY", "")

        models: dict = {}
        if anthropic_key:
            models["anthropic"] = [
                {"model": "claude-sonnet-4-6",         "description": "Claude Sonnet 4.6 — balanced speed/intelligence"},
                {"model": "claude-opus-4-7",           "description": "Claude Opus 4.7 — most capable"},
                {"model": "claude-haiku-4-5-20251001", "description": "Claude Haiku 4.5 — fast and economical"},
            ]
        if openai_key:
            models["openai"] = [
                {"model": "gpt-4o",       "description": "GPT-4o — flagship multimodal"},
                {"model": "gpt-4o-mini",  "description": "GPT-4o Mini — fast and cheap"},
                {"model": "gpt-3.5-turbo","description": "GPT-3.5 Turbo — legacy fast"},
            ]
        if openrouter_key:
            models["openrouter"] = [
                {"model": "openai/gpt-4o",                       "description": "GPT-4o via OpenRouter"},
                {"model": "anthropic/claude-3-5-sonnet",         "description": "Claude 3.5 Sonnet via OpenRouter"},
                {"model": "meta-llama/llama-3-70b-instruct",     "description": "Llama 3 70B via OpenRouter"},
                {"model": "mistralai/mistral-7b-instruct",       "description": "Mistral 7B via OpenRouter"},
            ]
        if together_key:
            models["together"] = [
                {"model": "meta-llama/Llama-3-8b-chat-hf",       "description": "Llama 3 8B"},
                {"model": "meta-llama/Llama-3-70b-chat-hf",      "description": "Llama 3 70B"},
                {"model": "mistralai/Mistral-7B-Instruct-v0.3",  "description": "Mistral 7B v0.3"},
            ]
        self._send_json(200, {"available_providers": list(models.keys()), "models": models})

    def _handle_chat(self):
        try:
            body = self._read_body()
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return

        message       = body.get("message", "").strip()
        if not message:
            # Allow template rendering path
            template_name = body.get("template_name", "")
            template_vars = body.get("template_vars", {})
            if template_name and template_vars:
                conn = _db()
                tpl = conn.execute(
                    "SELECT template FROM prompt_templates WHERE name=?", (template_name,)
                ).fetchone()
                conn.close()
                if not tpl:
                    self._send_error(400, f"Template '{template_name}' not found")
                    return
                message = _render_template(tpl["template"], template_vars)
            else:
                self._send_error(400, "message is required (or template_name + template_vars)")
                return

        # Optional: render template onto message
        elif body.get("template_name"):
            template_name = body.get("template_name", "")
            template_vars = body.get("template_vars", {})
            if template_name:
                conn = _db()
                tpl = conn.execute(
                    "SELECT template FROM prompt_templates WHERE name=?", (template_name,)
                ).fetchone()
                conn.close()
                if tpl:
                    rendered = _render_template(tpl["template"], template_vars)
                    message = rendered + "\n\n" + message

        thread_id     = body.get("thread_id", "")
        model_hint    = body.get("model", "")
        provider_hint = body.get("provider", "")
        system_prompt = body.get("system_prompt", "")
        max_tokens    = int(body.get("max_tokens", 2048))

        provider, model = _select_provider(model_hint, provider_hint)

        conn = _db()
        try:
            tid = _get_or_create_thread(conn, thread_id or None, model, provider,
                                        system_prompt, message)
            history = _load_thread_messages(conn, tid)
        except Exception as exc:
            conn.close()
            self._send_error(500, f"DB error: {exc}")
            return

        # Build message list for API call
        api_messages = history + [{"role": "user", "content": message}]

        t0 = time.time()
        try:
            result = _dispatch(provider, model, api_messages, system_prompt, max_tokens)
        except urllib.error.HTTPError as exc:
            conn.close()
            body_err = exc.read().decode(errors="replace")
            self._send_error(502, f"Provider HTTP {exc.code}: {body_err[:300]}")
            return
        except Exception as exc:
            conn.close()
            self._send_error(502, f"Provider error: {exc}")
            return

        latency_ms  = (time.time() - t0) * 1000
        in_tok      = result["input_tokens"]
        out_tok     = result["output_tokens"]
        cost        = _calc_cost(model, in_tok, out_tok)
        response    = result["content"]

        try:
            _save_message(conn, tid, "user", message, model, provider, in_tok, 0, 0.0, 0.0)
            _save_message(conn, tid, "assistant", response, model, provider,
                          in_tok, out_tok, cost, latency_ms)
            _update_thread_stats(conn, tid, in_tok + out_tok, cost)
            _update_usage_stats(conn, provider, model, in_tok, out_tok, cost)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            # Still return success — DB persistence failure is non-fatal
            conn.close()
            self._send_json(200, {
                "thread_id":  tid,
                "response":   response,
                "model":      model,
                "provider":   provider,
                "tokens":     {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
                "cost":       round(cost, 8),
                "latency_ms": round(latency_ms, 1),
                "warning":    f"DB write failed: {exc}",
            })
            return

        conn.close()
        self._send_json(200, {
            "thread_id":  tid,
            "response":   response,
            "model":      model,
            "provider":   provider,
            "tokens":     {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
            "cost":       round(cost, 8),
            "latency_ms": round(latency_ms, 1),
        })

    def _handle_batch(self):
        try:
            body = self._read_body()
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return

        prompts       = body.get("prompts", [])
        if not prompts or not isinstance(prompts, list):
            self._send_error(400, "prompts must be a non-empty list of strings")
            return

        model_hint    = body.get("model", "")
        provider_hint = body.get("provider", "")
        system_prompt = body.get("system_prompt", "")
        max_tokens    = int(body.get("max_tokens", 2048))

        provider, model = _select_provider(model_hint, provider_hint)

        results: list = [None] * len(prompts)
        errors:  list = [None] * len(prompts)

        def _worker(idx: int, prompt: str):
            msgs = [{"role": "user", "content": prompt}]
            t0 = time.time()
            try:
                res = _dispatch(provider, model, msgs, system_prompt, max_tokens)
                latency = (time.time() - t0) * 1000
                in_tok  = res["input_tokens"]
                out_tok = res["output_tokens"]
                cost    = _calc_cost(model, in_tok, out_tok)
                results[idx] = {
                    "index":      idx,
                    "prompt":     prompt,
                    "response":   res["content"],
                    "model":      model,
                    "provider":   provider,
                    "tokens":     {"input": in_tok, "output": out_tok, "total": in_tok + out_tok},
                    "cost":       round(cost, 8),
                    "latency_ms": round(latency, 1),
                }
                # Persist usage stats
                conn = _db()
                _update_usage_stats(conn, provider, model, in_tok, out_tok, cost)
                conn.commit()
                conn.close()
            except Exception as exc:
                errors[idx] = str(exc)
                results[idx] = {"index": idx, "prompt": prompt, "error": str(exc)}

        threads = []
        for i, prompt in enumerate(prompts):
            t = threading.Thread(target=_worker, args=(i, str(prompt)), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=180)

        self._send_json(200, {
            "model":    model,
            "provider": provider,
            "count":    len(prompts),
            "results":  results,
        })

    def _handle_create_template(self):
        if not _is_admin(self.headers):
            self._send_error(403, "Admin access required")
            return
        try:
            body = self._read_body()
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error(400, f"Invalid JSON: {exc}")
            return

        name        = body.get("name", "").strip()
        category    = body.get("category", "general").strip()
        template    = body.get("template", "").strip()
        variables   = body.get("variables", [])
        description = body.get("description", "").strip()

        if not name or not template:
            self._send_error(400, "name and template are required")
            return

        if isinstance(variables, list):
            variables_str = json.dumps(variables)
        else:
            variables_str = str(variables)

        now = time.time()
        conn = _db()
        try:
            conn.execute(
                "INSERT INTO prompt_templates "
                "(name, category, template, variables, description, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (name, category, template, variables_str, description, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            self._send_error(409, f"Template '{name}' already exists")
            return
        conn.close()
        self._send_json(201, {
            "created":     True,
            "name":        name,
            "category":    category,
            "description": description,
            "created_at":  now,
        })

    def _handle_delete_conversation(self, thread_id: str):
        if not _is_admin(self.headers):
            self._send_error(403, "Admin access required")
            return
        conn = _db()
        row = conn.execute(
            "SELECT id FROM conversations WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if not row:
            conn.close()
            self._send_error(404, "Thread not found")
            return
        conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        conn.execute("DELETE FROM conversations WHERE thread_id=?", (thread_id,))
        conn.commit()
        conn.close()
        self._send_json(200, {"deleted": True, "thread_id": thread_id})


# ── server bootstrap ───────────────────────────────────────────────────────────

def _run():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), AIAssistantHandler)
    print(f"[AI-Assistant] Listening on port {PORT}  db={DB_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[AI-Assistant] Shutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    _run()
