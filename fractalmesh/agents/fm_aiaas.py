#!/usr/bin/env python3
"""
fm_aiaas.py — AI-as-a-Service Hosting Agent (Port 7830)
FractalMesh OMEGA Titan | Multi-provider AI inference gateway.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import os
import secrets
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT             = int(os.getenv("AIAAS_PORT", "7830"))
ROOT             = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB               = ROOT / "database" / "sovereign.db"
LOG_FILE         = ROOT / "logs" / "aiaas.log"
IMAGES_DIR       = ROOT / "aiaas" / "images"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
TOGETHER_API_KEY  = os.getenv("TOGETHER_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ADMIN_SECRET      = os.getenv("ADMIN_SECRET", "")

ROOT.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AIAAS] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("aiaas")

# ── tier definitions ──────────────────────────────────────────────────────────
TIER_LIMITS = {
    "free":       {"token_limit": 100_000,    "req_day": 50},
    "starter":    {"token_limit": 1_000_000,  "req_day": 500},
    "pro":        {"token_limit": 10_000_000, "req_day": 5000},
    "enterprise": {"token_limit": 999_999_999, "req_day": 999_999},
}

# ── seed model data ───────────────────────────────────────────────────────────
_SEED_MODELS = [
    ("claude-3-5-haiku",      "anthropic",    "Claude 3.5 Haiku",         200000, 0.80, 4.00,  "chat,code,analysis", "active"),
    ("gpt-4o-mini",           "openai",       "GPT-4o Mini",              128000, 0.15, 0.60,  "chat,code",         "active"),
    ("mistral-7b-instruct",   "openrouter",   "Mistral 7B Instruct",       32000, 0.07, 0.07,  "chat,code",         "active"),
    ("llama-3-70b-instruct",  "openrouter",   "Llama 3 70B Instruct",       8000, 0.52, 0.75,  "chat",              "active"),
    ("bge-small-en",          "huggingface",  "BGE Small EN v1.5",           512, 0.0,  0.0,   "embeddings",        "active"),
    ("sdxl-base-1.0",         "huggingface",  "Stable Diffusion XL",          77, 0.0,  0.0,   "image_gen",         "active"),
    ("starcoder2-7b",         "huggingface",  "StarCoder2 7B",             16384, 0.0,  0.0,   "code_gen",          "active"),
    ("bart-large-cnn",        "huggingface",  "BART Large CNN",             1024, 0.0,  0.0,   "summarization",     "active"),
]

# ── database ──────────────────────────────────────────────────────────────────

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    conn = _db_connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS aiaas_keys (
            id              INTEGER PRIMARY KEY,
            key             TEXT UNIQUE,
            owner           TEXT,
            email           TEXT,
            tier            TEXT DEFAULT 'free',
            tokens_used     INTEGER DEFAULT 0,
            token_limit     INTEGER DEFAULT 100000,
            requests_today  INTEGER DEFAULT 0,
            created_at      REAL,
            last_used       REAL
        );
        CREATE TABLE IF NOT EXISTS aiaas_requests (
            id              INTEGER PRIMARY KEY,
            api_key         TEXT,
            model           TEXT,
            endpoint        TEXT,
            input_tokens    INTEGER,
            output_tokens   INTEGER,
            latency_ms      REAL,
            cost_usd        REAL,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS aiaas_models (
            id                  INTEGER PRIMARY KEY,
            model_id            TEXT UNIQUE,
            provider            TEXT,
            display_name        TEXT,
            context_length      INTEGER,
            cost_per_1k_input   REAL,
            cost_per_1k_output  REAL,
            capabilities        TEXT,
            status              TEXT
        );
    """)
    conn.commit()
    # Seed models
    for row in _SEED_MODELS:
        conn.execute(
            "INSERT OR IGNORE INTO aiaas_models "
            "(model_id, provider, display_name, context_length, "
            "cost_per_1k_input, cost_per_1k_output, capabilities, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            row,
        )
    conn.commit()
    conn.close()
    log.info("DB tables verified / models seeded at %s", DB)


# ── helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _log_request(key: str, model: str, endpoint: str,
                 in_tok: int, out_tok: int, latency: float, cost: float):
    try:
        conn = _db_connect()
        conn.execute(
            "INSERT INTO aiaas_requests "
            "(api_key, model, endpoint, input_tokens, output_tokens, latency_ms, cost_usd, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (key, model, endpoint, in_tok, out_tok, latency, cost, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("_log_request failed: %s", exc)


def _check_auth(handler) -> tuple:
    """
    Validate X-API-Key header against aiaas_keys.
    Returns (row_dict, None) on success or (None, error_message) on failure.
    The row_dict is a plain dict copy of the sqlite3.Row.
    """
    api_key = handler.headers.get("X-API-Key", "").strip()
    if not api_key:
        return None, "Missing X-API-Key header"
    try:
        conn = _db_connect()
        row = conn.execute(
            "SELECT * FROM aiaas_keys WHERE key=?", (api_key,)
        ).fetchone()
        conn.close()
    except Exception as exc:
        return None, f"DB error: {exc}"
    if not row:
        return None, "Invalid API key"
    row = dict(row)
    # Check token limit
    if row["tier"] != "enterprise" and row["tokens_used"] >= row["token_limit"]:
        return None, "Token limit exceeded for this billing period"
    # Check daily requests
    tier_cfg = TIER_LIMITS.get(row["tier"], TIER_LIMITS["free"])
    if row["tier"] != "enterprise" and row["requests_today"] >= tier_cfg["req_day"]:
        return None, "Daily request limit exceeded"
    # Increment requests_today and last_used
    try:
        conn = _db_connect()
        conn.execute(
            "UPDATE aiaas_keys SET requests_today=requests_today+1, last_used=? WHERE key=?",
            (time.time(), api_key),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("Failed to increment requests_today: %s", exc)
    return row, None


def _hf_post(model_path: str, body_dict: dict) -> dict:
    """POST to HF Inference API; returns parsed JSON response."""
    url = f"https://api-inference.huggingface.co/models/{model_path}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body_dict).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HF API error {exc.code}: {body}") from exc


def _hf_post_binary(model_path: str, body_dict: dict) -> bytes:
    """POST to HF Inference API; returns raw bytes (for image generation)."""
    url = f"https://api-inference.huggingface.co/models/{model_path}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body_dict).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HF API error {exc.code}: {body}") from exc


def _anthropic_chat(messages: list, max_tokens: int, model: str) -> tuple:
    """
    Call Anthropic Messages API.
    Returns (text, input_tokens, output_tokens).
    """
    url = "https://api.anthropic.com/v1/messages"
    # Map generic model name to versioned model ID
    if model.startswith("claude-3-5-haiku"):
        versioned = "claude-3-5-haiku-20241022"
    elif model.startswith("claude-3-5-sonnet"):
        versioned = "claude-3-5-sonnet-20241022"
    elif model.startswith("claude-3-opus"):
        versioned = "claude-3-opus-20240229"
    else:
        versioned = "claude-3-5-haiku-20241022"

    payload = {
        "model": versioned,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error {exc.code}: {body}") from exc

    text = ""
    if result.get("content"):
        for block in result["content"]:
            if block.get("type") == "text":
                text += block.get("text", "")
    usage = result.get("usage", {})
    in_tok = usage.get("input_tokens", _estimate_tokens(str(messages)))
    out_tok = usage.get("output_tokens", _estimate_tokens(text))
    return text, in_tok, out_tok


def _openai_chat(messages: list, max_tokens: int, model: str,
                 temperature: float = 0.7) -> tuple:
    """
    Call OpenAI Chat Completions API.
    Returns (text, input_tokens, output_tokens).
    """
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc

    text = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    in_tok = usage.get("prompt_tokens", _estimate_tokens(str(messages)))
    out_tok = usage.get("completion_tokens", _estimate_tokens(text))
    return text, in_tok, out_tok


def _openrouter_chat(messages: list, max_tokens: int, model: str,
                     temperature: float = 0.7) -> tuple:
    """
    Call OpenRouter Chat Completions API.
    Returns (text, input_tokens, output_tokens).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    # Map short model names to OpenRouter model IDs
    model_map = {
        "mistral-7b-instruct":  "mistralai/mistral-7b-instruct",
        "llama-3-70b-instruct": "meta-llama/llama-3-70b-instruct",
    }
    or_model = model_map.get(model, model)
    payload = {
        "model": or_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://fractalmesh.net",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter API error {exc.code}: {body}") from exc

    text = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    in_tok = usage.get("prompt_tokens", _estimate_tokens(str(messages)))
    out_tok = usage.get("completion_tokens", _estimate_tokens(text))
    return text, in_tok, out_tok


def _get_model_costs(model_id: str) -> tuple:
    """
    Fetch cost_per_1k_input and cost_per_1k_output from aiaas_models.
    Falls back to Claude pricing if not found.
    """
    try:
        conn = _db_connect()
        row = conn.execute(
            "SELECT cost_per_1k_input, cost_per_1k_output FROM aiaas_models WHERE model_id=?",
            (model_id,),
        ).fetchone()
        conn.close()
        if row:
            return row["cost_per_1k_input"], row["cost_per_1k_output"]
    except Exception:
        pass
    return 0.80, 4.00  # default: Claude 3.5 Haiku


def _update_tokens_used(api_key: str, tokens: int):
    try:
        conn = _db_connect()
        conn.execute(
            "UPDATE aiaas_keys SET tokens_used=tokens_used+? WHERE key=?",
            (tokens, api_key),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("Failed to update tokens_used: %s", exc)


# ── request handler ───────────────────────────────────────────────────────────

class AIaaSHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-AIaaS/1.0"

    def log_message(self, fmt, *args):  # noqa: N802
        log.info("%s - %s", self.address_string(), fmt % args)

    # ── response helpers ──────────────────────────────────────────────────────

    def _send_json(self, code: int, body: dict):
        payload = json.dumps(body, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, code: int, message: str):
        self._send_json(code, {"error": message})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ── routing ───────────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._handle_health()
        elif path == "/models":
            self._handle_models()
        elif path == "/docs":
            self._handle_docs()
        elif path == "/admin/usage":
            self._handle_admin_usage()
        elif path == "/v1/usage":
            self._handle_v1_usage()
        else:
            self._send_error(404, f"Not found: {path}")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path == "/admin/keys/create":
            self._handle_admin_keys_create()
        elif path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif path == "/v1/embeddings":
            self._handle_embeddings()
        elif path == "/v1/completions":
            self._handle_completions()
        elif path == "/v1/classify":
            self._handle_classify()
        elif path == "/v1/summarize":
            self._handle_summarize()
        elif path == "/v1/image/generate":
            self._handle_image_generate()
        elif path == "/v1/code":
            self._handle_code()
        else:
            self._send_error(404, f"Not found: {path}")

    # ── GET /health ───────────────────────────────────────────────────────────

    def _handle_health(self):
        self._send_json(200, {
            "status": "ok",
            "service": "fm-aiaas",
            "port": PORT,
        })

    # ── GET /models ───────────────────────────────────────────────────────────

    def _handle_models(self):
        try:
            conn = _db_connect()
            rows = conn.execute(
                "SELECT model_id, provider, display_name, context_length, "
                "cost_per_1k_input, cost_per_1k_output, capabilities, status "
                "FROM aiaas_models ORDER BY provider, model_id"
            ).fetchall()
            conn.close()
            models = [dict(r) for r in rows]
            self._send_json(200, {"models": models, "count": len(models)})
        except Exception as exc:
            self._send_error(500, str(exc))

    # ── GET /docs ─────────────────────────────────────────────────────────────

    def _handle_docs(self):
        docs = {
            "service": "FractalMesh AI-as-a-Service",
            "version": "1.0",
            "auth": "Pass X-API-Key header on all endpoints except /health, /models, /docs",
            "admin_auth": "Pass X-Admin-Secret header on /admin/* endpoints",
            "endpoints": {
                "GET /health": "Service health check",
                "GET /models": "List all available models",
                "GET /docs": "This documentation",
                "POST /admin/keys/create": {
                    "description": "Create a new customer API key",
                    "auth": "X-Admin-Secret",
                    "body": {"owner": "string", "email": "string", "tier": "free|starter|pro|enterprise"},
                },
                "GET /admin/usage": {
                    "description": "Usage stats per key, revenue estimate, model breakdown",
                    "auth": "X-Admin-Secret",
                },
                "POST /v1/chat/completions": {
                    "description": "OpenAI-compatible chat completions",
                    "body": {
                        "model": "claude-3-5-haiku|gpt-4o-mini|mistral-7b-instruct|llama-3-70b-instruct",
                        "messages": [{"role": "user", "content": "..."}],
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "stream": False,
                    },
                },
                "POST /v1/embeddings": {
                    "description": "Text embeddings",
                    "body": {"model": "bge-small-en|text-embedding-3-small", "input": "text or [array]"},
                },
                "POST /v1/completions": {
                    "description": "Legacy text completions",
                    "body": {"model": "string", "prompt": "string", "max_tokens": 200},
                },
                "POST /v1/classify": {
                    "description": "Zero-shot text classification",
                    "body": {"text": "string", "labels": ["positive", "negative", "neutral"]},
                },
                "POST /v1/summarize": {
                    "description": "Text summarization via BART-Large-CNN",
                    "body": {"text": "string", "max_length": 150, "min_length": 50},
                },
                "POST /v1/image/generate": {
                    "description": "Image generation via Stable Diffusion XL",
                    "body": {"prompt": "string", "width": 512, "height": 512},
                },
                "POST /v1/code": {
                    "description": "Code generation via StarCoder2-7B",
                    "body": {"prompt": "string", "language": "python", "max_tokens": 500},
                },
                "GET /v1/usage": "Current key usage stats",
            },
            "tiers": {
                "free":       {"tokens_month": "100k",  "req_day": 50},
                "starter":    {"tokens_month": "1M",    "req_day": 500,  "price_usd": 29},
                "pro":        {"tokens_month": "10M",   "req_day": 5000, "price_usd": 99},
                "enterprise": {"tokens_month": "unlimited", "req_day": "unlimited"},
            },
        }
        self._send_json(200, docs)

    # ── POST /admin/keys/create ───────────────────────────────────────────────

    def _handle_admin_keys_create(self):
        admin_secret = self.headers.get("X-Admin-Secret", "").strip()
        if not ADMIN_SECRET or admin_secret != ADMIN_SECRET:
            self._send_error(403, "Invalid or missing X-Admin-Secret")
            return
        body = self._read_body()
        owner = body.get("owner", "").strip()
        email = body.get("email", "").strip()
        tier  = body.get("tier", "free").strip().lower()
        if not owner or not email:
            self._send_error(400, "owner and email are required")
            return
        if tier not in TIER_LIMITS:
            self._send_error(400, f"Invalid tier: {tier}. Valid: {list(TIER_LIMITS)}")
            return
        new_key   = f"fmai_{secrets.token_hex(24)}"
        tok_limit = TIER_LIMITS[tier]["token_limit"]
        try:
            conn = _db_connect()
            conn.execute(
                "INSERT INTO aiaas_keys (key, owner, email, tier, token_limit, created_at, last_used) "
                "VALUES (?,?,?,?,?,?,?)",
                (new_key, owner, email, tier, tok_limit, time.time(), time.time()),
            )
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            self._send_error(409, "Key collision — retry")
            return
        except Exception as exc:
            self._send_error(500, str(exc))
            return
        self._send_json(201, {
            "key": new_key,
            "owner": owner,
            "email": email,
            "tier": tier,
            "token_limit": tok_limit,
            "requests_per_day": TIER_LIMITS[tier]["req_day"],
        })

    # ── GET /admin/usage ──────────────────────────────────────────────────────

    def _handle_admin_usage(self):
        admin_secret = self.headers.get("X-Admin-Secret", "").strip()
        if not ADMIN_SECRET or admin_secret != ADMIN_SECRET:
            self._send_error(403, "Invalid or missing X-Admin-Secret")
            return
        try:
            conn = _db_connect()
            keys = [dict(r) for r in conn.execute(
                "SELECT key, owner, email, tier, tokens_used, token_limit, "
                "requests_today, created_at, last_used FROM aiaas_keys ORDER BY created_at DESC"
            ).fetchall()]
            # Revenue estimate: approximate based on tier
            tier_prices = {"free": 0, "starter": 29, "pro": 99, "enterprise": 500}
            revenue_est = sum(tier_prices.get(k["tier"], 0) for k in keys)
            # Model breakdown from requests
            model_stats = [dict(r) for r in conn.execute(
                "SELECT model, COUNT(*) as requests, SUM(input_tokens) as total_input, "
                "SUM(output_tokens) as total_output, SUM(cost_usd) as total_cost "
                "FROM aiaas_requests GROUP BY model ORDER BY requests DESC"
            ).fetchall()]
            total_requests = conn.execute(
                "SELECT COUNT(*) as n FROM aiaas_requests"
            ).fetchone()["n"]
            conn.close()
            self._send_json(200, {
                "keys": keys,
                "key_count": len(keys),
                "monthly_revenue_estimate_usd": revenue_est,
                "total_requests_logged": total_requests,
                "model_breakdown": model_stats,
            })
        except Exception as exc:
            self._send_error(500, str(exc))

    # ── POST /v1/chat/completions ─────────────────────────────────────────────

    def _handle_chat_completions(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        body        = self._read_body()
        model       = body.get("model", "claude-3-5-haiku")
        messages    = body.get("messages", [])
        max_tokens  = int(body.get("max_tokens", 1000))
        temperature = float(body.get("temperature", 0.7))

        if not messages:
            self._send_error(400, "messages array is required")
            return

        t0 = time.time()
        try:
            if model.startswith("claude"):
                text, in_tok, out_tok = _anthropic_chat(messages, max_tokens, model)
            elif model.startswith("gpt"):
                text, in_tok, out_tok = _openai_chat(messages, max_tokens, model, temperature)
            else:
                text, in_tok, out_tok = _openrouter_chat(messages, max_tokens, model, temperature)
        except Exception as exc:
            log.error("chat completions error: %s", exc)
            self._send_error(502, str(exc))
            return

        latency = (time.time() - t0) * 1000
        cost_in, cost_out = _get_model_costs(model)
        cost_usd = (in_tok / 1000 * cost_in + out_tok / 1000 * cost_out) * 3  # 3x markup

        _update_tokens_used(row["key"], in_tok + out_tok)
        _log_request(row["key"], model, "/v1/chat/completions",
                     in_tok, out_tok, latency, cost_usd)

        response = {
            "id": f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
        }
        self._send_json(200, response)

    # ── POST /v1/embeddings ───────────────────────────────────────────────────

    def _handle_embeddings(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        body  = self._read_body()
        model = body.get("model", "bge-small-en")
        inp   = body.get("input", "")

        if isinstance(inp, str):
            inputs_list = [inp]
        elif isinstance(inp, list):
            inputs_list = inp
        else:
            self._send_error(400, "input must be a string or array of strings")
            return

        t0 = time.time()
        try:
            if model == "bge-small-en":
                result = _hf_post(
                    "BAAI/bge-small-en-v1.5",
                    {"inputs": inputs_list},
                )
                # HF returns list of embedding arrays
                if isinstance(result, list):
                    embeddings = result
                else:
                    embeddings = [result]
            elif model.startswith("text-embedding"):
                url = "https://api.openai.com/v1/embeddings"
                payload = {"model": model, "input": inputs_list}
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                data = json.dumps(payload).encode()
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    oe_result = json.loads(resp.read().decode())
                embeddings = [item["embedding"] for item in oe_result["data"]]
            else:
                self._send_error(400, f"Unsupported embedding model: {model}")
                return
        except Exception as exc:
            log.error("embeddings error: %s", exc)
            self._send_error(502, str(exc))
            return

        latency = (time.time() - t0) * 1000
        total_tok = sum(_estimate_tokens(t) for t in inputs_list)
        _update_tokens_used(row["key"], total_tok)
        _log_request(row["key"], model, "/v1/embeddings", total_tok, 0, latency, 0.0)

        data_out = [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ]
        self._send_json(200, {
            "object": "list",
            "data": data_out,
            "model": model,
            "usage": {"prompt_tokens": total_tok, "total_tokens": total_tok},
        })

    # ── POST /v1/completions ──────────────────────────────────────────────────

    def _handle_completions(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        body       = self._read_body()
        model      = body.get("model", "claude-3-5-haiku")
        prompt     = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 200))

        if not prompt:
            self._send_error(400, "prompt is required")
            return

        # Wrap as chat message and delegate to chat logic
        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            if model.startswith("claude"):
                text, in_tok, out_tok = _anthropic_chat(messages, max_tokens, model)
            elif model.startswith("gpt"):
                text, in_tok, out_tok = _openai_chat(messages, max_tokens, model)
            else:
                text, in_tok, out_tok = _openrouter_chat(messages, max_tokens, model)
        except Exception as exc:
            log.error("completions error: %s", exc)
            self._send_error(502, str(exc))
            return

        latency = (time.time() - t0) * 1000
        cost_in, cost_out = _get_model_costs(model)
        cost_usd = (in_tok / 1000 * cost_in + out_tok / 1000 * cost_out) * 3

        _update_tokens_used(row["key"], in_tok + out_tok)
        _log_request(row["key"], model, "/v1/completions",
                     in_tok, out_tok, latency, cost_usd)

        self._send_json(200, {
            "id": f"cmpl-{secrets.token_hex(12)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
        })

    # ── POST /v1/classify ─────────────────────────────────────────────────────

    def _handle_classify(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        body   = self._read_body()
        text   = body.get("text", "")
        labels = body.get("labels", ["positive", "negative", "neutral"])

        if not text:
            self._send_error(400, "text is required")
            return

        t0 = time.time()
        try:
            result = _hf_post(
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                {"inputs": text},
            )
        except Exception as exc:
            log.error("classify error: %s", exc)
            self._send_error(502, str(exc))
            return

        latency = (time.time() - t0) * 1000
        in_tok  = _estimate_tokens(text)
        _update_tokens_used(row["key"], in_tok)
        _log_request(row["key"], "bart-large-cnn", "/v1/classify",
                     in_tok, 0, latency, 0.0)

        # HF returns list of [{"label":..., "score":...}, ...] or nested list
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                scores = result[0]
            else:
                scores = result
        else:
            scores = []

        # Map HF labels to user-supplied labels where possible
        all_labels = [{"label": s.get("label", ""), "score": round(s.get("score", 0.0), 4)}
                      for s in scores]
        top = all_labels[0] if all_labels else {"label": "unknown", "score": 0.0}

        self._send_json(200, {
            "label": top["label"],
            "score": top["score"],
            "all_labels": all_labels,
            "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        })

    # ── POST /v1/summarize ────────────────────────────────────────────────────

    def _handle_summarize(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        body       = self._read_body()
        text       = body.get("text", "")
        max_length = int(body.get("max_length", 150))
        min_length = int(body.get("min_length", 50))

        if not text:
            self._send_error(400, "text is required")
            return

        t0 = time.time()
        try:
            result = _hf_post(
                "facebook/bart-large-cnn",
                {
                    "inputs": text,
                    "parameters": {"max_length": max_length, "min_length": min_length},
                },
            )
        except Exception as exc:
            log.error("summarize error: %s", exc)
            self._send_error(502, str(exc))
            return

        latency = (time.time() - t0) * 1000
        in_tok  = _estimate_tokens(text)
        _update_tokens_used(row["key"], in_tok)
        _log_request(row["key"], "bart-large-cnn", "/v1/summarize",
                     in_tok, 0, latency, 0.0)

        # HF returns [{"summary_text": "..."}]
        if isinstance(result, list) and len(result) > 0:
            summary = result[0].get("summary_text", "")
        elif isinstance(result, dict):
            summary = result.get("summary_text", str(result))
        else:
            summary = str(result)

        self._send_json(200, {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "model": "facebook/bart-large-cnn",
        })

    # ── POST /v1/image/generate ───────────────────────────────────────────────

    def _handle_image_generate(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        body   = self._read_body()
        prompt = body.get("prompt", "")
        width  = int(body.get("width", 512))
        height = int(body.get("height", 512))

        if not prompt:
            self._send_error(400, "prompt is required")
            return

        t0 = time.time()
        try:
            image_bytes = _hf_post_binary(
                "stabilityai/stable-diffusion-xl-base-1.0",
                {"inputs": prompt},
            )
        except Exception as exc:
            log.error("image generate error: %s", exc)
            self._send_error(502, str(exc))
            return

        latency   = (time.time() - t0) * 1000
        ts        = int(time.time() * 1000)
        img_path  = IMAGES_DIR / f"{ts}.png"
        try:
            img_path.write_bytes(image_bytes)
        except Exception as exc:
            self._send_error(500, f"Failed to save image: {exc}")
            return

        in_tok = _estimate_tokens(prompt)
        _update_tokens_used(row["key"], in_tok)
        _log_request(row["key"], "sdxl-base-1.0", "/v1/image/generate",
                     in_tok, 0, latency, 0.0)

        self._send_json(200, {
            "image_path": str(img_path),
            "prompt": prompt,
            "size": f"{width}x{height}",
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
        })

    # ── POST /v1/code ─────────────────────────────────────────────────────────

    def _handle_code(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        body       = self._read_body()
        prompt     = body.get("prompt", "")
        language   = body.get("language", "python")
        max_tokens = int(body.get("max_tokens", 500))

        if not prompt:
            self._send_error(400, "prompt is required")
            return

        hf_input = f"# {language}\n{prompt}\n"
        t0 = time.time()
        try:
            result = _hf_post(
                "bigcode/starcoder2-7b",
                {
                    "inputs": hf_input,
                    "parameters": {"max_new_tokens": max_tokens},
                },
            )
        except Exception as exc:
            log.error("code gen error: %s", exc)
            self._send_error(502, str(exc))
            return

        latency = (time.time() - t0) * 1000
        # HF returns [{"generated_text": "..."}]
        if isinstance(result, list) and len(result) > 0:
            generated = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated = result.get("generated_text", str(result))
        else:
            generated = str(result)

        # Strip the echo of the input prefix if present
        if generated.startswith(hf_input):
            code = generated[len(hf_input):]
        else:
            code = generated

        in_tok  = _estimate_tokens(hf_input)
        out_tok = _estimate_tokens(code)
        _update_tokens_used(row["key"], in_tok + out_tok)
        _log_request(row["key"], "starcoder2-7b", "/v1/code",
                     in_tok, out_tok, latency, 0.0)

        self._send_json(200, {
            "code": code,
            "language": language,
            "model": "bigcode/starcoder2-7b",
        })

    # ── GET /v1/usage ─────────────────────────────────────────────────────────

    def _handle_v1_usage(self):
        row, err = _check_auth(self)
        if err:
            self._send_error(401, err)
            return
        # Re-fetch fresh row (requests_today was already incremented by _check_auth)
        try:
            conn = _db_connect()
            fresh = conn.execute(
                "SELECT tokens_used, token_limit, requests_today, tier FROM aiaas_keys WHERE key=?",
                (row["key"],),
            ).fetchone()
            conn.close()
        except Exception as exc:
            self._send_error(500, str(exc))
            return
        self._send_json(200, {
            "tokens_used":     fresh["tokens_used"],
            "token_limit":     fresh["token_limit"],
            "requests_today":  fresh["requests_today"],
            "tier":            fresh["tier"],
        })


# ── startup ───────────────────────────────────────────────────────────────────

def _run():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), AIaaSHandler)
    log.info("FractalMesh AI-as-a-Service listening on port %d", PORT)
    log.info("DB: %s", DB)
    log.info("Images: %s", IMAGES_DIR)

    def _shutdown(sig, frame):  # noqa: ARG001
        log.info("Shutting down AIaaS (signal %d)", sig)
        server.server_close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    except SystemExit:
        pass


if __name__ == "__main__":
    _run()
