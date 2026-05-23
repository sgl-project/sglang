#!/usr/bin/env python3
"""
fm_openrouter.py — OpenRouter LLM Routing Agent (Port 7791)
Smart model selection, cost tracking, fallback chains, and response caching.
Routes to cheapest capable model; escalates on failure.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hashlib
import signal
import sqlite3
import logging
import urllib.request
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT            = int(os.getenv("OPENROUTER_PORT", "7791"))
ROOT            = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB              = ROOT / "database" / "sovereign.db"
LOG             = ROOT / "logs" / "openrouter.log"
OR_KEY          = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENROUTER_KEY", ""))
OR_BASE         = "https://openrouter.ai/api/v1"
CACHE_TTL       = int(os.getenv("OR_CACHE_TTL", "3600"))  # 1 hr default
SITE_URL        = os.getenv("SITE_URL", "https://fractalmesh.net")
SITE_NAME       = os.getenv("SITE_NAME", "FractalMesh Omega Titan")

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OPENROUTER] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("openrouter")

# ── model tiers (cheapest→most capable) ──────────────────────────────────────
MODEL_TIERS = {
    "fast": [
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
    ],
    "balanced": [
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "anthropic/claude-haiku-4-5",
    ],
    "premium": [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4o-mini",
        "google/gemini-flash-1.5",
    ],
    "apex": [
        "anthropic/claude-opus-4-7",
        "openai/gpt-4o",
        "google/gemini-pro-1.5",
    ],
}

# Task → default tier mapping
TASK_TIERS = {
    "classify":    "fast",
    "summarize":   "fast",
    "extract":     "balanced",
    "draft":       "balanced",
    "code":        "premium",
    "analyse":     "premium",
    "reason":      "apex",
    "strategy":    "apex",
}

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS or_requests (
            id           INTEGER PRIMARY KEY,
            task         TEXT,
            model        TEXT,
            tier         TEXT,
            prompt_tok   INTEGER,
            completion_tok INTEGER,
            cost_usd     REAL,
            latency_ms   REAL,
            cached       INTEGER,
            status       TEXT,
            ts           DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS or_cache (
            cache_key    TEXT PRIMARY KEY,
            response     TEXT,
            model        TEXT,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS or_cost_summary (
            model        TEXT PRIMARY KEY,
            total_usd    REAL DEFAULT 0,
            call_count   INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def _db_log(task: str, model: str, tier: str, prompt_tok: int,
            compl_tok: int, cost_usd: float, latency_ms: float,
            cached: bool, status: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO or_requests (task,model,tier,prompt_tok,completion_tok,cost_usd,latency_ms,cached,status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (task, model, tier, prompt_tok, compl_tok, cost_usd, latency_ms, int(cached), status),
        )
        conn.execute("""
            INSERT INTO or_cost_summary (model,total_usd,call_count,total_tokens)
            VALUES (?,?,1,?)
            ON CONFLICT(model) DO UPDATE SET
                total_usd=total_usd+excluded.total_usd,
                call_count=call_count+1,
                total_tokens=total_tokens+excluded.total_tokens
        """, (model, cost_usd, prompt_tok + compl_tok))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

def _cache_get(key: str) -> str | None:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        row  = conn.execute(
            "SELECT response FROM or_cache WHERE cache_key=? "
            "AND created_at > datetime('now',?)",
            (key, f"-{CACHE_TTL} seconds"),
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None

def _cache_set(key: str, response: str, model: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT OR REPLACE INTO or_cache (cache_key,response,model) VALUES (?,?,?)",
            (key, response, model),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("cache_set error: %s", e)

def _cost_summary() -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute(
            "SELECT model,total_usd,call_count,total_tokens FROM or_cost_summary ORDER BY total_usd DESC"
        ).fetchall()
        conn.close()
        return [{"model": r[0], "total_usd": round(r[1], 6),
                 "calls": r[2], "tokens": r[3]} for r in rows]
    except Exception:
        return []

# ── OpenRouter API ────────────────────────────────────────────────────────────

def _or_chat(model: str, messages: list, max_tokens: int = 1024,
             temperature: float = 0.7) -> dict:
    if not OR_KEY:
        return {"error": "OPENROUTER_API_KEY not configured"}
    payload = json.dumps({
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{OR_BASE}/chat/completions",
        data=payload,
        headers={
            "Authorization":  f"Bearer {OR_KEY}",
            "Content-Type":   "application/json",
            "HTTP-Referer":   SITE_URL,
            "X-Title":        SITE_NAME,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"http_{e.code}", "detail": body[:300]}
    except Exception as e:
        return {"error": str(e)}

def _list_models() -> dict:
    if not OR_KEY:
        return {"error": "no_key", "tiers": MODEL_TIERS}
    req = urllib.request.Request(
        f"{OR_BASE}/models",
        headers={"Authorization": f"Bearer {OR_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

# ── smart routing ─────────────────────────────────────────────────────────────

def _route(task: str, prompt: str, tier: str | None,
           max_tokens: int, temperature: float,
           use_cache: bool, system: str = "") -> dict:
    tier      = tier or TASK_TIERS.get(task, "balanced")
    models    = MODEL_TIERS.get(tier, MODEL_TIERS["balanced"])
    cache_key = hashlib.sha256(f"{tier}:{prompt}:{max_tokens}".encode()).hexdigest()

    if use_cache:
        cached = _cache_get(cache_key)
        if cached:
            log.info("cache_hit task=%s tier=%s", task, tier)
            _db_log(task, "cache", tier, 0, 0, 0.0, 0.0, True, "cache_hit")
            return {"task": task, "tier": tier, "model": "cache", "cached": True,
                    "content": cached}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for model in models:
        t0     = time.time()
        result = _or_chat(model, messages, max_tokens, temperature)
        latency = (time.time() - t0) * 1000

        if "error" in result:
            last_error = result["error"]
            log.warning("model_failed model=%s error=%s — trying next", model, last_error)
            continue

        usage   = result.get("usage", {})
        pt      = usage.get("prompt_tokens", 0)
        ct      = usage.get("completion_tokens", 0)
        # OpenRouter cost in USD is sometimes in usage.cost
        cost    = usage.get("cost", 0.0)
        content = result["choices"][0]["message"]["content"] if result.get("choices") else ""

        if use_cache and content:
            _cache_set(cache_key, content, model)
        _db_log(task, model, tier, pt, ct, cost, latency, False, "ok")
        log.info("routed task=%s model=%s tier=%s latency=%.0fms cost=$%.6f",
                 task, model, tier, latency, cost)
        return {
            "task": task, "tier": tier, "model": model, "cached": False,
            "content": content, "usage": {"prompt_tokens": pt, "completion_tokens": ct},
            "cost_usd": cost, "latency_ms": round(latency, 1),
        }

    _db_log(task, "all_failed", tier, 0, 0, 0.0, 0.0, False, "failed")
    return {"error": "all_models_failed", "last_error": last_error, "tier": tier}

# ── HTTP handler ───────────────────────────────────────────────────────────────

class OpenRouterHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "api_key": bool(OR_KEY),
                                "tiers": list(MODEL_TIERS.keys()),
                                "task_defaults": TASK_TIERS})
        elif self.path == "/models":
            self._respond(200, _list_models())
        elif self.path == "/tiers":
            self._respond(200, MODEL_TIERS)
        elif self.path == "/cost":
            self._respond(200, {"cost_summary": _cost_summary()})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))

            if self.path in ("/chat", "/route", "/generate"):
                prompt     = data.get("prompt", data.get("message", ""))
                task       = data.get("task", "draft")
                tier       = data.get("tier")
                max_tokens = int(data.get("max_tokens", 1024))
                temperature= float(data.get("temperature", 0.7))
                use_cache  = bool(data.get("cache", True))
                system     = data.get("system", "")

                if not prompt:
                    self._respond(400, {"error": "prompt required"})
                    return
                result = _route(task, prompt, tier, max_tokens, temperature, use_cache, system)
                code   = 200 if "error" not in result else 502
                self._respond(code, result)

            # Passthrough — raw OpenRouter chat completions format
            elif self.path == "/v1/chat/completions":
                model    = data.get("model", MODEL_TIERS["balanced"][0])
                messages = data.get("messages", [])
                max_tok  = int(data.get("max_tokens", 1024))
                temp     = float(data.get("temperature", 0.7))
                t0       = time.time()
                result   = _or_chat(model, messages, max_tok, temp)
                latency  = (time.time() - t0) * 1000
                usage    = result.get("usage", {})
                _db_log("passthrough", model, "passthrough",
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("cost", 0.0), latency, False,
                        "error" if "error" in result else "ok")
                self._respond(200 if "choices" in result else 502, result)

            else:
                self._respond(404, {"error": "unknown_path"})

        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
        except Exception as e:
            log.error("handler_error: %s", e)
            self._respond(500, {"error": str(e)})

# ── main ───────────────────────────────────────────────────────────────────────

_running = True

def _shutdown(*_):
    global _running
    log.info("shutdown signal — exiting cleanly")
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), OpenRouterHandler)
    log.info("OpenRouter agent listening on port %d", PORT)
    log.info("API key: %s | Tiers: %s | Cache TTL: %ds",
             "configured" if OR_KEY else "fallback",
             ", ".join(MODEL_TIERS.keys()), CACHE_TTL)
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("OpenRouter agent stopped")

if __name__ == "__main__":
    main()
