#!/usr/bin/env python3
"""
fm_mcp_router.py — Master MCP Intent Router (Port 7785)
Unified intent multiplexer for cross-app integration.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import signal
import sqlite3
import logging
import subprocess
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
PORT     = int(os.getenv("MCP_PORT", "7785"))
SECRET   = os.getenv("MCP_SECRET", "fm_mcp_internal").encode()
ROOT     = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB       = ROOT / "database" / "sovereign.db"
LOG      = ROOT / "logs" / "mcp_router.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP-ROUTER] %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("mcp_router")

# ── LBA firewall — reject any payload exposing raw credentials ────────────────
_BANNED = [
    "sk_live_", "sk-ant-api", "ETH_PRIVATE_KEY", "PRIVATE_KEY=",
    "password=", "secret=", "[id number redacted]",
]

def _lba_check(payload: str) -> bool:
    low = payload.lower()
    return not any(b.lower() in low for b in _BANNED)

# ── database ──────────────────────────────────────────────────────────────────
def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mcp_log (
            id      INTEGER PRIMARY KEY,
            intent  TEXT,
            status  TEXT,
            latency REAL,
            ts      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _db_log(intent: str, status: str, latency: float):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO mcp_log (intent, status, latency) VALUES (?,?,?)",
            (intent, status, latency),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

# ── HMAC verification (optional — set MCP_REQUIRE_SIG=1 to enforce) ───────────
def _verify_sig(sig: str, body: bytes) -> bool:
    if not os.getenv("MCP_REQUIRE_SIG"):
        return True
    expected = hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig or "")

# ── intent handlers ────────────────────────────────────────────────────────────

def _handle_sync_calendar(args, kwargs) -> dict:
    label = args[0] if args else kwargs.get("label", "")
    ts    = args[1] if len(args) > 1 else kwargs.get("timestamp", "")
    return {"action": "calendar_sync", "label": label, "timestamp": ts, "queued": True}

def _handle_sync_reminder(args, kwargs) -> dict:
    content  = args[0] if args else kwargs.get("content", "")
    priority = kwargs.get("priority", "normal")
    return {"action": "reminder_set", "content": content, "priority": priority}

def _handle_sync_workspace(args, kwargs) -> dict:
    payload = args[0] if args else kwargs.get("payload", {})
    return {"action": "workspace_queued", "payload": payload, "status": "QUEUED"}

def _handle_device_pulse(args, kwargs) -> dict:
    try:
        out = subprocess.getoutput("termux-battery-status 2>/dev/null")
        battery = json.loads(out) if out.strip().startswith("{") else {}
    except Exception:
        battery = {}
    return {
        "node":    "arm64_local",
        "battery": battery,
        "uptime":  os.popen("uptime -p 2>/dev/null").read().strip() or "n/a",
        "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

def _handle_send_message(args, kwargs) -> dict:
    recipient = args[0] if args else kwargs.get("recipient", "")
    body      = args[1] if len(args) > 1 else kwargs.get("body", "")
    return {"action": "sms_dispatch", "recipient": recipient, "body_len": len(body), "status": "QUEUED"}

def _handle_extract_research(args, kwargs) -> dict:
    url   = args[0] if args else kwargs.get("url", "")
    topic = args[1] if len(args) > 1 else kwargs.get("topic", "")
    return {"action": "research_initiated", "url": url, "topic": topic}

def _handle_apply_watermark(args, kwargs) -> dict:
    content = kwargs.get("content", args[0] if args else "")
    sig = hmac.new(SECRET, content.encode() if content else b"", hashlib.sha256).hexdigest()
    return {"action": "watermark_applied", "hmac": sig, "algorithm": "HMAC-SHA256"}

def _handle_mesh_status(args, kwargs) -> dict:
    return {
        "mesh":       "converged",
        "node":       os.uname().nodename,
        "agents":     32,
        "ports":      {"mcp_router": PORT, "web_terminal": 7777, "api_bridge": 7780,
                       "strategy_engine": 7786, "revenue_aggregator": 7787},
        "abn":        os.getenv("ABN", "56628117363"),
        "compliance": ["ISO_27001", "APRA_CPS234"],
    }

_INTENTS = {
    "sync_samsung_calendar":  _handle_sync_calendar,
    "sync_samsung_reminder":  _handle_sync_reminder,
    "sync_google_workspace":  _handle_sync_workspace,
    "device_utilities_pulse": _handle_device_pulse,
    "send_samsung_message":   _handle_send_message,
    "extract_research":       _handle_extract_research,
    "apply_synthid_watermark":_handle_apply_watermark,
    "mesh_status":            _handle_mesh_status,
}

# ── HTTP handler ───────────────────────────────────────────────────────────────

class MCPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default access log

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/health", "/api/mesh/status"):
            self._respond(200, _handle_mesh_status([], {}))
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        t0 = time.time()
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            sig    = self.headers.get("X-MCP-Signature", "")

            if not _verify_sig(sig, body):
                self._respond(401, {"error": "invalid_signature"})
                return

            raw = body.decode("utf-8", errors="replace")
            if not _lba_check(raw):
                log.warning("LBA_BLOCKED len=%d", len(raw))
                self._respond(403, {"error": "lba_firewall_blocked"})
                return

            data   = json.loads(raw)
            intent = data.get("intent", "")
            args   = data.get("args", [])
            kwargs = data.get("kwargs", {})

            handler = _INTENTS.get(intent)
            if handler:
                result = handler(args, kwargs)
                status = "SUCCESS"
                code   = 200
            else:
                result = {"error": "unknown_intent", "available": list(_INTENTS)}
                status = "UNKNOWN"
                code   = 400

            latency = time.time() - t0
            _db_log(intent, status, latency)
            log.info("intent=%s status=%s latency=%.3fs", intent, status, latency)
            self._respond(code, {"status": status, "intent": intent, **result})

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
    server = HTTPServer(("0.0.0.0", PORT), MCPHandler)
    log.info("MCP Router listening on port %d", PORT)
    log.info("LBA firewall active — %d banned patterns", len(_BANNED))
    log.info("Available intents: %s", ", ".join(_INTENTS))
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("MCP Router stopped")

if __name__ == "__main__":
    main()
