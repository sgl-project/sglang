#!/usr/bin/env python3
"""
fm_zapier_bridge.py — Zapier Webhook Bridge (Port 7788)
Receives inbound Zapier webhooks and fires outbound Zap triggers.
All credentials sourced from ~/.secrets/fractal.env at runtime.
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
PORT             = int(os.getenv("ZAPIER_PORT", "7788"))
ROOT             = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB               = ROOT / "database" / "sovereign.db"
LOG              = ROOT / "logs" / "zapier_bridge.log"
ZAPIER_SECRET    = os.getenv("ZAPIER_WEBHOOK_SECRET", "")
MCP_URL          = os.getenv("MCP_URL", "http://127.0.0.1:7785")

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ZAPIER] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("zapier_bridge")

# ── registered Zap trigger URLs (outbound) ────────────────────────────────────
# Set in vault: ZAPIER_ZAP_<NAME>=https://hooks.zapier.com/hooks/catch/...
_ZAP_REGISTRY: dict[str, str] = {}

def _load_zap_registry():
    global _ZAP_REGISTRY
    for k, v in os.environ.items():
        if k.startswith("ZAPIER_ZAP_") and v.startswith("https://hooks.zapier.com"):
            name = k[len("ZAPIER_ZAP_"):].lower()
            _ZAP_REGISTRY[name] = v
    log.info("Loaded %d outbound Zap endpoints", len(_ZAP_REGISTRY))

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS zapier_events (
            id          INTEGER PRIMARY KEY,
            direction   TEXT,
            zap_name    TEXT,
            payload     TEXT,
            status      TEXT,
            latency_ms  REAL,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _db_log(direction: str, zap_name: str, payload: dict, status: str, latency_ms: float):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO zapier_events (direction,zap_name,payload,status,latency_ms) VALUES (?,?,?,?,?)",
            (direction, zap_name, json.dumps(payload)[:2000], status, latency_ms),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

# ── outbound trigger ──────────────────────────────────────────────────────────

def _fire_zap(zap_name: str, payload: dict) -> dict:
    t0  = time.time()
    url = _ZAP_REGISTRY.get(zap_name.lower())
    if not url:
        return {"error": f"zap_not_registered:{zap_name}",
                "available": list(_ZAP_REGISTRY.keys())}
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(data))},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body   = resp.read().decode("utf-8", errors="replace")
            status = "fired"
    except urllib.error.URLError as e:
        body   = str(e)
        status = "failed"

    latency = (time.time() - t0) * 1000
    _db_log("outbound", zap_name, payload, status, latency)
    log.info("zap_fire name=%s status=%s latency=%.0fms", zap_name, status, latency)
    return {"status": status, "zap": zap_name, "response": body[:300]}

# ── inbound event router ──────────────────────────────────────────────────────

def _route_inbound(event_type: str, payload: dict) -> dict:
    """Route inbound Zapier events to appropriate MCP intents or local handlers."""
    t0 = time.time()

    route_map = {
        "new_stripe_payment":   ("sync_google_workspace", {"payload": payload}),
        "new_form_submission":  ("sync_samsung_reminder", {"content": str(payload), "priority": "high"}),
        "new_email":            ("sync_samsung_calendar", {"label": payload.get("subject",""), "timestamp": payload.get("date","")}),
        "content_published":    ("extract_research",     {"url": payload.get("url",""), "topic": payload.get("title","")}),
        "new_lead":             ("send_samsung_message", {"recipient": payload.get("phone",""), "body": f"New lead: {payload.get('name','')}"}),
    }

    if event_type in route_map:
        intent, kwargs = route_map[event_type]
        mcp_payload = json.dumps({"intent": intent, "args": [], "kwargs": kwargs}).encode()
        try:
            req = urllib.request.Request(
                f"{MCP_URL}/",
                data=mcp_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                mcp_result = json.loads(r.read())
        except Exception as e:
            mcp_result = {"error": str(e)}
        status = "routed"
    else:
        mcp_result = {"note": "no_route_matched"}
        status     = "unrouted"

    latency = (time.time() - t0) * 1000
    _db_log("inbound", event_type, payload, status, latency)
    log.info("zap_inbound type=%s status=%s", event_type, status)
    return {"status": status, "event_type": event_type, "mcp": mcp_result}

# ── HTTP handler ───────────────────────────────────────────────────────────────

class ZapierHandler(BaseHTTPRequestHandler):
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
            self._respond(200, {"status": "ok", "zaps": len(_ZAP_REGISTRY),
                                "endpoints": list(_ZAP_REGISTRY.keys())})
        elif self.path == "/zaps":
            self._respond(200, {"zaps": list(_ZAP_REGISTRY.keys()), "count": len(_ZAP_REGISTRY)})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        t0 = time.time()
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body.decode("utf-8", errors="replace"))

            # Inbound webhook from Zapier
            if self.path.startswith("/webhook"):
                event_type = data.get("event_type", data.get("type", "unknown"))
                result = _route_inbound(event_type, data)
                self._respond(200, result)

            # Outbound — fire a registered Zap
            elif self.path.startswith("/fire"):
                zap_name = data.get("zap")
                if not zap_name:
                    self._respond(400, {"error": "zap field required"})
                    return
                payload_to_send = data.get("payload", data)
                result = _fire_zap(zap_name, payload_to_send)
                code   = 200 if result.get("status") == "fired" else 502
                self._respond(code, result)

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
    _load_zap_registry()
    server = HTTPServer(("0.0.0.0", PORT), ZapierHandler)
    log.info("Zapier Bridge listening on port %d", PORT)
    log.info("Inbound: POST /webhook | Outbound: POST /fire | Registry: GET /zaps")
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Zapier Bridge stopped")

if __name__ == "__main__":
    main()
