#!/usr/bin/env python3
"""
fm_revenue_aggregator.py — Revenue Aggregation & Reporting Agent (Port 7787)
Polls Stripe, affiliate dashboards, and internal agents to produce a unified
revenue snapshot. All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import logging
import os
import signal
import sqlite3
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("REVENUE_PORT", "7787"))
ROOT         = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
LOG          = ROOT / "logs" / "revenue_aggregator.log"
STRIPE_KEY   = os.getenv("STRIPE_SECRET_KEY", "")
MCP_URL      = os.getenv("MCP_URL", "http://127.0.0.1:7785")
STRATEGY_URL = os.getenv("STRATEGY_URL", "http://127.0.0.1:7786")

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REVENUE] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("revenue_aggregator")

# ── revenue channels config ───────────────────────────────────────────────────
CHANNELS = [
    {
        "id": "stripe",
        "name": "Stripe Payments",
        "type": "api",
        "url": "https://api.stripe.com/v1/balance",
        "auth_env": "STRIPE_SECRET_KEY",
        "currency": "AUD",
    },
    {
        "id": "manus_affiliate",
        "name": "Manus Affiliate (30% recurring)",
        "type": "manual",
        "ref_code": os.getenv("MANUS_REF_CODE", "XDCMWO3VETC7FV"),
        "commission": 0.30,
        "currency": "USD",
    },
    {
        "id": "blofin_affiliate",
        "name": "BloFin Futures Affiliate (50% rev-share)",
        "type": "manual",
        "commission": 0.50,
        "currency": "USD",
    },
    {
        "id": "together_ai",
        "name": "Together AI API Credits",
        "type": "api",
        "auth_env": "TOGETHER_API_KEY",
        "currency": "USD",
    },
    {
        "id": "depin_streamr",
        "name": "Streamr DePIN Node",
        "type": "depin",
        "protocol": "streamr",
        "currency": "DATA",
    },
    {
        "id": "devto_content",
        "name": "Dev.to Content Revenue",
        "type": "content",
        "auth_env": "DEVTO_API_KEY",
        "currency": "USD",
    },
]

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS revenue_snapshots (
            id         INTEGER PRIMARY KEY,
            channel_id TEXT,
            amount     REAL,
            currency   TEXT,
            period     TEXT,
            raw        TEXT,
            ts         DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS revenue_targets (
            channel_id TEXT PRIMARY KEY,
            daily_target REAL,
            monthly_target REAL,
            currency TEXT
        )
    """)
    conn.commit()
    conn.close()

def _db_log_snapshot(channel_id: str, amount: float, currency: str, period: str, raw: dict):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO revenue_snapshots (channel_id,amount,currency,period,raw) VALUES (?,?,?,?,?)",
            (channel_id, amount, currency, period, json.dumps(raw)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_snapshot error: %s", e)

def _db_get_totals() -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute("""
            SELECT channel_id, SUM(amount) as total, currency,
                   MAX(ts) as last_update, COUNT(*) as snapshots
            FROM revenue_snapshots
            WHERE ts > datetime('now','-30 days')
            GROUP BY channel_id, currency
            ORDER BY total DESC
        """).fetchall()
        conn.close()
        return [
            {"channel_id": r[0], "total_30d": r[1], "currency": r[2],
             "last_update": r[3], "snapshots": r[4]}
            for r in rows
        ]
    except Exception:
        return []

# ── channel polling ────────────────────────────────────────────────────────────

def _poll_stripe() -> dict:
    key = os.getenv("STRIPE_SECRET_KEY", "")
    if not key or not key.startswith("sk_"):
        return {"available": False, "reason": "no_key"}
    try:
        req = urllib.request.Request(
            "https://api.stripe.com/v1/balance",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        available = sum(b["amount"] for b in data.get("available", [])) / 100
        pending   = sum(b["amount"] for b in data.get("pending", []))   / 100
        return {"available": available, "pending": pending, "livemode": data.get("livemode")}
    except Exception as e:
        return {"available": False, "error": str(e)}

def _poll_internal_agent(url: str, path: str = "/health") -> dict:
    try:
        with urllib.request.urlopen(f"{url}{path}", timeout=3) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"reachable": False, "error": str(e)}

def _aggregate_all() -> dict:
    t0 = time.time()
    snapshot = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "period": "realtime",
        "channels": {},
        "totals_30d": [],
        "mesh_health": {},
    }

    # Stripe
    stripe_data = _poll_stripe()
    snapshot["channels"]["stripe"] = stripe_data
    if isinstance(stripe_data.get("available"), float):
        _db_log_snapshot("stripe", stripe_data["available"], "AUD", "realtime", stripe_data)

    # Internal agent health
    snapshot["mesh_health"]["mcp_router"]      = _poll_internal_agent(MCP_URL, "/health")
    snapshot["mesh_health"]["strategy_engine"] = _poll_internal_agent(STRATEGY_URL, "/health")

    # Manual / affiliate channels (status only — amounts entered manually or via webhook)
    for ch in CHANNELS:
        if ch["type"] == "manual":
            snapshot["channels"][ch["id"]] = {
                "name": ch["name"],
                "type": "manual",
                "commission": ch.get("commission"),
                "currency": ch.get("currency"),
                "note": "manual_entry_required",
            }

    # 30-day aggregated totals from DB
    snapshot["totals_30d"] = _db_get_totals()
    snapshot["latency_ms"] = round((time.time() - t0) * 1000, 1)

    return snapshot

# ── HTTP handler ───────────────────────────────────────────────────────────────

class RevenueHandler(BaseHTTPRequestHandler):
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
            self._respond(200, {"status": "ok", "channels": len(CHANNELS)})

        elif self.path in ("/", "/snapshot"):
            self._respond(200, _aggregate_all())

        elif self.path == "/channels":
            self._respond(200, {"channels": CHANNELS})

        elif self.path == "/totals":
            self._respond(200, {"totals_30d": _db_get_totals()})

        elif self.path == "/stripe":
            self._respond(200, _poll_stripe())

        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        # Manual revenue entry: {"channel_id": "...", "amount": 100.0, "currency": "AUD", "period": "day"}
        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))
            cid    = data.get("channel_id", "")
            amount = float(data.get("amount", 0))
            cur    = data.get("currency", "AUD")
            period = data.get("period", "manual")

            if not cid:
                self._respond(400, {"error": "channel_id required"})
                return

            _db_log_snapshot(cid, amount, cur, period, data)
            log.info("manual_entry channel=%s amount=%.2f %s", cid, amount, cur)
            self._respond(200, {"status": "logged", "channel_id": cid,
                                "amount": amount, "currency": cur})
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
    server = HTTPServer(("0.0.0.0", PORT), RevenueHandler)
    log.info("Revenue Aggregator listening on port %d", PORT)
    log.info("Tracking %d revenue channels", len(CHANNELS))
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Revenue Aggregator stopped")

if __name__ == "__main__":
    main()
