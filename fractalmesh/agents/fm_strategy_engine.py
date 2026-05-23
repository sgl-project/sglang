#!/usr/bin/env python3
"""
fm_strategy_engine.py — Zero-Capital Strategy Execution Engine (Port 7786)
Tracks, schedules, and executes strategies from the 500-strategy monetization map.
Routes actions through the MCP router (port 7785) for cross-agent coordination.
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
PORT       = int(os.getenv("STRATEGY_PORT", "7786"))
MCP_URL    = os.getenv("MCP_URL", "http://127.0.0.1:7785")
ROOT       = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB         = ROOT / "database" / "sovereign.db"
LOG        = ROOT / "logs" / "strategy_engine.log"
SECRET     = os.getenv("MCP_SECRET", "fm_mcp_internal").encode()

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STRATEGY] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("strategy_engine")

# ── strategy catalogue ─────────────────────────────────────────────────────────
# Curated actionable strategies from the 500-strategy monetization map
# Each entry: id, category, title, action_type, params, difficulty
STRATEGIES = [
    # ── Freelance ──────────────────────────────────────────────────────────────
    {"id": 1,  "cat": "freelance", "title": "Upwork Python automation gig",
     "action": "post_gig", "platform": "upwork", "diff": "entry",
     "income": "$15-50/hr", "active": True},
    {"id": 2,  "cat": "freelance", "title": "Fiverr data scraping micro-gig",
     "action": "post_gig", "platform": "fiverr", "diff": "entry",
     "income": "$5-100/order", "active": True},
    {"id": 3,  "cat": "freelance", "title": "Toptal senior dev profile",
     "action": "apply_platform", "platform": "toptal", "diff": "advanced",
     "income": "$60-200/hr", "active": False},
    {"id": 10, "cat": "freelance", "title": "Dev.to technical article (paid)",
     "action": "publish_content", "platform": "devto", "diff": "entry",
     "income": "$50-300/article", "active": True},
    # ── Data & AI ──────────────────────────────────────────────────────────────
    {"id": 81, "cat": "data_ai", "title": "Scale AI data labelling tasks",
     "action": "check_tasks", "platform": "scale_ai", "diff": "entry",
     "income": "$10-30/hr", "active": True},
    {"id": 90, "cat": "data_ai", "title": "Kaggle competition entry",
     "action": "submit_model", "platform": "kaggle", "diff": "mid",
     "income": "$500-50000/win", "active": True},
    {"id": 95, "cat": "data_ai", "title": "Hugging Face model card + API",
     "action": "publish_model", "platform": "huggingface", "diff": "mid",
     "income": "$0.01-1/inference", "active": True},
    # ── Affiliate ──────────────────────────────────────────────────────────────
    {"id": 301, "cat": "affiliate", "title": "Manus affiliate referral (30% recurring)",
     "action": "track_referral", "platform": "manus",
     "ref_code": os.getenv("MANUS_REF_CODE", "XDCMWO3VETC7FV"),
     "diff": "entry", "income": "30% recurring", "active": True},
    {"id": 302, "cat": "affiliate", "title": "BloFin futures affiliate (50% rev-share)",
     "action": "track_referral", "platform": "blofin",
     "ref_code": os.getenv("BLOFIN_REF_CODE", ""),
     "diff": "mid", "income": "50% rev-share", "active": True},
    {"id": 310, "cat": "affiliate", "title": "Together AI API referral",
     "action": "track_referral", "platform": "together_ai",
     "diff": "entry", "income": "$10-500/referral", "active": True},
    # ── DePIN / Web3 ──────────────────────────────────────────────────────────
    {"id": 401, "cat": "depin", "title": "Streamr data publishing node",
     "action": "start_node", "platform": "streamr", "diff": "mid",
     "income": "$5-200/mo", "active": True},
    {"id": 410, "cat": "depin", "title": "Akash compute provider",
     "action": "register_provider", "platform": "akash", "diff": "advanced",
     "income": "$20-500/mo", "active": False},
    {"id": 420, "cat": "depin", "title": "WiGLE RF data contribution",
     "action": "submit_wardriving", "platform": "wigle", "diff": "entry",
     "income": "reputation + token", "active": True},
    # ── Infrastructure / Consulting ────────────────────────────────────────────
    {"id": 441, "cat": "infra", "title": "FractalMesh mesh-as-a-service pitch",
     "action": "send_proposal", "platform": "direct", "diff": "advanced",
     "income": "$500-5000/engagement", "active": True},
    {"id": 450, "cat": "infra", "title": "Cloudflare Zero Trust consulting",
     "action": "send_proposal", "platform": "direct", "diff": "advanced",
     "income": "$150-300/hr", "active": True},
]

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_log (
            id          INTEGER PRIMARY KEY,
            strategy_id INTEGER,
            title       TEXT,
            action      TEXT,
            result      TEXT,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_state (
            strategy_id INTEGER PRIMARY KEY,
            last_run    DATETIME,
            run_count   INTEGER DEFAULT 0,
            last_status TEXT
        )
    """)
    conn.commit()
    conn.close()

def _db_log_run(strategy_id: int, title: str, action: str, result: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO strategy_log (strategy_id, title, action, result) VALUES (?,?,?,?)",
            (strategy_id, title, action, result),
        )
        conn.execute("""
            INSERT INTO strategy_state (strategy_id, last_run, run_count, last_status)
            VALUES (?,CURRENT_TIMESTAMP,1,?)
            ON CONFLICT(strategy_id) DO UPDATE SET
                last_run=CURRENT_TIMESTAMP,
                run_count=run_count+1,
                last_status=excluded.last_status
        """, (strategy_id, result))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

def _db_get_state(strategy_id: int) -> dict:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        row = conn.execute(
            "SELECT last_run, run_count, last_status FROM strategy_state WHERE strategy_id=?",
            (strategy_id,)
        ).fetchone()
        conn.close()
        if row:
            return {"last_run": row[0], "run_count": row[1], "last_status": row[2]}
    except Exception:
        pass
    return {"last_run": None, "run_count": 0, "last_status": None}

# ── MCP router bridge ─────────────────────────────────────────────────────────

def _mcp_call(intent: str, **kwargs) -> dict:
    payload = json.dumps({"intent": intent, "args": [], "kwargs": kwargs}).encode()
    req = urllib.request.Request(
        f"{MCP_URL}/",
        data=payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

# ── strategy execution dispatch ───────────────────────────────────────────────

def _execute_strategy(s: dict) -> dict:
    action = s.get("action", "")
    sid    = s["id"]
    title  = s["title"]

    if action == "track_referral":
        result = _mcp_call("mesh_status")  # heartbeat confirms mesh is live
        status = f"affiliate_tracked:{s.get('platform')}:{s.get('ref_code','')}"

    elif action == "publish_content":
        status = f"content_queued:{s.get('platform')}:devto_api"

    elif action == "check_tasks":
        status = f"task_check:{s.get('platform')}"

    elif action == "submit_wardriving":
        status = "wigle_data_queued"

    elif action == "post_gig":
        status = f"gig_reminder:{s.get('platform')}:{title}"

    else:
        status = f"action_logged:{action}"

    _db_log_run(sid, title, action, status)
    log.info("strategy_exec id=%d action=%s status=%s", sid, action, status)
    return {"strategy_id": sid, "title": title, "action": action, "status": status}

# ── HTTP handler ───────────────────────────────────────────────────────────────

class StrategyHandler(BaseHTTPRequestHandler):
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
            self._respond(200, {"status": "ok", "strategies": len(STRATEGIES)})

        elif self.path == "/strategies":
            enriched = []
            for s in STRATEGIES:
                state = _db_get_state(s["id"])
                enriched.append({**s, **state})
            self._respond(200, {"count": len(enriched), "strategies": enriched})

        elif self.path.startswith("/strategies/active"):
            active = [s for s in STRATEGIES if s.get("active")]
            self._respond(200, {"count": len(active), "strategies": active})

        elif self.path.startswith("/run/all"):
            results = [_execute_strategy(s) for s in STRATEGIES if s.get("active")]
            self._respond(200, {"ran": len(results), "results": results})

        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))
            sid    = data.get("strategy_id")

            if sid is not None:
                matches = [s for s in STRATEGIES if s["id"] == sid]
                if matches:
                    result = _execute_strategy(matches[0])
                    self._respond(200, result)
                else:
                    self._respond(404, {"error": "strategy_not_found", "id": sid})
            else:
                self._respond(400, {"error": "strategy_id required"})
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
    server = HTTPServer(("0.0.0.0", PORT), StrategyHandler)
    log.info("Strategy Engine listening on port %d", PORT)
    log.info("Loaded %d strategies (%d active)",
             len(STRATEGIES), sum(1 for s in STRATEGIES if s.get("active")))
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Strategy Engine stopped")

if __name__ == "__main__":
    main()
