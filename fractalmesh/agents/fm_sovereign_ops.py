#!/usr/bin/env python3
"""
fm_sovereign_ops.py — ABN/ATO authority heartbeat
Logs operator identity and authorisation state; HMAC-signs each pulse.
TFN is NEVER read or logged by this agent.
Samuel James Hiotis | ABN 56628117363
"""
import os
import sys
import json
import time
import hmac
import signal
import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SOVEREIGN-OPS] %(message)s")
log = logging.getLogger("sovereign_ops")

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
NAME     = os.getenv("NAME", "Samuel James Hiotis")
ABN      = os.getenv("ABN",  "56628117363")
INTERVAL = int(os.getenv("SOVEREIGN_OPS_INTERVAL", "3600"))
_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS sovereign_ops_log (
        id INTEGER PRIMARY KEY, principal TEXT, abn TEXT,
        status TEXT, hmac_sig TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _sign(payload: dict) -> str:
    secret = os.getenv("BUS_SECRET", "fallback").encode()
    body   = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def _log(principal, abn, status, sig):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO sovereign_ops_log (principal,abn,status,hmac_sig) VALUES (?,?,?,?)""",
                 (principal, abn, status, sig))
    conn.execute("""INSERT INTO pulse_log (source,event,priority)
        SELECT 'fm-sovereign-ops','authority_heartbeat',1.618
        WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='pulse_log')""")
    conn.commit(); conn.close()


def authorise():
    ts  = datetime.now(timezone.utc).isoformat()
    out = {"agent": "fm-sovereign-ops", "principal": NAME,
           "abn": ABN, "status": "AUTHORIZED", "ts": ts}
    sig = _sign(out)
    out["hmac_sig"] = sig[:16] + "…"
    print(json.dumps(out), flush=True)
    log.info("Principal: %s | ABN: %s | AUTHORIZED | sig=%s…", NAME, ABN, sig[:12])
    _log(NAME, ABN, "AUTHORIZED", sig)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    log.info("fm-sovereign-ops online | interval=%ds", INTERVAL)
    while _running:
        try:
            authorise()
        except Exception as e:
            log.error("ERR %s", e)
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    log.info("fm-sovereign-ops stopped.")
