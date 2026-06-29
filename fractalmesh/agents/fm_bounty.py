"""
FractalMesh Bounty Agent
Tracks open tasks with φ-weighted priority scoring; logs to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("BOUNTY_INTERVAL", "1800"))
PHI      = 1.6180339887

BOUNTIES = [
    {"id": "infra-audit",     "title": "Full infrastructure audit",          "aud": 750.0, "priority": 1},
    {"id": "compliance-map",  "title": "NCC 2025 + SOC 2 control mapping",   "aud": 450.0, "priority": 2},
    {"id": "agent-hardening", "title": "Harden all 26 agents for prod",      "aud": 300.0, "priority": 1},
    {"id": "stripe-test",     "title": "End-to-end Stripe webhook test",     "aud": 150.0, "priority": 2},
    {"id": "devto-publish",   "title": "Publish 8 Dev.to articles",          "aud": 100.0, "priority": 3},
    {"id": "supabase-sync",   "title": "Validate Supabase real-time sync",   "aud": 200.0, "priority": 2},
    {"id": "tunnel-stable",   "title": "CF tunnel 99.9% uptime validation",  "aud": 250.0, "priority": 1},
    {"id": "android-health",  "title": "Android node health dashboard",      "aud": 175.0, "priority": 3},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS bounty_log (
        id INTEGER PRIMARY KEY, bounty_id TEXT, title TEXT,
        aud_value REAL, phi_score REAL, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _score(b: dict) -> float:
    return round((b["aud"] * PHI) / b["priority"], 4)


def _log(b: dict, status: str):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO bounty_log (bounty_id,title,aud_value,phi_score,status) VALUES (?,?,?,?,?)",
                 (b["id"], b["title"], b["aud"], _score(b), status))
    conn.commit(); conn.close()


def run_cycle():
    total = sum(b["aud"] for b in BOUNTIES)
    print(f"[fm-bounty] {datetime.utcnow().isoformat()} | {len(BOUNTIES)} bounties | AUD ${total:.2f}")
    for b in sorted(BOUNTIES, key=_score, reverse=True):
        print(f"   → [{b['id']:<20}] ${b['aud']:>6.2f} | φ={_score(b):.2f} | P{b['priority']}")
        _log(b, "tracked")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-bounty] Active | interval={INTERVAL}s")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-bounty] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-bounty] Stopped.")
