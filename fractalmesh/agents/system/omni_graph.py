#!/usr/bin/env python3
"""
omni_graph.py — Omni-graph state renderer
Builds a live JSON snapshot of all PM2 agent states and sovereign.db table counts
FractalMesh Omega Titan | Samuel James Hiotis | ABN 56628117363
"""
import json
import time
import signal
import sys
import os
import sqlite3
import subprocess
from datetime import datetime, timezone

ROOT    = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB      = os.path.join(ROOT, "database", "sovereign.db")
running = True


def _stop(*_a):
    global running
    running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def _pm2_snapshot() -> list:
    try:
        r     = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=10)
        procs = json.loads(r.stdout)
        return [{"name": p["name"],
                 "status": p.get("pm2_env", {}).get("status", "?"),
                 "restarts": p.get("pm2_env", {}).get("restart_time", 0)}
                for p in procs]
    except Exception:
        return []


def _db_counts() -> dict:
    tables = ["pulse_log", "leads", "revenue", "affiliate_log",
              "domain_log", "device_health", "enochian_log"]
    counts = {}
    if not os.path.exists(DB):
        return counts
    try:
        conn = sqlite3.connect(DB, timeout=5)
        for t in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                counts[t] = row[0] if row else 0
            except Exception:
                pass
        conn.close()
    except Exception:
        pass
    return counts


print("[omni-graph] online", flush=True)
while running:
    procs  = _pm2_snapshot()
    counts = _db_counts()
    online = sum(1 for p in procs if p["status"] == "online")
    payload = {
        "agent":       "omni-graph",
        "status":      "online",
        "pm2_total":   len(procs),
        "pm2_online":  online,
        "pm2_agents":  procs,
        "db_counts":   counts,
        "ts":          datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload), flush=True)
    time.sleep(60)

sys.exit(0)
