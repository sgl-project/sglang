"""
FractalMesh Healer Agent
Self-healing: monitors PM2 process list, restarts stopped agents, logs to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import signal
import sqlite3
import subprocess
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("HEALER_INTERVAL", "60"))
DRY_RUN  = os.getenv("ENABLE_HEALER_RESTARTS", "false").lower() != "true"

CRITICAL_NODES = [
    "fm-bus", "fm-pulse-bus", "fm-gitops-runner", "fm-integrator",
    "fm-stripe-mon", "fm-harmonic", "fm-warden", "fm-sovereign-ops",
    "fm-lba-bridge", "fm-samsung-warden", "fm-email-listener",
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS healer_log (
        id INTEGER PRIMARY KEY, node TEXT, action TEXT, prev_status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _pm2_status() -> dict:
    try:
        r = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return {}
        procs = json.loads(r.stdout)
        return {p["name"]: p.get("pm2_env", {}).get("status", "unknown") for p in procs}
    except Exception:
        return {}


def _restart(node: str):
    try:
        subprocess.run(["pm2", "restart", node], capture_output=True, timeout=15)
    except Exception as e:
        print(f"   [healer] restart failed for {node}: {e}")


def _log(node, action, prev_status):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO healer_log (node,action,prev_status) VALUES (?,?,?)",
                 (node, action, prev_status))
    conn.commit(); conn.close()


def run_cycle():
    status_map = _pm2_status()
    ts         = datetime.utcnow().isoformat()
    healed     = 0
    for node in CRITICAL_NODES:
        status = status_map.get(node, "missing")
        if status not in ("online", "stopping"):
            action = "restart" if not DRY_RUN else "dry_restart"
            print(f"   [healer] {node} → {status} → {action}")
            if not DRY_RUN:
                _restart(node)
            _log(node, action, status)
            healed += 1
    total = len(status_map)
    print(f"[fm-healer] {ts} | pm2={total} procs | healed={healed} | dry={DRY_RUN}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-healer] Active | interval={INTERVAL}s | dry={DRY_RUN} | watching={len(CRITICAL_NODES)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-healer] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-healer] Stopped.")
