"""
FractalMesh Toolkit Agent
Shared utility runner: env validation, vault check, disk health, port scan
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import sys
import time
import signal
import socket
import shutil
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("TOOLKIT_INTERVAL", "900"))   # 15 minutes

VAULT_PATH    = os.path.expanduser("~/.secrets/fractal.env")
REQUIRED_VARS = ["BUS_SECRET", "GMAIL_USER", "STRIPE_SECRET_KEY"]
REQUIRED_DIRS = ["database", "dist", "backups", "logs"]

CHECK_PORTS = [
    {"port": 5057, "label": "health-api"},
    {"port": 5060, "label": "pulse-bus"},
    {"port": 8090, "label": "integrator"},
    {"port": 8091, "label": "stripe-mon"},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS toolkit_log (
        id INTEGER PRIMARY KEY, check_name TEXT, result TEXT,
        detail TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log(check, result, detail=""):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO toolkit_log (check_name,result,detail) VALUES (?,?,?)",
                 (check, result, detail[:200]))
    conn.commit(); conn.close()


def _check_vault() -> str:
    if not os.path.exists(VAULT_PATH):
        return "MISSING"
    mode = oct(os.stat(VAULT_PATH).st_mode)[-3:]
    return "ok" if mode == "600" else f"perms:{mode}"


def _check_dirs() -> list:
    missing = []
    for d in REQUIRED_DIRS:
        path = os.path.join(ROOT, d)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            missing.append(d)
    return missing


def _check_vars() -> list:
    return [v for v in REQUIRED_VARS if not os.getenv(v)]


def _check_port(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except Exception:
        return False


def _disk_pct() -> float:
    try:
        total, used, free = shutil.disk_usage(ROOT)
        return round(used / total * 100, 1)
    except Exception:
        return -1.0


def run_cycle():
    ts = datetime.utcnow().isoformat()
    print(f"[fm-toolkit] {ts}")

    vault_status = _check_vault()
    print(f"   vault      {vault_status}")
    _log("vault", vault_status)

    missing_vars = _check_vars()
    vars_status  = "ok" if not missing_vars else f"missing:{','.join(missing_vars)}"
    print(f"   env_vars   {vars_status}")
    _log("env_vars", vars_status, vars_status)

    missing_dirs = _check_dirs()
    dirs_status  = "ok" if not missing_dirs else f"created:{','.join(missing_dirs)}"
    print(f"   dirs       {dirs_status}")
    _log("dirs", dirs_status)

    disk = _disk_pct()
    disk_status = "ok" if 0 <= disk < 85 else f"WARN:{disk}%"
    print(f"   disk       {disk}% used → {disk_status}")
    _log("disk", disk_status, f"{disk}%")

    for p in CHECK_PORTS:
        up = _check_port(p["port"])
        status = "up" if up else "down"
        print(f"   port:{p['port']:<5} {p['label']:<14} {status}")
        _log(f"port:{p['port']}", status, p["label"])


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-toolkit] Active | interval={INTERVAL}s")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-toolkit] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-toolkit] Stopped.")
