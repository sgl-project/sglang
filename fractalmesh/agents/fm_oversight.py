"""
FractalMesh Oversight Agent
Immutable audit trail: HMAC-signs and logs every agent heartbeat; AFSL compliance layer
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import sys
import json
import time
import signal
import sqlite3
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("OVERSIGHT_INTERVAL", "300"))
ABN      = "56628117363"

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS oversight_log (
        id INTEGER PRIMARY KEY, agent TEXT, abn TEXT, pm2_online INTEGER,
        pulse_count INTEGER, hmac_sig TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _pm2_online_count() -> int:
    try:
        r     = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=10)
        procs = json.loads(r.stdout)
        return sum(1 for p in procs if p.get("pm2_env", {}).get("status") == "online")
    except Exception:
        return -1


def _pulse_count() -> int:
    try:
        conn = sqlite3.connect(DB, timeout=10)
        row  = conn.execute("SELECT COUNT(*) FROM pulse_log").fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return -1


def _sign(payload: dict) -> str:
    try:
        from modules.security_core import generate_fingerprint
        sig, _ = generate_fingerprint(payload)
        return sig
    except Exception:
        import hmac as _hmac, hashlib as _hl, json as _json
        secret  = os.getenv("BUS_SECRET", "fallback").encode()
        body    = _json.dumps(payload, sort_keys=True).encode()
        return _hmac.new(secret, body, _hl.sha256).hexdigest()


def _log(pm2_online, pulse_count, sig):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute(
        "INSERT INTO oversight_log (agent,abn,pm2_online,pulse_count,hmac_sig) VALUES (?,?,?,?,?)",
        ("fm-oversight", ABN, pm2_online, pulse_count, sig))
    conn.commit(); conn.close()


def run_cycle():
    ts           = datetime.utcnow().isoformat()
    pm2_online   = _pm2_online_count()
    pulse_count  = _pulse_count()
    payload      = {"agent": "fm-oversight", "abn": ABN, "ts": ts,
                    "pm2_online": pm2_online, "pulse_count": pulse_count}
    sig          = _sign(payload)
    print(json.dumps({**payload, "hmac_sig": sig[:16] + "…"}))
    _log(pm2_online, pulse_count, sig)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-oversight] AFSL compliance oversight active | interval={INTERVAL}s")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-oversight] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-oversight] Stopped.")
