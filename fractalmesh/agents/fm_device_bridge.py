"""
FractalMesh Device Bridge Agent
Monitors Android + Raspberry Pi nodes via ADB/SSH; logs health to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import sqlite3
import subprocess
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("DEVICE_BRIDGE_INTERVAL", "300"))

DEVICES = [
    {"id": "android-primary", "type": "android", "addr": os.getenv("ANDROID_ADDR", ""),  "proto": "adb"},
    {"id": "pi4-node-01",     "type": "rpi",     "addr": os.getenv("PI4_ADDR_01", ""),   "proto": "ssh"},
    {"id": "pi4-node-02",     "type": "rpi",     "addr": os.getenv("PI4_ADDR_02", ""),   "proto": "ssh"},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS device_health (
        id INTEGER PRIMARY KEY, device_id TEXT, device_type TEXT,
        status TEXT, latency_ms REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _probe(d: dict) -> tuple:
    addr = d["addr"]
    if not addr:
        return "no_addr", -1.0
    t0 = time.monotonic()
    try:
        if d["proto"] == "adb":
            r = subprocess.run(["adb", "-s", addr, "shell", "echo ok"],
                               capture_output=True, timeout=8)
        else:
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                                 addr, "echo ok"], capture_output=True, timeout=8)
        ms     = round((time.monotonic() - t0) * 1000, 1)
        status = "ok" if r.returncode == 0 else f"{d['proto']}_err"
        return status, ms
    except Exception as e:
        return f"err:{e}", -1.0


def _log(d_id, d_type, status, ms):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO device_health (device_id,device_type,status,latency_ms) VALUES (?,?,?,?)",
                 (d_id, d_type, status, ms))
    conn.commit(); conn.close()


def run_cycle():
    print(f"[fm-device-bridge] {datetime.utcnow().isoformat()} | {len(DEVICES)} devices")
    for d in DEVICES:
        status, ms = _probe(d)
        print(f"   → {d['id']:<18} [{d['type']}] {status:<12} {f'{ms:.1f}ms' if ms >= 0 else 'n/a'}")
        _log(d["id"], d["type"], status, ms)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-device-bridge] Active | interval={INTERVAL}s")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-device-bridge] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-device-bridge] Stopped.")
