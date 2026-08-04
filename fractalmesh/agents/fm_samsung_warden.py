#!/usr/bin/env python3
"""
fm_samsung_warden.py — Samsung/Android device health guardian
Monitors Android node connectivity via ADB; logs to sovereign.db
Samuel James Hiotis | ABN 56628117363
"""
import os
import json
import time
import signal
import sqlite3
import logging
import subprocess
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SAMSUNG-WARDEN] %(message)s")
log = logging.getLogger("samsung_warden")

ROOT               = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB                 = os.path.join(ROOT, "database", "sovereign.db")
HEARTBEAT_INTERVAL = int(os.getenv("SAMSUNG_HEARTBEAT_INTERVAL", "300"))

DEVICES = [
    {"id": "samsung-a51",    "addr": os.getenv("ANDROID_ADDR_01", ""), "label": "Samsung A51"},
    {"id": "samsung-s20",    "addr": os.getenv("ANDROID_ADDR_02", ""), "label": "Samsung S20"},
    {"id": "android-tablet", "addr": os.getenv("ANDROID_ADDR_03", ""), "label": "Android Tablet"},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS android_health (
        id INTEGER PRIMARY KEY, device_id TEXT, label TEXT, addr TEXT,
        status TEXT, battery_pct INTEGER, uptime_s INTEGER,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _adb_probe(addr: str) -> dict:
    if not addr:
        return {"status": "no_addr", "battery": -1, "uptime": -1}
    try:
        bat_r = subprocess.run(
            ["adb", "-s", addr, "shell", "dumpsys battery | grep level"],
            capture_output=True, text=True, timeout=8)
        bat = -1
        if bat_r.returncode == 0:
            for part in bat_r.stdout.split():
                try:
                    bat = int(part); break
                except ValueError:
                    pass
        up_r = subprocess.run(
            ["adb", "-s", addr, "shell", "cat /proc/uptime"],
            capture_output=True, text=True, timeout=8)
        uptime = -1
        if up_r.returncode == 0:
            try:
                uptime = int(float(up_r.stdout.split()[0]))
            except Exception:
                pass
        return {"status": "online", "battery": bat, "uptime": uptime}
    except FileNotFoundError:
        return {"status": "adb_not_installed", "battery": -1, "uptime": -1}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "battery": -1, "uptime": -1}
    except Exception as e:
        return {"status": f"err:{e}", "battery": -1, "uptime": -1}


def _log(d_id, label, addr, status, battery, uptime):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO android_health (device_id,label,addr,status,battery_pct,uptime_s) VALUES (?,?,?,?,?,?)",
                 (d_id, label, addr, status, battery, uptime))
    conn.commit(); conn.close()


def check_devices():
    ts      = datetime.now(timezone.utc).isoformat()
    results = []
    for d in DEVICES:
        probe = _adb_probe(d["addr"])
        results.append({**d, **probe})
        bat_str = f"bat={probe['battery']}%" if probe["battery"] >= 0 else ""
        up_str  = f"up={probe['uptime']}s"   if probe["uptime"]  >= 0 else ""
        log.info("%s [%s] %s %s %s", d["label"], d["id"], probe["status"], bat_str, up_str)
        _log(d["id"], d["label"], d["addr"], probe["status"], probe["battery"], probe["uptime"])
    out = {"agent": "fm-samsung-warden", "devices": results, "ts": ts}
    print(json.dumps(out), flush=True)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    log.info("fm-samsung-warden online | interval=%ds | devices=%d", HEARTBEAT_INTERVAL, len(DEVICES))
    while _running:
        try:
            check_devices()
        except Exception as e:
            log.error("ERR %s", e)
        for _ in range(HEARTBEAT_INTERVAL):
            if not _running: break
            time.sleep(1)
    log.info("fm-samsung-warden stopped.")
