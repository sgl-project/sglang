"""
FractalMesh WiGLE Oracle Agent
Queries WiGLE API for local WiFi network telemetry; logs to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import signal
import sqlite3
import base64
import urllib.request
import urllib.parse
from datetime import datetime

ROOT      = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB        = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL  = int(os.getenv("WIGLE_INTERVAL", "3600"))
DRY_RUN   = os.getenv("ENABLE_WIGLE_QUERY", "false").lower() != "true"

WIGLE_API_NAME   = os.getenv("WIGLE_API_NAME", "")
WIGLE_API_TOKEN  = os.getenv("WIGLE_API_TOKEN", "")
WIGLE_LAT        = float(os.getenv("WIGLE_LAT", "-36.0737"))   # Albury NSW
WIGLE_LON        = float(os.getenv("WIGLE_LON", "146.9135"))
WIGLE_RADIUS     = float(os.getenv("WIGLE_RADIUS_KM", "1.0"))

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS wigle_telemetry_ledger (
        id INTEGER PRIMARY KEY, ssid TEXT, bssid TEXT UNIQUE,
        lat REAL, lon REAL, signal_dbm INTEGER, encryption TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _wigle_search() -> list:
    if not WIGLE_API_NAME or not WIGLE_API_TOKEN or DRY_RUN:
        return []
    try:
        creds  = base64.b64encode(f"{WIGLE_API_NAME}:{WIGLE_API_TOKEN}".encode()).decode()
        params = urllib.parse.urlencode({
            "latrange1": WIGLE_LAT - WIGLE_RADIUS / 111,
            "latrange2": WIGLE_LAT + WIGLE_RADIUS / 111,
            "longrange1": WIGLE_LON - WIGLE_RADIUS / 111,
            "longrange2": WIGLE_LON + WIGLE_RADIUS / 111,
            "resultsPerPage": 50,
        })
        req = urllib.request.Request(
            f"https://api.wigle.net/api/v2/network/search?{params}",
            headers={"Authorization": f"Basic {creds}",
                     "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get("results", [])
    except Exception as e:
        print(f"   [wigle] query err: {e}")
        return []


def _upsert(net: dict):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO wigle_telemetry_ledger (ssid,bssid,lat,lon,signal_dbm,encryption)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(bssid) DO UPDATE SET ts=CURRENT_TIMESTAMP""",
        (net.get("ssid", "")[:64], net.get("netid", ""),
         net.get("trilat", 0.0), net.get("trilong", 0.0),
         net.get("rssi", 0), net.get("encryption", "")))
    conn.commit(); conn.close()


def run_cycle():
    ts      = datetime.utcnow().isoformat()
    results = _wigle_search()
    print(f"[fm-wigle-oracle] {ts} | lat={WIGLE_LAT} lon={WIGLE_LON} r={WIGLE_RADIUS}km | results={len(results)} | dry={DRY_RUN}")
    for net in results[:10]:
        print(f"   → {net.get('ssid','?'):<32} {net.get('netid',''):<18} {net.get('encryption','')}")
        _upsert(net)
    if DRY_RUN:
        print("   → dry_run: WiGLE query disabled (set ENABLE_WIGLE_QUERY=true + credentials)")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-wigle-oracle] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-wigle-oracle] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-wigle-oracle] Stopped.")
